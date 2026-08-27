"""Third-party read-only data providers.

Dexscreener is the backbone because it is free, keyless, multi-chain and
indexes new pairs within seconds. Helius and Birdeye are optional accelerators
for the two things Dexscreener does not expose: an exact holder count and
candle history.

Every method here degrades to None or an empty list instead of raising. A dead
provider should cost you one feature, not the tick - and critically, a missing
feature must read as "unknown" downstream, never as a favourable value. Silently
substituting zero for an unknown holder count is how a bot ends up buying a
token with four holders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .log import get
from .models import Candidate, Candle, Chain, now_ms
from .net import Http, HttpError

log = get("providers")

DEXSCREENER = "https://api.dexscreener.com"

# Dexscreener's chain slugs. Robinhood Chain is absent as of 2026-08; those
# candidates come from the chain's own RPC/subgraph instead of being faked.
DEX_CHAIN_SLUG: dict[Chain, str] = {
    Chain.SOLANA: "solana",
    Chain.BNB: "bsc",
    Chain.BASE: "base",
}
SLUG_TO_CHAIN = {v: k for k, v in DEX_CHAIN_SLUG.items()}


@dataclass(slots=True)
class PairSnapshot:
    chain: Chain
    pair_address: str
    token_address: str
    symbol: str
    name: str
    price_usd: float
    liquidity_usd: float
    mcap_usd: float
    volume_m5: float
    volume_h1: float
    buys_m5: int
    sells_m5: int
    price_change_m5: float
    price_change_h1: float
    created_at_ms: int
    dex_id: str = ""
    quote_symbol: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_candidate(self, source: str) -> Candidate:
        return Candidate(
            chain=self.chain,
            address=self.token_address,
            symbol=self.symbol,
            name=self.name,
            created_at_ms=self.created_at_ms,
            price_usd=self.price_usd,
            liquidity_usd=self.liquidity_usd,
            mcap_usd=self.mcap_usd,
            volume_5m_usd=self.volume_m5,
            pool_address=self.pair_address,
            quote_asset=self.quote_symbol,
            source=source,
        )


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_pair(pair: dict[str, Any]) -> PairSnapshot | None:
    slug = pair.get("chainId", "")
    chain = SLUG_TO_CHAIN.get(slug)
    if chain is None:
        return None
    base = pair.get("baseToken") or {}
    txns = (pair.get("txns") or {}).get("m5") or {}
    return PairSnapshot(
        chain=chain,
        pair_address=pair.get("pairAddress", ""),
        token_address=base.get("address", ""),
        symbol=base.get("symbol", ""),
        name=base.get("name", ""),
        price_usd=_f(pair.get("priceUsd")),
        liquidity_usd=_f((pair.get("liquidity") or {}).get("usd")),
        mcap_usd=_f(pair.get("marketCap") or pair.get("fdv")),
        volume_m5=_f((pair.get("volume") or {}).get("m5")),
        volume_h1=_f((pair.get("volume") or {}).get("h1")),
        buys_m5=int(_f(txns.get("buys"))),
        sells_m5=int(_f(txns.get("sells"))),
        price_change_m5=_f((pair.get("priceChange") or {}).get("m5")) / 100.0,
        price_change_h1=_f((pair.get("priceChange") or {}).get("h1")) / 100.0,
        created_at_ms=int(_f(pair.get("pairCreatedAt"))),
        dex_id=pair.get("dexId", ""),
        quote_symbol=(pair.get("quoteToken") or {}).get("symbol", ""),
        raw=pair,
    )


class Dexscreener:
    def __init__(self, http: Http) -> None:
        self.http = http
        # Documented as 300 req/min for the pairs endpoints. Staying under it
        # deliberately, because being rate-limited during a launch is the one
        # moment the data is worth anything.
        http.limit("api.dexscreener.com", rate_per_sec=4.0, burst=8)

    async def _get(self, path: str, **kw: Any) -> Any:
        try:
            return await self.http.get(f"{DEXSCREENER}{path}", **kw)
        except HttpError as exc:
            log.warning("dexscreener failed", extra={"path": path, "status": exc.status})
            return None

    async def token_pairs(self, addresses: list[str]) -> list[PairSnapshot]:
        """Best pair per token, for up to 30 tokens in one call."""
        if not addresses:
            return []
        data = await self._get("/latest/dex/tokens/" + ",".join(addresses[:30]))
        pairs = (data or {}).get("pairs") or []
        best: dict[str, PairSnapshot] = {}
        for raw in pairs:
            snap = parse_pair(raw)
            if snap is None or not snap.token_address:
                continue
            current = best.get(snap.token_address)
            if current is None or snap.liquidity_usd > current.liquidity_usd:
                best[snap.token_address] = snap
        return list(best.values())

    async def search(self, query: str) -> list[PairSnapshot]:
        data = await self._get("/latest/dex/search", params={"q": query})
        return [s for s in (parse_pair(p) for p in (data or {}).get("pairs") or []) if s]

    async def new_profiles(self) -> list[str]:
        """Tokens that just created a Dexscreener profile.

        A profile means somebody bothered to fill in socials, which is a weak
        but real filter against pure noise launches - and it lands earlier than
        the token shows up in any trending list.
        """
        data = await self._get("/token-profiles/latest/v1")
        out = []
        for item in data or []:
            slug, address = item.get("chainId"), item.get("tokenAddress")
            if slug in SLUG_TO_CHAIN and address:
                out.append(address)
        return out

    async def boosted(self) -> list[str]:
        """Paid-boost tokens. Included as a signal about the *promoter*, not the
        token: someone spending on visibility expects an audience, and the
        audience is what the strategy is trying to arrive before."""
        data = await self._get("/token-boosts/latest/v1")
        return [
            item["tokenAddress"]
            for item in data or []
            if item.get("chainId") in SLUG_TO_CHAIN and item.get("tokenAddress")
        ]

    async def pair(self, chain: Chain, pair_address: str) -> PairSnapshot | None:
        slug = DEX_CHAIN_SLUG.get(chain)
        if not slug:
            return None
        data = await self._get(f"/latest/dex/pairs/{slug}/{pair_address}")
        pairs = (data or {}).get("pairs") or []
        return parse_pair(pairs[0]) if pairs else None


class Helius:
    """Optional. Used for the one number nothing else gives away for free: the
    exact holder count."""

    def __init__(self, http: Http, api_key: str) -> None:
        self.http = http
        self.api_key = api_key
        self.url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def holder_count(self, mint: str) -> int | None:
        if not self.enabled:
            return None
        try:
            data = await self.http.post(
                self.url,
                json_body={
                    "jsonrpc": "2.0",
                    "id": "holders",
                    "method": "getTokenAccounts",
                    "params": {"mint": mint, "limit": 1, "options": {"showZeroBalance": False}},
                },
            )
        except HttpError as exc:
            log.debug("helius holder_count failed", extra={"status": exc.status})
            return None
        result = (data or {}).get("result") or {}
        total = result.get("total")
        return int(total) if isinstance(total, (int, float)) else None


class Birdeye:
    """Optional. Candle history for tokens old enough to have any."""

    def __init__(self, http: Http, api_key: str) -> None:
        self.http = http
        self.api_key = api_key
        http.limit("public-api.birdeye.so", rate_per_sec=1.0, burst=3)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def candles(
        self, address: str, *, chain: Chain = Chain.SOLANA, minutes: int = 60
    ) -> list[Candle]:
        if not self.enabled:
            return []
        now_s = now_ms() // 1000
        try:
            data = await self.http.get(
                "https://public-api.birdeye.so/defi/ohlcv",
                params={
                    "address": address,
                    "type": "1m",
                    "time_from": now_s - minutes * 60,
                    "time_to": now_s,
                },
                headers={
                    "X-API-KEY": self.api_key,
                    "x-chain": DEX_CHAIN_SLUG.get(chain, "solana"),
                },
            )
        except HttpError as exc:
            log.debug("birdeye candles failed", extra={"status": exc.status})
            return []
        items = ((data or {}).get("data") or {}).get("items") or []
        out: list[Candle] = []
        for item in items:
            out.append(
                Candle(
                    ts=int(_f(item.get("unixTime")) * 1000),
                    open=_f(item.get("o")),
                    high=_f(item.get("h")),
                    low=_f(item.get("l")),
                    close=_f(item.get("c")),
                    volume=_f(item.get("v")),
                )
            )
        out.sort(key=lambda c: c.ts)
        return out
