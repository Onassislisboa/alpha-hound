"""Feature assembly.

One entry point - `Enricher.enrich` - turns a cheap Candidate into the full
feature vector the model scores. It is the only place that knows which
providers exist, so adding a data source does not ripple through the engine.

Two rules hold throughout:

1. A measurement that failed is recorded as unknown, never as a favourable
   default. `Enrichment.unknown` is what the gate layer reads to decide whether
   it is allowed to have an opinion at all.
2. Cheap features first. Nothing calls the chain until the free Dexscreener
   snapshot has already had a chance to disqualify the candidate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

from ..log import get
from ..models import Candidate, Candle, Chain, Features, RoundTrip, now_ms
from ..settings import Config
from ..store import Store
from . import chart, flow, terminals
from .distribution import HolderStats, analyze, holder_growth
from .terminals import Attribution, ShareTracker, TerminalRegistry

if TYPE_CHECKING:
    # Providers and the chain reader pull in httpx. They are only ever passed
    # in, never constructed here, so importing them lazily keeps the pure
    # feature maths usable without the HTTP stack installed.
    from ..providers import Birdeye, Dexscreener, Helius, PairSnapshot
    from .solana import MintInfo, SolanaReader

log = get("signals")

CostProbe = Callable[[Candidate, float], Awaitable[RoundTrip]]


@dataclass(slots=True)
class Enrichment:
    candidate: Candidate
    features: Features
    holders: HolderStats = field(default_factory=HolderStats)
    mint: MintInfo | None = None
    attribution: Attribution = field(default_factory=Attribution)
    round_trip: RoundTrip = field(default_factory=lambda: RoundTrip(ok=False, note="not probed"))
    snap: PairSnapshot | None = None
    candles: list[Candle] = field(default_factory=list)
    trades: list[flow.Trade] = field(default_factory=list)
    snipers: set[str] = field(default_factory=set)
    # Names of features that could not be measured. Gates that depend on an
    # unknown feature abstain rather than pass.
    unknown: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


class Enricher:
    def __init__(
        self,
        *,
        store: Store,
        strategy: Config,
        dexscreener: Dexscreener,
        registry: TerminalRegistry,
        solana: SolanaReader | None = None,
        helius: Helius | None = None,
        birdeye: Birdeye | None = None,
        probe: CostProbe | None = None,
    ) -> None:
        self.store = store
        self.strategy = strategy
        self.dex = dexscreener
        self.registry = registry
        self.solana = solana
        self.helius = helius
        self.birdeye = birdeye
        self.probe = probe

        self.shares = ShareTracker()
        self._holder_history: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._launch_slots: dict[str, int] = {}
        self._smart_cache: dict[Chain, set[str]] = {}

    # -- public ------------------------------------------------------------
    async def refresh(self, candidate: Candidate) -> PairSnapshot | None:
        """Update price/liquidity in place from the free provider."""
        snaps = await self.dex.token_pairs([candidate.address])
        snap = next((s for s in snaps if s.token_address == candidate.address), None)
        if snap is None:
            return None
        candidate.price_usd = snap.price_usd or candidate.price_usd
        candidate.liquidity_usd = snap.liquidity_usd
        candidate.mcap_usd = snap.mcap_usd or candidate.mcap_usd
        candidate.volume_5m_usd = snap.volume_m5
        candidate.pool_address = candidate.pool_address or snap.pair_address
        candidate.created_at_ms = candidate.created_at_ms or snap.created_at_ms
        candidate.symbol = candidate.symbol or snap.symbol
        return snap

    async def enrich(self, candidate: Candidate, probe_size_usd: float) -> Enrichment:
        result = Enrichment(candidate=candidate, features=Features())
        snap = await self.refresh(candidate)
        result.snap = snap
        if snap is None:
            result.unknown.update({"price_usd", "liquidity_usd"})
            result.notes.append("no market data from dexscreener")

        values: dict[str, float] = {}
        values.update(await self._chart_features(candidate, snap, result))
        values.update(await self._onchain_features(candidate, result))
        values.update(await self._cost_features(candidate, probe_size_usd, result))

        values["token_age_minutes"] = candidate.age_minutes
        values["liquidity_usd"] = candidate.liquidity_usd
        values["liq_to_mcap"] = (
            candidate.liquidity_usd / candidate.mcap_usd if candidate.mcap_usd > 0 else 0.0
        )

        known = set(Features.names())
        result.features = Features(**{k: v for k, v in values.items() if k in known})
        return result

    # -- chart -------------------------------------------------------------
    async def _chart_features(
        self, candidate: Candidate, snap: PairSnapshot | None, result: Enrichment
    ) -> dict[str, float]:
        cfg = self.strategy.section("chart")

        candles: list[Candle] = []
        if self.birdeye and self.birdeye.enabled:
            candles = await self.birdeye.candles(
                candidate.address,
                chain=candidate.chain,
                minutes=int(cfg.get("lookback_candles", 30)),
            )
        result.candles = candles
        if candles:
            return chart.extract(candles, cfg)

        # No candle provider. On Solana the trade stream gives us real candles
        # a few lines further down; everywhere else we fall back to the
        # aggregate price changes, which are coarse but not wrong.
        if snap is not None:
            result.unknown.update({"vwap_dev", "atr_pct", "breakout", "volume_z", "body_ratio"})
            result.notes.append("chart from aggregate price changes, no candles")
            return {
                "ret_5m": snap.price_change_m5,
                "ret_15m": snap.price_change_h1,
                "parabolic": min(
                    1.0,
                    max(0.0, snap.price_change_h1 / float(cfg.get("parabolic_return_threshold", 1.5))),
                ),
            }
        result.unknown.add("chart")
        return {}

    # -- chain -------------------------------------------------------------
    async def _onchain_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        if candidate.chain is Chain.SOLANA and self.solana is not None:
            return await self._solana_features(candidate, result)
        return self._aggregate_only_features(candidate, result)

    async def _solana_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        assert self.solana is not None
        values: dict[str, float] = {}
        cfg_t = self.strategy.section("terminals")
        window_min = float(cfg_t.get("attribution_window_minutes", 10))
        max_txs = int(cfg_t.get("max_txs_inspected", 250))

        mint = None
        try:
            mint = await self.solana.mint_info(candidate.address)
        except Exception as exc:  # noqa: BLE001 - degrade, do not crash the tick
            result.notes.append(f"mint_info failed: {exc}")
        result.mint = mint
        if mint is None:
            result.unknown.update({"mint_authority", "freeze_authority"})

        launch_slot = self._launch_slots.get(candidate.address)
        if launch_slot is None:
            try:
                launch_slot = await self.solana.launch_slot(candidate.address)
            except Exception:  # noqa: BLE001
                launch_slot = 0
            self._launch_slots[candidate.address] = launch_slot
        if not launch_slot:
            result.unknown.add("bundle_pct")

        pool_addresses = {candidate.pool_address} if candidate.pool_address else set()
        holders = []
        try:
            holders = await self.solana.largest_holders(
                candidate.address,
                pool_addresses=pool_addresses,
                deployer=candidate.deployer,
                resolve_ages=10,
            )
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"holders failed: {exc}")
            result.unknown.update({"top10_pct", "gini", "fresh_wallet_pct"})

        stats = analyze(holders, now_ms=now_ms(), launch_slot=launch_slot or 0)
        result.holders = stats

        exact_count = None
        if self.helius and self.helius.enabled:
            exact_count = await self.helius.holder_count(candidate.address)
        if exact_count is None:
            result.unknown.add("holder_count")
            result.notes.append("holder count unknown (no Helius key)")
        else:
            stats.holder_count = exact_count
            history = self._holder_history[candidate.key]
            history.append((now_ms(), exact_count))
            del history[:-40]
            values["holder_growth_5m"] = holder_growth(history, 300_000)

        trades: list[flow.Trade] = []
        buys: list[terminals.BuyerTx] = []
        source = candidate.pool_address or candidate.address
        try:
            trades, buys = await self.solana.recent_activity(
                source, candidate.address, candidate.price_usd, max_txs=max_txs
            )
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"activity failed: {exc}")
            result.unknown.update({"retail_share", "bot_share", "axiom_share"})
        result.trades = trades

        if not result.candles and trades:
            result.candles = chart.candles_from_trades(
                [(t.ts_ms, t.price, t.size_usd) for t in trades],
                int(self.strategy.get("chart.candle_seconds", 60)),
            )
            if result.candles:
                values.update(chart.extract(result.candles, self.strategy.section("chart")))
                result.unknown -= {"vwap_dev", "atr_pct", "breakout", "volume_z", "body_ratio"}

        attribution = terminals.attribute(buys, self.registry)
        result.attribution = attribution
        values.update(
            terminals.extract(
                attribution,
                self.shares,
                candidate.key,
                now_ms(),
                int(window_min * 60_000),
            )
        )
        # Zero detectable terminals means the attribution features are noise,
        # not evidence. Flagging it here is what stops the strategy from
        # silently degenerating into "buy anything with volume".
        if not self.registry.attributable_labels:
            result.unknown.update({"retail_share", "retail_share_delta_5m", "axiom_share"})
            result.notes.append(
                "no terminal fee accounts labeled; run `alphahound discover-terminals`"
            )

        smart = self._smart_cache.get(candidate.chain)
        if smart is None:
            smart = self.store.smart_wallets(candidate.chain)
            self._smart_cache[candidate.chain] = smart
        values.update(flow.extract(trades, now_ms(), smart=smart))

        values.update(
            {
                "holder_count": float(stats.holder_count),
                "top10_pct": stats.top10_pct,
                "gini": stats.gini,
                "fresh_wallet_pct": stats.fresh_wallet_pct,
                "bundle_pct": stats.bundle_pct,
                "dev_holding_pct": stats.dev_holding_pct,
                "lp_locked_pct": stats.burned_pct,
            }
        )
        values.setdefault("holder_growth_5m", 0.0)
        return values

    def _aggregate_only_features(
        self, candidate: Candidate, result: Enrichment
    ) -> dict[str, float]:
        """The aggregate view only: every EVM chain, and Solana without an RPC.

        Holder distribution, wallet ages and terminal attribution all require a
        per-chain indexer we do not have, so those features are marked unknown
        rather than estimated. The honeypot and tax risk that distribution
        would have caught is instead caught by the round-trip probe, which is
        chain-agnostic and, for that specific risk, strictly better evidence.
        """
        result.unknown.update(
            {
                "holder_count",
                "top10_pct",
                "gini",
                "fresh_wallet_pct",
                "bundle_pct",
                "dev_holding_pct",
                "retail_share",
                "retail_share_delta_5m",
                "bot_share",
                "axiom_share",
                "mint_authority",
                "freeze_authority",
            }
        )
        result.notes.append(f"{candidate.chain.value}: aggregate-only enrichment")

        snap = result.snap
        if snap is None:
            result.unknown.update({"buy_sell_ratio", "unique_buyers_5m", "net_inflow_usd_5m"})
            return {}

        # Transaction counts, not dollars. Weaker than the USD-weighted ratio
        # the Solana path computes, and marked as such so the model does not
        # treat the two as interchangeable.
        total = snap.buys_m5 + snap.sells_m5
        ratio = min(10.0, snap.buys_m5 / snap.sells_m5) if snap.sells_m5 else (3.0 if snap.buys_m5 else 1.0)
        avg = snap.volume_m5 / total if total else 0.0
        result.unknown.add("net_inflow_usd_5m")
        return {
            "buy_sell_ratio": ratio,
            "unique_buyers_5m": float(snap.buys_m5),
            "avg_buy_size_usd": avg,
            "net_inflow_usd_5m": snap.volume_m5 * (ratio - 1.0) / (ratio + 1.0) if ratio else 0.0,
        }

    # -- cost --------------------------------------------------------------
    async def _cost_features(
        self, candidate: Candidate, probe_size_usd: float, result: Enrichment
    ) -> dict[str, float]:
        if self.probe is None or probe_size_usd <= 0:
            result.unknown.update({"price_impact", "round_trip_cost", "sellable"})
            return {}
        try:
            rt = await self.probe(candidate, probe_size_usd)
        except Exception as exc:  # noqa: BLE001
            result.round_trip = RoundTrip(ok=False, note=str(exc))
            result.unknown.update({"price_impact", "round_trip_cost", "sellable"})
            return {}
        result.round_trip = rt
        if not rt.ok:
            result.unknown.add("sellable")
        return {"price_impact": rt.price_impact, "round_trip_cost": rt.total_cost_pct}

    def forget(self, key: str) -> None:
        self.shares.forget(key)
        self._holder_history.pop(key, None)
