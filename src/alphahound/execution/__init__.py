"""Execution routing.

One `Venue` interface, one `Router` that picks a venue per chain, and adapters
that each know exactly one API. The router also owns the round-trip probe that
the signal layer uses for cost and sellability, which is deliberate: the cost
estimate that gates a trade should come from the same code path that will
execute it, or you are gating on a number nobody will honour.

Adapter status, honestly stated:

    paper       simulated fills. The default, and the only mode that cannot
                lose money.
    jupiter     REAL. api.jup.ag/swap/v2 /order + /execute.
    zeroex      REAL. 0x Swap v2 allowance-holder, BNB (56) and Base (8453).
    uniswap_v3  REAL. On-chain SwapRouter02 for Robinhood Chain (4663), which
                0x does not cover.
    robinhood   REAL. Robinhood Crypto brokerage, Ed25519-signed. Majors only.

Fomo and Moby are absent from that list on purpose: neither publishes a
trading API. They are handled by `relay.IntentRelay`, which is a notification
sink rather than a venue, because a venue that cannot fill has no business
pretending to be one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..log import get
from ..models import (
    Candidate,
    Chain,
    Fill,
    Position,
    Quote,
    RoundTrip,
    Side,
    VenueId,
)
from ..providers import Dexscreener
from ..settings import Config

log = get("execution")

WRAPPED_NATIVE: dict[Chain, str] = {
    Chain.SOLANA: "So11111111111111111111111111111111111111112",
    Chain.BNB: "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    Chain.BASE: "0x4200000000000000000000000000000000000006",
    # Robinhood Chain uses ETH for gas; its WETH comes from config because the
    # chain is new enough that hardcoding it would age badly.
    Chain.ROBINHOOD_CHAIN: "",
}


class ExecutionError(RuntimeError):
    pass


class NotSupported(ExecutionError):
    pass


@runtime_checkable
class Venue(Protocol):
    id: VenueId

    def supports(self, chain: Chain) -> bool: ...

    async def quote(self, candidate: Candidate, side: Side, amount: float) -> Quote: ...

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill: ...


class Router:
    """Chain -> venue, plus the shared round-trip probe and native price cache."""

    def __init__(
        self,
        venues: list[Venue],
        dexscreener: Dexscreener,
        strategy: Config,
        *,
        native_addresses: dict[Chain, str] | None = None,
    ) -> None:
        self.venues = venues
        self.dex = dexscreener
        self.strategy = strategy
        self.native = dict(WRAPPED_NATIVE)
        self.native.update(native_addresses or {})
        self._native_price: dict[Chain, tuple[int, float]] = {}

    def venue_for(self, chain: Chain) -> Venue:
        for venue in self.venues:
            if venue.supports(chain):
                return venue
        raise NotSupported(f"no venue configured for {chain.value}")

    def has_venue(self, chain: Chain) -> bool:
        return any(v.supports(chain) for v in self.venues)

    # -- pricing -----------------------------------------------------------
    async def native_price_usd(self, chain: Chain) -> float:
        """USD price of the chain's gas token, cached for 60s.

        Needed because every venue quotes in the native asset while every risk
        limit is denominated in dollars, and getting this conversion wrong
        scales every position by the same silent factor.
        """
        from ..models import now_ms

        cached = self._native_price.get(chain)
        if cached and now_ms() - cached[0] < 60_000:
            return cached[1]
        address = self.native.get(chain, "")
        if not address:
            return 0.0
        snaps = await self.dex.token_pairs([address])
        price = max((s.price_usd for s in snaps), default=0.0)
        if price > 0:
            self._native_price[chain] = (now_ms(), price)
            return price
        return cached[1] if cached else 0.0

    # -- probe -------------------------------------------------------------
    async def round_trip(self, candidate: Candidate, size_usd: float) -> RoundTrip:
        """Quote a buy, then quote selling exactly what that buy would give.

        This is the honeypot check, the tax check and the cost check in one
        request pair. A token that cannot be sold fails here rather than after
        you own it, which is the only useful time to find out.
        """
        try:
            venue = self.venue_for(candidate.chain)
        except NotSupported as exc:
            return RoundTrip(ok=False, note=str(exc))

        try:
            buy = await venue.quote(candidate, Side.BUY, size_usd)
        except Exception as exc:  # noqa: BLE001
            return RoundTrip(ok=False, note=f"buy quote failed: {exc}")
        if buy.out_amount <= 0:
            return RoundTrip(ok=False, note="buy quote returned zero output")

        try:
            sell = await venue.quote(candidate, Side.SELL, buy.out_amount)
        except Exception as exc:  # noqa: BLE001
            # Cannot quote an exit. Treat as unsellable; this is the single most
            # valuable false negative in the system.
            return RoundTrip(
                ok=False,
                price_impact=buy.price_impact,
                note=f"sell quote failed (treat as honeypot): {exc}",
            )
        if sell.out_amount <= 0:
            return RoundTrip(ok=False, price_impact=buy.price_impact, note="sell quote is zero")

        total_cost = 1.0 - (sell.out_amount / size_usd) if size_usd > 0 else 1.0
        return RoundTrip(
            ok=True,
            price_impact=buy.price_impact,
            sell_slippage=max(sell.price_impact, 0.0),
            total_cost_pct=max(0.0, total_cost),
            note=f"{buy.venue.value}: in {size_usd:.2f} -> out {sell.out_amount:.2f} USD",
        )

    # -- trading -----------------------------------------------------------
    async def buy(self, candidate: Candidate, size_usd: float) -> Fill:
        venue = self.venue_for(candidate.chain)
        quote = await venue.quote(candidate, Side.BUY, size_usd)
        max_drift = float(self.strategy.get("execution.max_price_drift_from_signal", 0.05))
        if candidate.price_usd > 0 and quote.price > 0:
            drift = abs(quote.price / candidate.price_usd - 1.0)
            if drift > max_drift:
                # The signal was computed against a price that no longer
                # exists. Chasing it is how a good decision becomes a bad fill.
                raise ExecutionError(
                    f"price drifted {drift:.1%} since the signal (limit {max_drift:.1%})"
                )
        return await venue.execute(candidate, Side.BUY, size_usd, quote)

    async def sell(self, position: Position, tokens: float) -> Fill:
        venue = self.venue_for(position.candidate.chain)
        quote = await venue.quote(position.candidate, Side.SELL, tokens)
        return await venue.execute(position.candidate, Side.SELL, tokens, quote)


def build_router(
    settings, strategy: Config, dexscreener: Dexscreener, http
) -> Router:
    """Assemble the venue list from configuration.

    Paper is always appended last so that an unconfigured chain simulates
    rather than raising. In live mode a chain without a real venue is a
    configuration error, and `Settings.validate` catches it at boot instead of
    at the first trade.
    """
    from .evm import UniswapV3Venue, ZeroExVenue
    from .jupiter import JupiterVenue
    from .paper import PaperVenue
    from .robinhood import RobinhoodVenue

    venues: list[Venue] = []
    if settings.live:
        if Chain.SOLANA in settings.enabled_chains and settings.jupiter_api_key:
            venues.append(JupiterVenue(http, settings, strategy))
        if settings.zeroex_api_key:
            venues.append(ZeroExVenue(http, settings, strategy, dexscreener))
        if Chain.ROBINHOOD_CHAIN in settings.enabled_chains and settings.rh_chain_swap_router:
            venues.append(UniswapV3Venue(http, settings, strategy, dexscreener))
        if Chain.ROBINHOOD_BROKER in settings.enabled_chains and settings.rh_api_key:
            venues.append(RobinhoodVenue(http, settings))

    venues.append(PaperVenue(dexscreener, strategy))
    native = {Chain.ROBINHOOD_CHAIN: settings.rh_chain_weth} if settings.rh_chain_weth else {}
    return Router(venues, dexscreener, strategy, native_addresses=native)
