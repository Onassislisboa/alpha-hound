"""Paper venue.

The default mode, and the one the whole project is designed around: the
learning loop needs hundreds of closed trades before its weights mean anything,
and paying for that education in real money is optional.

The fill model is deliberately pessimistic rather than convenient. Constant
product impact plus a latency haircut plus fees, on both legs. A paper mode
that fills at the mid price teaches the model that costs do not exist, and it
will size accordingly the day you switch to live.
"""

from __future__ import annotations

from ..log import get
from ..models import Candidate, Chain, Fill, Quote, Side, VenueId, now_ms
from ..providers import Dexscreener
from ..settings import Config
from ..fees import FeePlan

log = get("paper")


class PaperVenue:
    id = VenueId.PAPER

    def __init__(self, dexscreener: Dexscreener, strategy: Config) -> None:
        self.dex = dexscreener
        self.slippage = float(strategy.get("execution.paper_slippage_bps", 180)) / 10_000.0
        self.fee = float(strategy.get("execution.paper_fee_bps", 75)) / 10_000.0

    def supports(self, chain: Chain) -> bool:
        return True

    @staticmethod
    def _impact(size_usd: float, liquidity_usd: float) -> float:
        """Constant-product price impact for a trade of `size_usd` against a
        pool holding `liquidity_usd` total.

        For x_in against reserve R, the executed price is worse than mid by
        roughly x/(R+x). Using half the reported liquidity as the quote-side
        reserve, since Dexscreener reports both sides combined.
        """
        reserve = max(1.0, liquidity_usd / 2.0)
        return min(0.95, size_usd / (reserve + size_usd))

    async def quote(
        self, candidate: Candidate, side: Side, amount: float, fees: FeePlan | None = None
    ) -> Quote:
        price = candidate.price_usd
        if price <= 0:
            snaps = await self.dex.token_pairs([candidate.address])
            price = next((s.price_usd for s in snaps), 0.0)
        if price <= 0:
            return Quote(venue=self.id, in_amount=amount, out_amount=0.0, price=0.0, price_impact=1.0)

        slip = (fees.slippage_bps / 10_000.0) if fees is not None else self.slippage

        if side is Side.BUY:
            notional = amount
            impact = self._impact(notional, candidate.liquidity_usd)
            effective = price * (1.0 + impact + slip)
            tokens = (notional * (1.0 - self.fee)) / effective
            return Quote(
                venue=self.id,
                in_amount=notional,
                out_amount=tokens,
                price=effective,
                price_impact=impact,
                fee_usd=notional * self.fee,
            )

        notional = amount * price
        impact = self._impact(notional, candidate.liquidity_usd)
        effective = price * (1.0 - impact - slip)
        usd_out = amount * effective * (1.0 - self.fee)
        return Quote(
            venue=self.id,
            in_amount=amount,
            out_amount=usd_out,
            price=effective,
            price_impact=impact,
            fee_usd=notional * self.fee,
        )

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill:
        if quote.out_amount <= 0:
            raise RuntimeError(f"paper: no price for {candidate.symbol or candidate.address}")
        return Fill(
            venue=self.id,
            side=side,
            amount_in=quote.in_amount,
            amount_out=quote.out_amount,
            price=quote.price,
            fee_usd=quote.fee_usd,
            tx_id=f"paper-{now_ms()}",
            slippage_vs_quote=0.0,
        )
