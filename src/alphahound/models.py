"""Core data types.

Deliberately stdlib-only: every module that imports this one must be runnable
and testable without installing a single dependency. Network and chain clients
live behind the adapters in `execution/` and import their deps lazily.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum


def now_ms() -> int:
    return int(time.time() * 1000)


class Chain(str, Enum):
    SOLANA = "solana"
    BNB = "bnb"
    BASE = "base"
    ROBINHOOD_CHAIN = "robinhood_chain"
    # Not a chain: the Robinhood brokerage. Majors only, no memecoins. Kept in
    # the same enum because it is a routing destination like any other.
    ROBINHOOD_BROKER = "robinhood_broker"


EVM_CHAIN_IDS = {
    Chain.BNB: 56,
    Chain.BASE: 8453,
    Chain.ROBINHOOD_CHAIN: 4663,
}


class VenueId(str, Enum):
    PAPER = "paper"
    JUPITER = "jupiter"
    ZEROEX = "zeroex"
    UNISWAP_V3 = "uniswap_v3"
    ROBINHOOD = "robinhood"
    RELAY = "relay"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Action(str, Enum):
    ENTER = "enter"
    # Rejected, but shadow-tracked so we can measure what the filter cost us.
    REJECT_GATE = "reject_gate"
    REJECT_SCORE = "reject_score"
    REJECT_RISK = "reject_risk"


class ExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    LIQUIDITY_DRAIN = "liquidity_drain"
    KILL_SWITCH = "kill_switch"
    MANUAL = "manual"


class ErrorClass(str, Enum):
    """Loss taxonomy. Each class maps to one specific parameter nudge in
    `learning.postmortem`; a bucket that does not imply an action is a bucket
    that teaches nothing."""

    WIN = "win"
    LATE_ENTRY = "late_entry"
    RUG = "rug"
    SLIPPAGE_BLOWOUT = "slippage_blowout"
    EXIT_TOO_SLOW = "exit_too_slow"
    EXIT_TOO_FAST = "exit_too_fast"
    ADVERSE_SELECTION = "adverse_selection"
    EXECUTION_FAIL = "execution_fail"
    # The honest bucket. The signal was fine and the token simply faded. If
    # this is not your largest loss class, your taxonomy is lying to you.
    NO_EDGE = "no_edge"


@dataclass(slots=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class Candidate:
    """A token that might be worth trading. Cheap to create, expensive to
    enrich, so discovery produces these and the engine enriches on demand."""

    chain: Chain
    address: str
    symbol: str = ""
    name: str = ""
    created_at_ms: int = 0
    price_usd: float = 0.0
    liquidity_usd: float = 0.0
    mcap_usd: float = 0.0
    volume_5m_usd: float = 0.0
    pool_address: str = ""
    quote_asset: str = ""
    deployer: str = ""
    source: str = ""
    discovered_at_ms: int = field(default_factory=now_ms)

    @property
    def key(self) -> str:
        return f"{self.chain.value}:{self.address}"

    @property
    def age_minutes(self) -> float:
        if not self.created_at_ms:
            return 0.0
        return (now_ms() - self.created_at_ms) / 60_000.0


@dataclass(slots=True)
class Features:
    """The model's entire view of the world.

    Field order IS the vector order. Appending a field is safe (it gets the
    prior weight); reordering or removing one invalidates saved weights, which
    is why `scoring.Model` stores weights by name, not by index.
    """

    # --- chart -------------------------------------------------------------
    ret_5m: float = 0.0
    ret_15m: float = 0.0
    vwap_dev: float = 0.0
    atr_pct: float = 0.0
    breakout: float = 0.0
    volume_z: float = 0.0
    body_ratio: float = 0.0
    parabolic: float = 0.0

    # --- holders / distribution -------------------------------------------
    holder_count: float = 0.0
    holder_growth_5m: float = 0.0
    top10_pct: float = 0.0
    gini: float = 0.0
    fresh_wallet_pct: float = 0.0
    bundle_pct: float = 0.0
    dev_holding_pct: float = 0.0
    lp_locked_pct: float = 0.0

    # --- terminal attribution ---------------------------------------------
    # Level and derivative carry opposite signs on purpose. See the
    # [terminals] block in config/strategy.toml.
    retail_share: float = 0.0
    retail_share_delta_5m: float = 0.0
    bot_share: float = 0.0
    axiom_share: float = 0.0
    axiom_share_delta_5m: float = 0.0
    unknown_share: float = 0.0

    # --- order flow --------------------------------------------------------
    unique_buyers_5m: float = 0.0
    buy_sell_ratio: float = 0.0
    net_inflow_usd_5m: float = 0.0
    avg_buy_size_usd: float = 0.0
    smart_money_buys: float = 0.0

    # --- liquidity / microstructure ---------------------------------------
    liquidity_usd: float = 0.0
    liq_to_mcap: float = 0.0
    price_impact: float = 0.0
    round_trip_cost: float = 0.0

    # Deployer rug rate belongs here and is deliberately absent: scoring it
    # requires an index of every token an address has ever launched, which no
    # free provider exposes, and a defaulted 0.0 reads as a spotless record.
    # Per-deployer *exposure* is capped in risk.py instead - that only needs the
    # deployer's identity, which we do have.
    token_age_minutes: float = 0.0

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def vector(self) -> list[float]:
        return [getattr(self, n) for n in self.names()]


@dataclass(slots=True)
class Score:
    probability: float
    expected_value: float
    # Per-feature contribution to the logit, for explainability. Without this
    # you cannot tell a good model from a lucky one.
    contributions: dict[str, float] = field(default_factory=dict)
    veto_reasons: list[str] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return bool(self.veto_reasons)

    def top_drivers(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))[:n]


@dataclass(slots=True)
class Decision:
    candidate: Candidate
    features: Features
    score: Score
    action: Action
    size_usd: float = 0.0
    reason: str = ""
    ts_ms: int = field(default_factory=now_ms)
    weights_version: int = 0


@dataclass(slots=True)
class Quote:
    venue: VenueId
    in_amount: float
    out_amount: float
    price: float
    price_impact: float
    fee_usd: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class RoundTrip:
    """Result of quoting a buy and the matching sell back to back.

    This single probe answers three questions at once - what the entry costs,
    whether the token can be sold at all, and what the total round trip eats -
    and it does so venue-agnostically, using the same router that will execute.
    A honeypot fails here, and so does a token whose exit tax makes the trade
    unprofitable no matter how good the signal was.
    """

    ok: bool
    price_impact: float = 0.0
    sell_slippage: float = 0.0
    total_cost_pct: float = 0.0
    note: str = ""


@dataclass(slots=True)
class Fill:
    venue: VenueId
    side: Side
    amount_in: float
    amount_out: float
    price: float
    fee_usd: float
    tx_id: str = ""
    ts_ms: int = field(default_factory=now_ms)
    # Realized slippage against the quote we based the decision on. This is the
    # single most useful number for diagnosing execution, and it is the one
    # most bots never record.
    slippage_vs_quote: float = 0.0


@dataclass(slots=True)
class Position:
    candidate: Candidate
    venue: VenueId
    entry_price: float
    size_usd: float
    tokens: float
    opened_at_ms: int = field(default_factory=now_ms)
    tokens_remaining: float = 0.0
    realized_usd: float = 0.0
    fees_usd: float = 0.0
    last_exit_price: float = 0.0
    last_exit_reason: str = ""
    peak_price: float = 0.0
    trough_price: float = 0.0
    peak_liquidity_usd: float = 0.0
    ladder_filled: int = 0
    trailing_active: bool = False
    decision_id: int = 0
    entry_features: Features = field(default_factory=Features)
    # Price at the moment the decision was made, before execution latency. The
    # gap between this and entry_price is the late_entry error class.
    signal_price: float = 0.0

    def __post_init__(self) -> None:
        if self.tokens_remaining == 0.0:
            self.tokens_remaining = self.tokens
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.trough_price == 0.0:
            self.trough_price = self.entry_price
        if self.peak_liquidity_usd == 0.0:
            self.peak_liquidity_usd = self.candidate.liquidity_usd

    def unrealized_usd(self, price: float) -> float:
        return self.tokens_remaining * price - self.size_usd * (
            self.tokens_remaining / self.tokens if self.tokens else 0.0
        )

    def gain(self, price: float) -> float:
        if not self.entry_price:
            return 0.0
        return price / self.entry_price - 1.0


@dataclass(slots=True)
class TradeRecord:
    """A closed round trip. This is the only thing the learner trusts."""

    key: str
    chain: Chain
    venue: VenueId
    opened_at_ms: int
    closed_at_ms: int
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    fees_usd: float
    exit_reason: ExitReason
    error_class: ErrorClass
    features: Features
    weights_version: int
    signal_price: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    entry_slippage: float = 0.0
    notes: str = ""

    @property
    def pnl_pct(self) -> float:
        return self.pnl_usd / self.size_usd if self.size_usd else 0.0

    @property
    def won(self) -> bool:
        return self.pnl_usd > 0.0
