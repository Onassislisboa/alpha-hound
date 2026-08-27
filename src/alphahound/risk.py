"""Risk and position sizing.

The unglamorous truth about trading bots is that the signal decides which
trades you take and the risk layer decides whether you are still solvent to
take them. Everything here exists to bound a single failure: the bot being
confidently wrong many times in a row, faster than a human could intervene.

Four independent brakes, deliberately redundant:

* size caps      - no single trade can matter that much
* liquidity cap  - no position larger than the pool can give back
* daily loss     - a bad day ends by itself
* cooldown       - a losing streak gets slower, not louder
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .log import get
from .models import Candidate, Position, Score, now_ms
from .scoring import Payoff
from .settings import Config
from .store import Store

log = get("risk")

KILL_SWITCH_KEY = "kill_switch"
COOLDOWN_UNTIL_KEY = "cooldown_until_ms"
COOLDOWN_LEVEL_KEY = "cooldown_level"


def utc_day_start_ms(now: float | None = None) -> int:
    dt = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


@dataclass(slots=True)
class Sizing:
    allowed: bool
    size_usd: float = 0.0
    reason: str = ""
    kelly_raw: float = 0.0
    binding_constraint: str = ""


class RiskEngine:
    def __init__(self, strategy: Config, store: Store) -> None:
        self.strategy = strategy
        self.store = store

    # -- tunables ---------------------------------------------------------
    def _p(self, name: str, default: float) -> float:
        return self.store.param(f"risk.{name}", float(self.strategy.get(f"risk.{name}", default)))

    @property
    def base_equity(self) -> float:
        return self._p("equity_usd", 1000.0)

    def equity(self) -> float:
        """Starting capital plus everything realized since. Sizing off a static
        number means the bot risks the same dollars after halving the account
        as before, which is how a drawdown becomes terminal."""
        return max(0.0, self.base_equity + self.store.realized_pnl())

    # -- global state -----------------------------------------------------
    def halted(self) -> tuple[bool, str]:
        if self.store.get_kv(KILL_SWITCH_KEY) == "1":
            return True, "kill switch engaged (resume with `alphahound resume`)"

        limit = self._p("daily_loss_limit_pct", 0.12)
        day_pnl = self.store.realized_pnl(since_ms=utc_day_start_ms())
        if limit > 0 and day_pnl <= -limit * self.base_equity:
            self.engage_kill_switch(
                f"daily loss {day_pnl:+.2f} USD breached {limit:.0%} of equity"
            )
            return True, f"daily loss limit hit ({day_pnl:+.2f} USD)"

        until = int(self.store.get_kv(COOLDOWN_UNTIL_KEY, "0") or 0)
        if until > now_ms():
            remaining = (until - now_ms()) / 60_000.0
            return True, f"cooldown active for {remaining:.1f} more minutes"
        return False, ""

    def engage_kill_switch(self, reason: str) -> None:
        if self.store.get_kv(KILL_SWITCH_KEY) != "1":
            log.error("kill switch engaged", extra={"reason": reason})
        self.store.set_kv(KILL_SWITCH_KEY, "1")
        self.store.set_kv("kill_switch_reason", reason)

    def release_kill_switch(self) -> None:
        self.store.set_kv(KILL_SWITCH_KEY, "0")
        self.store.set_kv("kill_switch_reason", "")

    def note_trade_closed(self, won: bool) -> None:
        """Escalating cooldown on a losing streak.

        The point is not superstition about streaks. It is that N losses in a
        row is the cheapest available evidence that the model's current view of
        the market is wrong, and the correct response to being wrong at
        unknown scale is to trade less until you know more.
        """
        if won:
            self.store.set_kv(COOLDOWN_LEVEL_KEY, "0")
            return
        limit = int(self._p("consecutive_loss_limit", 4))
        streak = self.store.consecutive_losses()
        if streak < limit:
            return
        level = int(self.store.get_kv(COOLDOWN_LEVEL_KEY, "0") or 0)
        base = self._p("cooldown_minutes_base", 15.0)
        mult = self._p("cooldown_backoff_multiplier", 2.0)
        minutes = base * (mult**level)
        self.store.set_kv(COOLDOWN_UNTIL_KEY, str(now_ms() + int(minutes * 60_000)))
        self.store.set_kv(COOLDOWN_LEVEL_KEY, str(level + 1))
        log.warning(
            "cooldown engaged", extra={"streak": streak, "minutes": minutes, "level": level + 1}
        )

    # -- sizing -----------------------------------------------------------
    def size(
        self,
        candidate: Candidate,
        score: Score,
        payoff: Payoff,
        open_positions: list[Position],
    ) -> Sizing:
        halted, reason = self.halted()
        if halted:
            return Sizing(allowed=False, reason=reason)

        max_open = int(self._p("max_concurrent_positions", 4))
        if len(open_positions) >= max_open:
            return Sizing(allowed=False, reason=f"already at {max_open} open positions")

        per_chain = int(self._p("max_positions_per_chain", 3))
        same_chain = [p for p in open_positions if p.candidate.chain is candidate.chain]
        if len(same_chain) >= per_chain:
            return Sizing(
                allowed=False, reason=f"{candidate.chain.value} at its {per_chain}-position cap"
            )

        if any(p.candidate.address == candidate.address for p in open_positions):
            return Sizing(allowed=False, reason="already holding this token")

        equity = self.equity()
        if equity <= 0:
            return Sizing(allowed=False, reason="no equity left")

        kelly = kelly_fraction(score.probability, payoff)
        fraction = kelly * self._p("kelly_fraction", 0.25)
        constraint = "kelly"

        cap_pct = self._p("max_position_pct", 0.05)
        if fraction > cap_pct:
            fraction, constraint = cap_pct, "max_position_pct"
        if fraction <= 0:
            return Sizing(
                allowed=False,
                reason=f"Kelly says no bet (p={score.probability:.3f}, "
                f"win={payoff.avg_win:.2f}, loss={payoff.avg_loss:.2f})",
                kelly_raw=kelly,
            )

        size_usd = equity * fraction

        liq_cap_pct = self._p("max_position_pct_of_liquidity", 0.01)
        if candidate.liquidity_usd > 0:
            liq_cap = candidate.liquidity_usd * liq_cap_pct
            if liq_cap < size_usd:
                size_usd, constraint = liq_cap, "pool_liquidity"

        deployer_cap = self._p("max_exposure_per_deployer_pct", 0.06) * equity
        if candidate.deployer:
            existing = sum(
                p.size_usd for p in open_positions if p.candidate.deployer == candidate.deployer
            )
            room = deployer_cap - existing
            if room <= 0:
                return Sizing(
                    allowed=False,
                    reason=f"deployer {candidate.deployer[:8]} already at exposure cap",
                    kelly_raw=kelly,
                )
            if room < size_usd:
                size_usd, constraint = room, "deployer_exposure"

        min_size = self._p("min_position_usd", 15.0)
        if size_usd < min_size:
            return Sizing(
                allowed=False,
                reason=f"sized down to {size_usd:.2f} USD, below the {min_size:.0f} minimum "
                f"({constraint} binding)",
                kelly_raw=kelly,
                binding_constraint=constraint,
            )

        return Sizing(
            allowed=True,
            size_usd=round(size_usd, 2),
            reason=f"{constraint} binding",
            kelly_raw=kelly,
            binding_constraint=constraint,
        )


def kelly_fraction(probability: float, payoff: Payoff) -> float:
    """Kelly for an asymmetric bet: f* = (p*W - q*L) / (W*L).

    Returned raw and unclamped so the caller can see when the model wanted more
    than the caps allow - that gap is useful diagnostic information. Full Kelly
    is only optimal if your probabilities are calibrated, and yours are
    estimated from a few hundred memecoin trades, so the caller multiplies this
    by a fraction well under one.
    """
    win, loss = payoff.avg_win, payoff.avg_loss
    if win <= 0 or loss <= 0:
        return 0.0
    q = 1.0 - probability
    return max(0.0, (probability * win - q * loss) / (win * loss))
