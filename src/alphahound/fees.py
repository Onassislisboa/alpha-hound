"""Volume-scaled slippage and priority fee, with a capped retry bump.

Small bankroll cannot eat a 5% slip on a $12 fill, and it also cannot win a
congested pump.fun slot at the default 50k lamports. Both numbers move with
5-minute volume, then a failed submit is allowed two small bumps — never a
third, never past the hard cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Candidate
from .settings import Config


@dataclass(frozen=True, slots=True)
class FeePlan:
    slippage_bps: int
    priority_lamports: int
    attempt: int


def _num(strategy: Config, store, name: str, default: float) -> float:
    base = float(strategy.get(f"execution.{name}", default))
    return store.param(f"execution.{name}", base) if store is not None else base


def plan_for(
    candidate: Candidate,
    strategy: Config,
    attempt: int = 0,
    store=None,
) -> FeePlan:
    """attempt 0 = first try. Bumps apply on 1 and 2 only, even if retries > 3."""
    attempt = max(0, int(attempt))
    bumps = min(attempt, 2)

    vol = max(float(candidate.volume_5m_usd), 1.0)
    base_slip = int(_num(strategy, store, "slippage_bps", 150))
    floor = int(_num(strategy, store, "min_slippage_bps", 80))
    ceil = int(_num(strategy, store, "max_slippage_bps", 350))
    bump_slip = int(_num(strategy, store, "retry_slippage_bump_bps", 35))
    # Thin 5m volume needs more slip; a busy book does not. sqrt so a dead
    # pool cannot demand 10% from a $10 account.
    ref = max(_num(strategy, store, "slippage_volume_ref_usd", 20_000.0), 1.0)
    scale = min(2.0, max(0.7, (ref / vol) ** 0.5))
    slip = int(base_slip * scale) + bumps * bump_slip
    slip = max(floor, min(ceil, slip))

    base_prio = int(_num(strategy, store, "priority_lamports", 80_000))
    cap_prio = int(_num(strategy, store, "max_priority_lamports", 350_000))
    prio_mult = _num(strategy, store, "retry_priority_mult", 1.30)
    # Busy tokens compete for the same block. Quiet ones do not need a tip.
    prio_scale = min(2.2, max(0.8, (vol / 12_000.0) ** 0.35))
    prio = int(base_prio * prio_scale * (prio_mult**bumps))
    prio = max(0, min(cap_prio, prio))
    return FeePlan(slippage_bps=slip, priority_lamports=prio, attempt=attempt)


def slip_bps(fees: FeePlan | None, fallback: int) -> int:
    return fees.slippage_bps if fees is not None else fallback
