"""Per-chain playbooks: how we trade a *new* launch on that chain.

frankdegods on Fomo: size a narrative, and if the dump pattern repeats, cut
immediately — do not average, do not hope. We do not copy his bags (especially
not whales stacking $SOL). Copy is allowed only when labeled smart/Fomo money
is buying a still-young launchpad mint.

Nova (@badattrading_): bundle %, cluster/bubblemap, fresh wallets that look
organic but are all new. Buy early, sell the initial spike, leave a runner.

Robinhood Chain is the hot venue right now: shorter age window, faster first
take, volume + whether CT is even talking about the ticker.
"""

from __future__ import annotations

from .models import Chain
from .settings import Config


def section(strategy: Config, chain: Chain) -> dict:
    return strategy.section(f"playbooks.{chain.value}")


def max_age_minutes(strategy: Config, chain: Chain) -> float:
    pb = section(strategy, chain)
    if pb.get("max_age_minutes") is not None:
        return float(pb["max_age_minutes"])
    return float(strategy.get("loop.max_candidate_age_minutes", 180))


def ladder(strategy: Config, chain: Chain) -> list[tuple[float, float]]:
    pb = section(strategy, chain)
    raw = pb.get("take_profit_ladder") or strategy.get(
        "exits.take_profit_ladder", [[1.25, 0.30], [1.80, 0.20], [3.00, 0.20]]
    )
    return [(float(m), float(f)) for m, f in raw]


def gate(strategy: Config, chain: Chain, name: str, default: float, store=None) -> float:
    pb = section(strategy, chain)
    if pb.get(name) is not None:
        return float(pb[name])
    if store is not None:
        return store.param(f"gates.{name}", float(strategy.get(f"gates.{name}", default)))
    return float(strategy.get(f"gates.{name}", default))


def thesis_cut(strategy: Config, chain: Chain) -> float:
    pb = section(strategy, chain)
    if pb.get("thesis_cut_from_peak") is not None:
        return float(pb["thesis_cut_from_peak"])
    return float(strategy.get("exits.thesis_cut_from_peak", 0.28))


def copy_signal(
    *,
    age_minutes: float,
    mcap_usd: float,
    smart_buys: float,
    fomo_inside: float,
    fomo_net_flow: float,
    whale_net_flow: float,
    strategy: Config,
    chain: Chain,
) -> float:
    """1.0 only if someone worth chasing is buying a *new, small* launch.

    A whale printing into a mid-cap $SOL pet is not this. That is their game.
    """
    pb = section(strategy, chain)
    max_age = float(pb.get("copy_max_age_minutes", 25))
    max_mcap = float(pb.get("copy_max_mcap_usd", 2_000_000))
    min_mcap = float(pb.get("copy_min_mcap_usd", 100_000))
    if age_minutes > max_age:
        return 0.0
    if mcap_usd > max_mcap > 0:
        return 0.0
    if min_mcap > 0 and mcap_usd < min_mcap:
        return 0.0
    buying = smart_buys >= 1 or (fomo_inside >= 1 and fomo_net_flow > 0) or whale_net_flow > 0.25
    return 1.0 if buying else 0.0
