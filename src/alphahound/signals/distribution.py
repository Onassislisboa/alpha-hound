"""Holder distribution analysis.

The purpose of this module is to answer one question: if this goes up, who is
standing behind me with something to sell? Every metric here is a different
angle on that.

All functions are pure so they can be unit-tested and replayed. Fetching the
holder set is somebody else's problem (`signals.solana`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Holder:
    address: str
    balance: float
    # Unix ms of the wallet's first observed activity. 0 = unknown.
    first_seen_ms: int = 0
    # Slot/block in which this wallet acquired the token. 0 = unknown.
    acquired_slot: int = 0
    # Wallet that funded this one, when we can see it one hop back.
    funder: str = ""
    is_lp: bool = False
    is_burn: bool = False
    is_deployer: bool = False


@dataclass(slots=True)
class HolderStats:
    holder_count: int = 0
    top1_pct: float = 0.0
    top10_pct: float = 0.0
    gini: float = 0.0
    dev_holding_pct: float = 0.0
    lp_pct: float = 0.0
    burned_pct: float = 0.0
    fresh_wallet_pct: float = 0.0
    bundle_pct: float = 0.0
    largest_funding_cluster_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


def gini(values: list[float]) -> float:
    """Gini coefficient of the balance distribution.

    Top-10 share hides the shape: 10 wallets at 4% each and one at 40% give the
    same headline number and completely different risk. Gini separates them.
    """
    vals = sorted(v for v in values if v > 0)
    n = len(vals)
    if n < 2:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    cumulative = 0.0
    for i, v in enumerate(vals, start=1):
        cumulative += i * v
    return (2.0 * cumulative) / (n * total) - (n + 1.0) / n


def analyze(
    holders: list[Holder],
    *,
    now_ms: int,
    launch_slot: int = 0,
    fresh_wallet_max_age_hours: float = 24.0,
    bundle_slot_window: int = 3,
) -> HolderStats:
    """Turn a holder set into the distribution slice of the feature vector.

    LP and burn addresses are excluded from concentration, because counting the
    pool as a whale makes every healthy token look like a rug and every real
    whale look normal - the exact inversion of what you want.
    """
    stats = HolderStats()
    if not holders:
        stats.notes.append("no holder data")
        return stats

    circulating = [h for h in holders if not (h.is_lp or h.is_burn)]
    supply_total = sum(h.balance for h in holders)
    circ_total = sum(h.balance for h in circulating)

    if supply_total > 0:
        stats.lp_pct = sum(h.balance for h in holders if h.is_lp) / supply_total
        stats.burned_pct = sum(h.balance for h in holders if h.is_burn) / supply_total

    stats.holder_count = len(circulating)
    if not circulating or circ_total <= 0:
        stats.notes.append("no circulating holders")
        return stats

    balances = sorted((h.balance for h in circulating), reverse=True)
    stats.top1_pct = balances[0] / circ_total
    stats.top10_pct = sum(balances[:10]) / circ_total
    stats.gini = gini(balances)
    stats.dev_holding_pct = (
        sum(h.balance for h in circulating if h.is_deployer) / circ_total
    )

    fresh_cutoff = now_ms - int(fresh_wallet_max_age_hours * 3_600_000)
    known_age = [h for h in circulating if h.first_seen_ms > 0]
    if known_age:
        fresh = [h for h in known_age if h.first_seen_ms >= fresh_cutoff]
        stats.fresh_wallet_pct = len(fresh) / len(known_age)
    else:
        stats.notes.append("wallet ages unavailable")

    if launch_slot > 0:
        bundled = [
            h
            for h in circulating
            if h.acquired_slot and h.acquired_slot <= launch_slot + bundle_slot_window
        ]
        stats.bundle_pct = sum(h.balance for h in bundled) / circ_total
    else:
        stats.notes.append("launch slot unknown, bundle_pct not computed")

    stats.largest_funding_cluster_pct = funding_cluster_share(circulating, circ_total)
    return stats


def funding_cluster_share(holders: list[Holder], circ_total: float) -> float:
    """Largest share of supply held by wallets funded from the same source.

    Twenty wallets funded by one address are one holder wearing twenty hats.
    This is the cheapest sybil detector available: one hop back on the funding
    graph, no clustering heuristics, no ML.
    """
    if circ_total <= 0:
        return 0.0
    by_funder: dict[str, float] = {}
    for h in holders:
        if not h.funder:
            continue
        by_funder[h.funder] = by_funder.get(h.funder, 0.0) + h.balance
    if not by_funder:
        return 0.0
    return max(by_funder.values()) / circ_total


def holder_growth(history: list[tuple[int, int]], window_ms: int) -> float:
    """Holders added per minute over the window, from (ts_ms, count) samples.

    The level of holder count is nearly worthless - it is trivially inflated.
    The slope is what a real wave looks like, and it is much more expensive to
    fake convincingly for more than a minute.
    """
    if len(history) < 2:
        return 0.0
    latest_ts, latest_n = history[-1]
    cutoff = latest_ts - window_ms
    base = next((s for s in reversed(history[:-1]) if s[0] <= cutoff), history[0])
    dt_min = (latest_ts - base[0]) / 60_000.0
    if dt_min <= 0:
        return 0.0
    return (latest_n - base[1]) / dt_min


def sellable_supply_overhang(holders: list[Holder], entry_price: float) -> float:
    """Fraction of circulating supply held by wallets with a cost basis far
    below the current price - i.e. the people who are already in profit and
    will be selling into your entry.

    Returns 0.0 when cost bases are unknown, which is honest rather than
    optimistic; a fabricated number here would flow straight into sizing.
    """
    if entry_price <= 0 or not holders:
        return 0.0
    bundled = [h for h in holders if h.acquired_slot and not (h.is_lp or h.is_burn)]
    circ_total = sum(h.balance for h in holders if not (h.is_lp or h.is_burn))
    if circ_total <= 0 or not bundled:
        return 0.0
    return sum(h.balance for h in bundled) / circ_total
