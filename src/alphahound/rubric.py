"""Stage-2 0–10 rubric. Vetoes still sit in scoring.evaluate_gates.

Five categories, fixed weights. Missing data scores 5 (neutral), never 10.
The logistic model still sizes the trade; this only decides if the token is
even a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import Features, now_ms
from .signals import chart, flow

if TYPE_CHECKING:
    from .settings import Config
    from .signals import Enrichment
    from .store import Store

WEIGHTS = {
    "dist": 0.20,
    "crowd": 0.25,
    "flow": 0.20,
    "chart": 0.25,
    "narrative": 0.10,
}


def _clip10(x: float) -> float:
    return max(0.0, min(10.0, x))


def _dump_weight(store: Store, addresses: list[str]) -> float:
    """1.0 if unknown/smart; <<1 if those wallets dumped our previous copies."""
    worst = 1.0
    for addr in addresses:
        rec = store.wallet_record(addr)
        if rec is None or rec["trades"] < 2:
            continue
        wr = rec["wins"] / rec["trades"]
        if rec["pnl_usd"] < 0 or wr < 0.4:
            worst = min(worst, 0.25)
        elif rec["is_smart"]:
            worst = min(worst, 1.0)
        else:
            worst = min(worst, 0.6)
    return worst


def _dist(f: Features, unknown: set[str]) -> float:
    if "top10_pct" in unknown and "gini" in unknown:
        return 5.0
    score = 8.0
    if "top10_pct" not in unknown:
        score -= max(0.0, f.top10_pct - 0.25) * 16.0
    if "gini" not in unknown:
        score -= max(0.0, f.gini - 0.45) * 8.0
    if "cluster_pct" not in unknown:
        score -= f.cluster_pct * 10.0
    if "fresh_wallet_pct" not in unknown:
        score -= max(0.0, f.fresh_wallet_pct - 0.25) * 6.0
    if "known_holder_pct" not in unknown and f.known_holder_pct > 0.15:
        score += 1.5
    return _clip10(score)


def _crowd(f: Features, unknown: set[str], enr: Enrichment, store: Store) -> float:
    wallets = list((enr.crowd or {}).get("wallets") or [])
    dump = _dump_weight(store, wallets)
    kols = (enr.crowd or {}).get("kols") or []
    score = 4.0
    if "copy_signal" not in unknown and f.copy_signal >= 1.0:
        score += 3.0 * dump
    if "smart_money_buys" not in unknown and f.smart_money_buys > 0:
        score += min(2.0, f.smart_money_buys) * dump
    if "fomo_net_flow" not in unknown:
        if f.fomo_net_flow > 0:
            score += 1.5 * dump
        elif f.fomo_net_flow < 0:
            score -= 2.0
    if "whale_net_flow" not in unknown:
        if f.whale_net_flow > 0:
            score += 1.0 * dump
        elif f.whale_net_flow < 0:
            score -= 2.0
    if kols:
        score += min(1.5, 0.5 * len(kols)) * dump
    if dump < 0.5:
        score -= 1.5
    return _clip10(score)


def _flow(f: Features, unknown: set[str], enr: Enrichment) -> float:
    if "unique_buyers_5m" in unknown and "buy_sell_ratio" in unknown:
        return 5.0
    score = 5.0
    if "unique_buyers_5m" not in unknown:
        score += min(3.0, f.unique_buyers_5m / 12.0)
    if "buy_sell_ratio" not in unknown:
        score += min(2.0, max(-2.0, (f.buy_sell_ratio - 1.0) * 1.5))
    if "bot_share" not in unknown:
        score -= f.bot_share * 6.0
    trades = list(enr.trades or [])
    if trades:
        score -= flow.wash_ratio(trades) * 5.0
        accel = flow.buyer_acceleration(trades, now_ms())
        if accel < 1.0:
            score -= (1.0 - accel) * 3.0
        conc = flow.whale_concentration(trades)
        if conc > 0.45:
            score -= (conc - 0.45) * 6.0
    return _clip10(score)


def _chart(f: Features, unknown: set[str], enr: Enrichment) -> float:
    candles = list(enr.candles or [])
    if not candles and "parabolic" in unknown and "body_ratio" in unknown:
        return 5.0
    score = 5.5
    body_known = "body_ratio" not in unknown
    para_known = "parabolic" not in unknown
    if para_known:
        score -= f.parabolic * 5.0
    if body_known:
        if f.body_ratio > 0.35:
            score += 1.5
        elif f.body_ratio < -0.2:
            score -= 1.5
    if candles:
        score += chart.higher_lows(candles) * 2.0
        wick = chart.upper_wick(candles)
        if wick > 0.45 and body_known and f.body_ratio < 0.25:
            score -= 2.5
        spike = chart.last_range_vs_atr(candles)
        if spike > 2.0 and para_known and f.parabolic > 0.4:
            score -= 1.5
    if "vwap_dev" not in unknown and f.vwap_dev > 0.25:
        score -= 1.5
    return _clip10(score)


def _narrative(f: Features, unknown: set[str], enr: Enrichment) -> float:
    c = enr.candidate
    tw = enr.twitter or {}
    score = 4.0
    if c.dex_paid:
        score += 2.5
    if c.dex_photo:
        score += 1.0
    if c.dex_aligned:
        score += 1.5
    util = str(tw.get("utility") or "")
    if util == "claims utility":
        score += 0.8
    elif util == "meme":
        score += 0.2
    if "twitter_inst" not in unknown:
        score += f.twitter_inst * 2.5
    if "twitter_fresh" not in unknown and f.twitter_fresh >= 1.0:
        score += 1.5
    llm = tw.get("llm")
    if isinstance(llm, (int, float)):
        score = 0.5 * score + 0.5 * _clip10(float(llm))
    return _clip10(score)


@dataclass(slots=True)
class Rubric:
    dist: float
    crowd: float
    flow: float
    chart: float
    narrative: float
    total: float

    def as_visor(self) -> dict:
        return {
            "total": round(self.total, 1),
            "dist": round(self.dist, 1),
            "crowd": round(self.crowd, 1),
            "flow": round(self.flow, 1),
            "chart": round(self.chart, 1),
            "narrative": round(self.narrative, 1),
        }


def grade(enr: Enrichment, store: Store, strategy: Config | None = None) -> Rubric:
    f = enr.features
    unknown = enr.unknown
    weights = dict(WEIGHTS)
    if strategy is not None:
        raw = strategy.get("scoring.rubric_weights")
        if isinstance(raw, dict):
            weights.update({k: float(v) for k, v in raw.items() if k in WEIGHTS})
    cats = {
        "dist": _dist(f, unknown),
        "crowd": _crowd(f, unknown, enr, store),
        "flow": _flow(f, unknown, enr),
        "chart": _chart(f, unknown, enr),
        "narrative": _narrative(f, unknown, enr),
    }
    total = sum(cats[k] * weights[k] for k in WEIGHTS)
    return Rubric(total=total, **cats)
