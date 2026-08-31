"""Chart features.

Deliberately simple, and that is a decision rather than a shortcut. A token
that is forty minutes old has forty one-minute candles. Any indicator with a
lookback longer than the asset's life is fitting noise, and the ones that
"work" on that data work because they are reading the same three candles the
naive features read, with more steps and a worse name.

Everything here is a pure function of candles, which is what makes the
backtester and the live engine share one code path instead of two that drift.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev

from ..models import Candle


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def ret_over(candles: list[Candle], n: int) -> float:
    """Return over the last n candles. Uses the open of the reference candle,
    not its close, so the number means "what a buyer n minutes ago has now"."""
    if len(candles) < 2:
        return 0.0
    ref = candles[-min(n, len(candles))]
    base = ref.open or ref.close
    return _safe_div(candles[-1].close - base, base)


def vwap(candles: list[Candle]) -> float:
    num = sum(((c.high + c.low + c.close) / 3.0) * c.volume for c in candles)
    den = sum(c.volume for c in candles)
    if den <= 0:
        return fmean([c.close for c in candles]) if candles else 0.0
    return num / den


def vwap_deviation(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    v = vwap(candles)
    return _safe_div(candles[-1].close - v, v)


def atr_pct(candles: list[Candle], window: int = 14) -> float:
    """True range averaged over the window, normalized by price. Used as the
    volatility unit for stops: a fixed percentage stop is either noise-bait on
    a calm token or a coin flip on a violent one."""
    if len(candles) < 2:
        return 0.0
    trs = []
    for prev, cur in zip(candles[-window - 1 :], candles[-window:]):
        trs.append(
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
        )
    if not trs:
        return 0.0
    return _safe_div(fmean(trs), candles[-1].close)


def breakout(candles: list[Candle], lookback: int = 10) -> float:
    """1.0 if the last close cleared the prior `lookback` highs, else the
    fraction of the way there. A soft value beats a boolean because the model
    can learn where on that ramp the edge actually lives."""
    if len(candles) < lookback + 1:
        return 0.0
    prior = candles[-lookback - 1 : -1]
    ceiling = max(c.high for c in prior)
    if ceiling <= 0:
        return 0.0
    ratio = candles[-1].close / ceiling
    # The ramp starts at the ceiling, not below it. Sitting just under the
    # prior high is not a breakout, and crediting it as a partial one is how a
    # range-bound token reads as momentum.
    return max(0.0, min(1.0, (ratio - 1.0) / 0.03))


def volume_zscore(candles: list[Candle], window: int = 20) -> float:
    if len(candles) < 4:
        return 0.0
    vols = [c.volume for c in candles[-window - 1 : -1]]
    if len(vols) < 3:
        return 0.0
    mu = fmean(vols)
    sd = pstdev(vols)
    if sd <= 0:
        return 0.0 if candles[-1].volume <= mu else 3.0
    return max(-5.0, min(5.0, (candles[-1].volume - mu) / sd))


def upper_wick(candles: list[Candle]) -> float:
    """Share of the last candle's range that is upper wick. Rejection at highs."""
    if not candles:
        return 0.0
    c = candles[-1]
    rng = c.high - c.low
    if rng <= 0:
        return 0.0
    return max(0.0, (c.high - max(c.open, c.close)) / rng)


def higher_lows(candles: list[Candle], n: int = 4) -> float:
    """1.0 if the last n lows are non-decreasing (accumulation). 0 if collapsing."""
    if len(candles) < n:
        return 0.0
    lows = [c.low for c in candles[-n:]]
    steps = list(zip(lows, lows[1:]))
    if not steps:
        return 0.0
    return sum(1.0 for a, b in steps if b >= a * 0.998) / len(steps)


def last_range_vs_atr(candles: list[Candle]) -> float:
    """Last candle range / own ATR. >2 is a spike relative to this token, not a global cap."""
    if len(candles) < 3:
        return 1.0
    c = candles[-1]
    rng = c.high - c.low
    unit = atr_pct(candles[:-1] or candles) * (c.close or 1.0)
    if unit <= 0:
        return 1.0
    return rng / unit


def body_ratio(candles: list[Candle]) -> float:
    """Signed conviction of the last candle: how much of the range the body
    took. A long upper wick on huge volume is distribution wearing a green
    candle's clothes."""
    if not candles:
        return 0.0
    c = candles[-1]
    rng = c.high - c.low
    if rng <= 0:
        return 0.0
    return max(-1.0, min(1.0, (c.close - c.open) / rng))


def parabolic(candles: list[Candle], window: int, threshold: float) -> float:
    """1.0 when the token has already gone vertical inside the window.

    This is the single most expensive mistake in memecoin trading: the chart
    that convinces you is the chart that already paid someone else. It enters
    the model as a feature rather than a hard gate so the learner can price it,
    but the prior weight is strongly negative.
    """
    if len(candles) < 2 or threshold <= 0:
        return 0.0
    seg = candles[-min(window + 1, len(candles)) :]
    low = min(c.low for c in seg) or seg[0].close
    if low <= 0:
        return 0.0
    move = seg[-1].close / low - 1.0
    return max(0.0, min(1.0, move / threshold))


def drawdown_from_peak(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    peak = max(c.high for c in candles)
    return _safe_div(peak - candles[-1].close, peak)


def extract(candles: list[Candle], cfg: dict) -> dict[str, float]:
    """Assemble the chart slice of the feature vector."""
    if not candles:
        return {
            "ret_5m": 0.0,
            "ret_15m": 0.0,
            "vwap_dev": 0.0,
            "atr_pct": 0.0,
            "breakout": 0.0,
            "volume_z": 0.0,
            "body_ratio": 0.0,
            "parabolic": 0.0,
        }
    per_min = max(1, int(60 // max(1, int(cfg.get("candle_seconds", 60) // 60 or 1))))
    return {
        "ret_5m": ret_over(candles, 5 * per_min),
        "ret_15m": ret_over(candles, 15 * per_min),
        "vwap_dev": vwap_deviation(candles),
        "atr_pct": atr_pct(candles),
        "breakout": breakout(candles, int(cfg.get("breakout_lookback", 10))),
        "volume_z": volume_zscore(candles, int(cfg.get("volume_zscore_window", 20))),
        "body_ratio": body_ratio(candles),
        "parabolic": parabolic(
            candles,
            int(cfg.get("parabolic_window_candles", 5)),
            float(cfg.get("parabolic_return_threshold", 1.5)),
        ),
    }


def candles_from_trades(
    trades: list[tuple[int, float, float]], bucket_seconds: int = 60
) -> list[Candle]:
    """Build candles from (ts_ms, price, size_usd) trades.

    Needed because the tokens worth trading are too young for any candle API to
    have data on them. The first ten minutes of a launch exist only as a trade
    stream, and those ten minutes are the entire opportunity.
    """
    if not trades:
        return []
    step = bucket_seconds * 1000
    buckets: dict[int, list[tuple[float, float]]] = {}
    for ts, price, size in sorted(trades):
        buckets.setdefault(ts - (ts % step), []).append((price, size))
    out: list[Candle] = []
    for ts in sorted(buckets):
        prices = [p for p, _ in buckets[ts]]
        out.append(
            Candle(
                ts=ts,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(s for _, s in buckets[ts]),
            )
        )
    return out


def sharpe_like(returns: list[float]) -> float:
    """Mean over stdev of a return series. Used by the learner to compare
    weight versions, not to score tokens."""
    if len(returns) < 3:
        return 0.0
    sd = pstdev(returns)
    if sd <= 0:
        return 0.0
    return fmean(returns) / sd * math.sqrt(len(returns))
