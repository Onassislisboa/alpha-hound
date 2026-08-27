"""Order flow features.

Price tells you what happened. Flow tells you who made it happen and whether
they are finished. On a token whose entire history is twenty minutes long, flow
is the higher-information source by a wide margin, because price on thin
liquidity is mostly an artifact of whoever traded last.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from ..models import Side


@dataclass(slots=True)
class Trade:
    ts_ms: int
    side: Side
    price: float
    size_usd: float
    wallet: str = ""


def _window(trades: list[Trade], now_ms: int, minutes: float) -> list[Trade]:
    cutoff = now_ms - int(minutes * 60_000)
    return [t for t in trades if t.ts_ms >= cutoff]


def unique_buyers(trades: list[Trade]) -> int:
    return len({t.wallet for t in trades if t.side is Side.BUY and t.wallet})


def buy_sell_ratio(trades: list[Trade]) -> float:
    """USD-weighted, not count-weighted.

    Count ratios are the easiest number in crypto to fake: a hundred $2 buys
    cost nothing and look like a stampede. Dollars are harder to fake because
    faking them requires actually having them.
    """
    buys = sum(t.size_usd for t in trades if t.side is Side.BUY)
    sells = sum(t.size_usd for t in trades if t.side is Side.SELL)
    if sells <= 0:
        return 3.0 if buys > 0 else 1.0
    return min(10.0, buys / sells)


def net_inflow_usd(trades: list[Trade]) -> float:
    return sum(t.size_usd if t.side is Side.BUY else -t.size_usd for t in trades)


def avg_buy_size(trades: list[Trade]) -> float:
    sizes = [t.size_usd for t in trades if t.side is Side.BUY and t.size_usd > 0]
    return fmean(sizes) if sizes else 0.0


def buyer_acceleration(trades: list[Trade], now_ms: int) -> float:
    """Unique buyers in the last minute versus the average of the four before.

    >1 means the crowd is still arriving. <1 means you are looking at the tail
    of a move that already happened, which is the most common way a good signal
    turns into a bad trade.
    """
    recent = unique_buyers(_window(trades, now_ms, 1))
    prior_trades = [t for t in trades if now_ms - 300_000 <= t.ts_ms < now_ms - 60_000]
    prior = unique_buyers(prior_trades) / 4.0
    if prior <= 0:
        return 2.0 if recent > 0 else 0.0
    return min(5.0, recent / prior)


def whale_concentration(trades: list[Trade]) -> float:
    """Share of buy volume from the single largest buyer. High values mean the
    move is one wallet, and one wallet can leave."""
    buys: dict[str, float] = {}
    for t in trades:
        if t.side is Side.BUY and t.wallet:
            buys[t.wallet] = buys.get(t.wallet, 0.0) + t.size_usd
    total = sum(buys.values())
    if total <= 0:
        return 0.0
    return max(buys.values()) / total


def smart_money_buys(trades: list[Trade], smart: set[str]) -> int:
    if not smart:
        return 0
    return len({t.wallet for t in trades if t.side is Side.BUY and t.wallet in smart})


def sniper_exit_pressure(trades: list[Trade], snipers: set[str]) -> float:
    """Share of sell volume coming from launch snipers.

    When this turns up it is the clearest exit signal available on a young
    token: the wallets with the lowest cost basis have decided, and they know
    more about the token's distribution than you do.
    """
    if not snipers:
        return 0.0
    sells = [t for t in trades if t.side is Side.SELL]
    total = sum(t.size_usd for t in sells)
    if total <= 0:
        return 0.0
    return sum(t.size_usd for t in sells if t.wallet in snipers) / total


def extract(
    trades: list[Trade],
    now_ms: int,
    *,
    smart: set[str] | None = None,
    window_minutes: float = 5.0,
) -> dict[str, float]:
    win = _window(trades, now_ms, window_minutes)
    return {
        "unique_buyers_5m": float(unique_buyers(win)),
        "buy_sell_ratio": buy_sell_ratio(win),
        "net_inflow_usd_5m": net_inflow_usd(win),
        "avg_buy_size_usd": avg_buy_size(win),
        "smart_money_buys": float(smart_money_buys(win, smart or set())),
    }
