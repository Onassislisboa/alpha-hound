"""Replay backtesting on the bot's own decision log.

This is not a historical-data backtester, and pretending otherwise would be the
most expensive lie in the project. Nobody can honestly reconstruct, for a token
that launched three weeks ago, what the terminal attribution or holder ages
looked like at minute four - the features this strategy depends on are not
recoverable after the fact. A backtest built on reconstructed features would
produce a beautiful equity curve and no information.

So it replays what was actually observed. Every decision, taken or rejected,
was recorded with its full feature vector, and every rejection was shadow
tracked for an hour afterwards. That gives a real, if short-horizon, outcome for
candidates the bot declined - which is exactly what you need to answer the
questions that matter:

    * would different weights have entered different trades?
    * where is the probability threshold that maximises realized PnL?
    * what are my gates costing me?

The horizon is capped by `learning.shadow_track_minutes`, so results understate
the tail of big winners. Reported explicitly rather than quietly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .log import get
from .models import Action, Features
from .portfolio import banked_from_peak
from .scoring import Model, expected_value, normalize, payoff_from_history
from .settings import Config
from .store import Store, features_from_json, unknown_from_json

log = get("backtest")


@dataclass(slots=True)
class Outcome:
    key: str
    symbol: str
    was_entered: bool
    features: Features
    # Realized (for entries) or shadow-observed (for rejections) return.
    observed_return: float
    horizon_capped: bool
    unknown: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Result:
    label: str
    considered: int = 0
    entered: int = 0
    wins: int = 0
    total_return: float = 0.0
    avg_return: float = 0.0
    win_rate: float = 0.0
    horizon_capped: int = 0
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"{self.label:<28} entries={self.entered:<4} win={self.win_rate:>5.1%} "
            f"avg={self.avg_return:>+7.2%} total={self.total_return:>+8.2%}"
        )


def load_outcomes(store: Store, strategy: Config) -> list[Outcome]:
    """Everything with a measurable outcome: closed trades plus resolved
    shadows."""
    outcomes: list[Outcome] = []

    for trade in store.trades():
        outcomes.append(
            Outcome(
                key=trade.key,
                symbol="",
                was_entered=True,
                features=trade.features,
                observed_return=trade.pnl_pct,
                horizon_capped=False,
                unknown=trade.unknown,
            )
        )

    rows = store.conn.execute(
        """SELECT d.key, d.symbol, d.action, d.features, d.unknown, s.counterfactual_pct
           FROM shadow s JOIN decisions d ON d.id = s.decision_id
           WHERE s.resolved = 1"""
    ).fetchall()
    for row in rows:
        outcomes.append(
            Outcome(
                key=row["key"],
                symbol=row["symbol"] or "",
                was_entered=row["action"] == Action.ENTER.value,
                features=features_from_json(row["features"]),
                observed_return=float(row["counterfactual_pct"] or 0.0),
                horizon_capped=True,
                unknown=unknown_from_json(row["unknown"]),
            )
        )
    return outcomes


def simulate_exit(observed_peak_return: float, strategy: Config) -> float:
    """Convert an observed best-case return into what the exit policy would
    actually have banked.

    Shadow tracking records the peak, not the close, so using it raw would
    credit the strategy with perfect timing. Applying the real ladder and stop
    is the difference between a backtest and a fantasy.
    """
    return banked_from_peak(observed_peak_return, strategy)


def evaluate(
    outcomes: list[Outcome],
    model: Model,
    strategy: Config,
    store: Store,
    *,
    min_probability: float,
    min_expected_value: float,
    label: str = "",
) -> Result:
    payoff = payoff_from_history(store.trades(), strategy)
    result = Result(label=label or f"p>={min_probability:.2f} ev>={min_expected_value:.3f}")

    for outcome in outcomes:
        result.considered += 1
        normalized = normalize(outcome.features, outcome.unknown)
        probability, _ = model.probability(normalized)
        ev = expected_value(probability, payoff, outcome.features.round_trip_cost)
        if probability < min_probability or ev < min_expected_value:
            continue

        result.entered += 1
        if outcome.horizon_capped:
            result.horizon_capped += 1
            banked = simulate_exit(outcome.observed_return, strategy)
        else:
            banked = outcome.observed_return
        result.total_return += banked
        if banked > 0:
            result.wins += 1

    if result.entered:
        result.avg_return = result.total_return / result.entered
        result.win_rate = result.wins / result.entered
    if result.horizon_capped:
        result.notes.append(
            f"{result.horizon_capped} outcomes are shadow-horizon capped; the right tail "
            "is understated"
        )
    return result


def sweep(store: Store, strategy: Config, model: Model | None = None) -> list[Result]:
    """Threshold sweep. The useful output is not the best number, it is the
    shape: a flat curve means the threshold does not matter and the edge is
    elsewhere; a sharp peak on 30 trades means you are fitting noise."""
    model = model or Model.load(store)
    outcomes = load_outcomes(store, strategy)
    if not outcomes:
        return []
    base_ev = float(strategy.get("scoring.min_expected_value", 0.035))
    results = []
    for min_p in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        results.append(
            evaluate(
                outcomes,
                model,
                strategy,
                store,
                min_probability=min_p,
                min_expected_value=base_ev,
            )
        )
    return results


def compare_prior_vs_learned(store: Store, strategy: Config) -> list[Result]:
    """Has learning actually helped? Same outcomes, both weight sets."""
    from .learning import prior_model

    outcomes = load_outcomes(store, strategy)
    if not outcomes:
        return []
    min_p = float(strategy.get("scoring.min_probability", 0.56))
    min_ev = float(strategy.get("scoring.min_expected_value", 0.035))
    out = []
    for label, model in (("prior weights", prior_model()), ("learned weights", Model.load(store))):
        out.append(
            evaluate(
                outcomes,
                model,
                strategy,
                store,
                min_probability=min_p,
                min_expected_value=min_ev,
                label=label,
            )
        )
    return out


def dump(results: list[Result]) -> str:
    return json.dumps(
        [
            {
                "label": r.label,
                "considered": r.considered,
                "entered": r.entered,
                "win_rate": round(r.win_rate, 4),
                "avg_return": round(r.avg_return, 4),
                "total_return": round(r.total_return, 4),
                "notes": r.notes,
            }
            for r in results
        ],
        indent=2,
    )
