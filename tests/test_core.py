"""The runnable check.

Covers the logic that would lose money silently if it broke: distribution
maths, the sign of the terminal-attribution thesis, gate vetoes, Kelly sizing,
the exit ladder's fraction accounting, and the postmortem taxonomy.

Stdlib only, no fixtures, no plugins:

    python -m unittest discover -s tests -v
    python tests/test_core.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphahound import backtest, learning  # noqa: E402
from alphahound.models import (  # noqa: E402
    Candidate,
    Candle,
    Chain,
    ErrorClass,
    ExitReason,
    Features,
    Position,
    RoundTrip,
    Score,
    Side,
    TradeRecord,
    VenueId,
    now_ms,
)
from alphahound.portfolio import PositionManager, banked_from_peak  # noqa: E402
from alphahound.risk import RiskEngine, kelly_fraction  # noqa: E402
from alphahound.scoring import (  # noqa: E402
    Model,
    Payoff,
    evaluate_gates,
    expected_value,
    normalize,
    payoff_from_config,
)
from alphahound.settings import Config, load_strategy  # noqa: E402
from alphahound.signals import Enrichment  # noqa: E402
from alphahound.signals import chart, flow  # noqa: E402
from alphahound.signals.distribution import Holder, analyze, gini  # noqa: E402
from alphahound.signals.terminals import (  # noqa: E402
    BuyerTx,
    ShareTracker,
    TerminalRegistry,
    attribute,
    discover_fee_accounts,
)
from alphahound.store import Store  # noqa: E402

STRATEGY = load_strategy()


def make_store() -> tuple[Store, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    return Store(Path(tmp.name)), tmp


def candles(closes: list[float], volume: float = 1000.0) -> list[Candle]:
    out = []
    for i, close in enumerate(closes):
        prev = closes[i - 1] if i else close
        out.append(
            Candle(
                ts=i * 60_000,
                open=prev,
                high=max(prev, close) * 1.01,
                low=min(prev, close) * 0.99,
                close=close,
                volume=volume,
            )
        )
    return out


class TestDistribution(unittest.TestCase):
    def test_gini_separates_shapes_that_top10_hides(self):
        # Both distributions have 25 holders and 100 units of supply, and a
        # top-10 share that looks similar. Gini is what tells them apart.
        flat = [4.0] * 25
        one_whale = [40.0] + [2.5] * 24
        ten_medium = [4.0] * 10 + [2.5] * 24
        self.assertLess(gini(flat), 0.05)
        self.assertGreater(gini(one_whale), gini(ten_medium) + 0.15)

    def test_lp_and_burn_excluded_from_concentration(self):
        holders = [
            Holder("pool", 800.0, is_lp=True),
            Holder("burn", 100.0, is_burn=True),
            Holder("a", 50.0),
            Holder("b", 30.0),
            Holder("c", 20.0),
        ]
        stats = analyze(holders, now_ms=now_ms())
        self.assertEqual(stats.holder_count, 3)
        # 50 of 100 circulating, not 50 of 1000 total. Counting the pool as a
        # holder makes every healthy token look concentrated.
        self.assertAlmostEqual(stats.top1_pct, 0.5, places=6)
        self.assertAlmostEqual(stats.lp_pct, 0.8, places=6)

    def test_funding_cluster_catches_one_holder_in_twenty_hats(self):
        holders = [Holder(f"w{i}", 5.0, funder="sybil") for i in range(18)] + [
            Holder("real", 10.0, funder="cex")
        ]
        stats = analyze(holders, now_ms=now_ms())
        self.assertGreater(stats.largest_funding_cluster_pct, 0.85)

    def test_bundle_pct_not_reported_without_a_launch_slot(self):
        holders = [Holder("a", 100.0, acquired_slot=10)]
        stats = analyze(holders, now_ms=now_ms(), launch_slot=0)
        self.assertEqual(stats.bundle_pct, 0.0)
        self.assertTrue(any("launch slot" in note for note in stats.notes))


class TestChart(unittest.TestCase):
    def test_breakout_ramps_rather_than_flipping(self):
        flat = candles([1.0] * 12)
        self.assertLess(chart.breakout(flat, 10), 0.5)
        broke = candles([1.0] * 11 + [1.20])
        self.assertGreater(chart.breakout(broke, 10), 0.9)

    def test_parabolic_flags_the_chart_that_already_paid_someone_else(self):
        vertical = candles([1.0, 1.4, 1.9, 2.4, 3.0, 3.4])
        self.assertGreater(chart.parabolic(vertical, 5, 1.5), 0.9)
        calm = candles([1.0, 1.01, 1.02, 1.03, 1.02, 1.04])
        self.assertLess(chart.parabolic(calm, 5, 1.5), 0.1)

    def test_candles_built_from_a_trade_stream(self):
        trades = [
            (0, 1.0, 100.0),
            (10_000, 1.2, 50.0),
            (30_000, 0.9, 25.0),
            (61_000, 1.1, 10.0),
        ]
        built = chart.candles_from_trades(trades, 60)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0].high, 1.2)
        self.assertEqual(built[0].low, 0.9)
        self.assertEqual(built[0].volume, 175.0)

    def test_empty_candles_do_not_explode(self):
        values = chart.extract([], STRATEGY.section("chart"))
        self.assertEqual(values["ret_5m"], 0.0)


class TestFlow(unittest.TestCase):
    def test_buy_sell_ratio_is_usd_weighted_not_count_weighted(self):
        now = now_ms()
        spam = [flow.Trade(now, Side.BUY, 1.0, 2.0, f"w{i}") for i in range(100)]
        spam.append(flow.Trade(now, Side.SELL, 1.0, 5000.0, "whale"))
        # A hundred two-dollar buys must not outvote a five-thousand-dollar sell.
        self.assertLess(flow.buy_sell_ratio(spam), 1.0)

    def test_whale_concentration(self):
        now = now_ms()
        trades = [
            flow.Trade(now, Side.BUY, 1.0, 9000.0, "whale"),
            flow.Trade(now, Side.BUY, 1.0, 1000.0, "retail"),
        ]
        self.assertAlmostEqual(flow.whale_concentration(trades), 0.9, places=6)


class TestTerminalAttribution(unittest.TestCase):
    def registry(self) -> TerminalRegistry:
        return TerminalRegistry(
            [
                {"name": "axiom", "class": "retail", "fee_accounts": ["AXFEE"]},
                {"name": "trojan", "class": "bot", "fee_accounts": ["TJFEE"]},
            ],
            {"jupiter_v6": "JUPPROG"},
        )

    def test_retail_beats_neutral_on_the_same_transaction(self):
        # Every terminal swap also touches Jupiter. Matching greedily would
        # label all of them "jupiter" and destroy the only useful signal.
        label, klass = self.registry().classify(["JUPPROG", "AXFEE", "other"])
        self.assertEqual((label, klass), ("axiom", "retail"))

    def test_attribution_dedups_by_buyer(self):
        txs = [BuyerTx("bot1", ["TJFEE"], 0, 10.0) for _ in range(20)]
        txs.append(BuyerTx("human", ["AXFEE"], 0, 10.0))
        result = attribute(txs, self.registry())
        self.assertEqual(result.total, 2)
        self.assertAlmostEqual(result.class_share("retail"), 0.5, places=6)

    def test_share_tracker_reports_no_wave_from_one_sample(self):
        tracker = ShareTracker()
        tracker.observe("k", 1_000_000, {"retail": 0.4})
        self.assertEqual(tracker.delta("k", "retail", 300_000), 0.0)
        tracker.observe("k", 1_600_000, {"retail": 0.6})
        self.assertAlmostEqual(tracker.delta("k", "retail", 300_000), 0.2, places=6)

    def test_discovery_ranks_breadth_of_buyers_not_transaction_count(self):
        # FEEA is paid by twelve unrelated wallets: the shape of a real fee
        # account. "noise" only shows up in a few of them, and "busybot" pays
        # FEEB sixty times but is a single wallet - which is exactly the thing
        # a transaction-count ranking would put on top.
        txs = [BuyerTx(f"w{i}", ["FEEA"] + (["noise"] if i < 3 else []), 0) for i in range(12)]
        txs += [BuyerTx("busybot", ["FEEB"], 0) for _ in range(60)]
        found = discover_fee_accounts(txs, self.registry(), min_distinct_buyers=8)
        self.assertTrue(found)
        self.assertEqual(found[0].address, "FEEA")
        self.assertNotIn("FEEB", [c.address for c in found])


class TestScoring(unittest.TestCase):
    def test_unknown_features_are_neutral_not_favourable(self):
        features = Features(liquidity_usd=0.0, holder_count=0.0)
        blind = normalize(features, unknown={"liquidity_usd", "holder_count"})
        self.assertEqual(blind["liquidity_usd"], 0.0)
        self.assertEqual(blind["holder_count"], 0.0)
        # Measured-and-terrible must score worse than unmeasured.
        seen = normalize(features)
        self.assertLess(seen["liquidity_usd"], blind["liquidity_usd"])

    def test_the_thesis_level_and_derivative_have_opposite_signs(self):
        model = Model()
        base = dict(
            unique_buyers_5m=60.0,
            buy_sell_ratio=2.0,
            liquidity_usd=60_000.0,
            holder_count=400.0,
        )
        early = Features(retail_share=0.10, retail_share_delta_5m=0.12, **base)
        late = Features(retail_share=0.60, retail_share_delta_5m=0.00, **base)
        p_early, _ = model.probability(normalize(early))
        p_late, _ = model.probability(normalize(late))
        self.assertGreater(
            p_early,
            p_late,
            "arriving before the retail wave must score above arriving after it",
        )

    def test_parabolic_entries_are_penalised(self):
        model = Model()
        calm = Features(breakout=0.9, volume_z=2.0, parabolic=0.0)
        vertical = Features(breakout=0.9, volume_z=2.0, parabolic=1.0)
        self.assertGreater(
            model.probability(normalize(calm))[0], model.probability(normalize(vertical))[0]
        )

    def test_expected_value_subtracts_costs(self):
        payoff = Payoff(avg_win=0.40, avg_loss=0.25, samples=100, from_history=True)
        free = expected_value(0.60, payoff, 0.0)
        costly = expected_value(0.60, payoff, 0.06)
        self.assertAlmostEqual(free - costly, 0.06, places=9)
        # A 60% win rate is not enough when the round trip eats the edge.
        self.assertLess(expected_value(0.60, payoff, 0.16), 0.0)

    def test_prior_payoff_does_not_assume_every_winner_hits_the_top_rung(self):
        strategy = Config(
            {
                "exits": {
                    "take_profit_ladder": [[1.35, 0.34], [2.0, 0.33], [3.5, 0.33]],
                    "stop_loss_pct": 0.28,
                    "trailing_stop_pct": 0.22,
                },
                "scoring": {"prior_winner_ladder_reach": [[1, 0.6], [2, 0.25], [3, 0.15]]},
            }
        )
        prior = payoff_from_config(strategy)
        naive = sum((m - 1.0) * f for m, f in [(1.35, 0.34), (2.0, 0.33), (3.5, 0.33)])

        # The naive whole-ladder sum is what makes a fresh bot fearless.
        self.assertGreater(naive, 1.2)
        self.assertLess(prior.avg_win, naive / 2)
        # ...but not so pessimistic that a genuinely good signal cannot clear
        # the EV floor, or the bot never trades and never learns.
        self.assertGreater(prior.avg_win, banked_from_peak(0.35, strategy))
        floor = float(strategy.get("scoring.min_expected_value", 0.035))
        self.assertGreater(expected_value(0.62, prior, 0.05), floor)
        self.assertLess(expected_value(0.50, prior, 0.05), floor)

    def test_weights_are_stored_by_name_so_new_features_are_safe(self):
        model = Model({"a_brand_new_feature": 0.5})
        self.assertEqual(model.weights["a_brand_new_feature"], 0.5)
        self.assertIn("bundle_pct", model.weights)


class TestGates(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def enrichment(self, **overrides) -> Enrichment:
        features = Features(
            liquidity_usd=80_000.0,
            holder_count=500.0,
            top10_pct=0.30,
            dev_holding_pct=0.0,
            bundle_pct=0.05,
            fresh_wallet_pct=0.20,
            bot_share=0.10,
            retail_share=0.15,
            round_trip_cost=0.02,
        )
        for name, value in overrides.items():
            setattr(features, name, value)
        candidate = Candidate(chain=Chain.SOLANA, address="mint", price_usd=1.0)
        return Enrichment(
            candidate=candidate,
            features=features,
            round_trip=RoundTrip(ok=True, sell_slippage=0.03, total_cost_pct=0.02),
        )

    def test_clean_candidate_passes(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        vetoes, abstained = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertEqual(vetoes, [], msg=f"unexpected vetoes; abstained={abstained}")

    def test_unsellable_token_is_vetoed(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.round_trip = RoundTrip(ok=True, sell_slippage=0.60, total_cost_pct=0.02)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("sellable" in v for v in vetoes))

    def test_live_mode_vetoes_what_it_cannot_measure(self):
        enr = self.enrichment()
        enr.mint = _FakeMint(None, None)
        enr.unknown.add("holder_count")
        live_vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        paper_vetoes, paper_abstained = evaluate_gates(enr, STRATEGY, self.store, live=False)
        self.assertTrue(any("unmeasured" in v for v in live_vetoes))
        self.assertEqual(paper_vetoes, [])
        self.assertTrue(any("holder_count" in a for a in paper_abstained))

    def test_live_mint_authority_blocks_entry(self):
        enr = self.enrichment()
        enr.mint = _FakeMint("someauthority", None)
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("mint_authority" in v for v in vetoes))

    def test_a_gate_override_takes_effect_without_a_redeploy(self):
        enr = self.enrichment(liquidity_usd=20_000.0)
        enr.mint = _FakeMint(None, None)
        self.assertEqual(evaluate_gates(enr, STRATEGY, self.store, live=True)[0], [])
        self.store.set_param("gates.min_liquidity_usd", 50_000.0, "test")
        vetoes, _ = evaluate_gates(enr, STRATEGY, self.store, live=True)
        self.assertTrue(any("liquidity" in v for v in vetoes))


class _FakeMint:
    def __init__(self, mint_authority, freeze_authority):
        self.mint_authority = mint_authority
        self.freeze_authority = freeze_authority


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()
        self.risk = RiskEngine(STRATEGY, self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_kelly_refuses_a_negative_edge(self):
        payoff = Payoff(avg_win=0.30, avg_loss=0.30, samples=50, from_history=True)
        self.assertEqual(kelly_fraction(0.40, payoff), 0.0)
        self.assertGreater(kelly_fraction(0.70, payoff), 0.0)

    def test_thin_pool_binds_before_the_equity_cap(self):
        score = Score(probability=0.90, expected_value=0.5)
        payoff = Payoff(avg_win=1.0, avg_loss=0.25, samples=50, from_history=True)
        thin = Candidate(chain=Chain.SOLANA, address="a", liquidity_usd=2_000.0, price_usd=1.0)
        sizing = self.risk.size(thin, score, payoff, [])
        self.assertTrue(sizing.allowed)
        self.assertEqual(sizing.binding_constraint, "pool_liquidity")
        self.assertLessEqual(sizing.size_usd, 2_000.0 * 0.010 + 1e-9)

    def test_kill_switch_stops_sizing(self):
        self.risk.engage_kill_switch("test")
        sizing = self.risk.size(
            Candidate(chain=Chain.SOLANA, address="a", liquidity_usd=10**6, price_usd=1.0),
            Score(probability=0.99, expected_value=1.0),
            Payoff(1.0, 0.2, 50, True),
            [],
        )
        self.assertFalse(sizing.allowed)
        self.assertIn("kill switch", sizing.reason)

    def test_equity_shrinks_with_realized_losses(self):
        before = self.risk.equity()
        self.store.record_trade(_trade(pnl=-200.0))
        self.assertAlmostEqual(self.risk.equity(), before - 200.0, places=6)


class TestExits(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()
        self.manager = PositionManager(STRATEGY, self.store)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def position(self) -> Position:
        candidate = Candidate(
            chain=Chain.SOLANA, address="a", price_usd=1.0, liquidity_usd=100_000.0
        )
        return Position(
            candidate=candidate,
            venue=VenueId.PAPER,
            entry_price=1.0,
            size_usd=100.0,
            tokens=100.0,
        )

    def test_ladder_fractions_are_of_the_original_position(self):
        position = self.position()
        # A jump straight past every rung must sell the whole ladder, not one
        # rung, and the fractions must convert correctly to "of remaining".
        orders = self.manager.evaluate(position, 5.0, 100_000.0)
        self.assertEqual(len(orders), 3)
        remaining = 1.0
        sold = 0.0
        for order in orders:
            sold += remaining * order.fraction
            remaining -= remaining * order.fraction
        self.assertAlmostEqual(sold, 1.0, places=6)

    def test_liquidity_drain_outranks_take_profit(self):
        position = self.position()
        self.manager.observe(position, 1.0, 100_000.0)
        orders = self.manager.evaluate(position, 4.0, 40_000.0)
        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0].reason, ExitReason.LIQUIDITY_DRAIN)
        self.assertEqual(orders[0].fraction, 1.0)

    def test_stop_loss(self):
        orders = self.manager.evaluate(self.position(), 0.50, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.STOP_LOSS)

    def test_trailing_stop_only_arms_after_a_rung_fills(self):
        position = self.position()
        self.manager.observe(position, 1.30, 100_000.0)
        # Up 30%, then back to break-even: rung one never filled, so the trail
        # is not armed and the hard stop has not been hit.
        self.assertEqual(self.manager.evaluate(position, 1.0, 100_000.0), [])

        armed = self.position()
        self.manager.evaluate(armed, 1.40, 100_000.0)
        self.assertTrue(armed.trailing_active)
        orders = self.manager.evaluate(armed, 1.40 * 0.70, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.TRAILING_STOP)

    def test_time_stop_releases_dead_capital(self):
        position = self.position()
        position.opened_at_ms = now_ms() - 90 * 60_000
        orders = self.manager.evaluate(position, 1.02, 100_000.0)
        self.assertIs(orders[0].reason, ExitReason.TIME_STOP)

    def test_excursions(self):
        position = self.position()
        self.manager.observe(position, 3.0, 100_000.0)
        self.manager.observe(position, 0.6, 100_000.0)
        mfe, mae = PositionManager.excursions(position)
        self.assertAlmostEqual(mfe, 2.0, places=6)
        self.assertAlmostEqual(mae, -0.4, places=6)


def _trade(
    pnl: float = -50.0,
    *,
    exit_reason: ExitReason = ExitReason.STOP_LOSS,
    mfe: float = 0.0,
    entry_price: float = 1.0,
    signal_price: float = 1.0,
    slippage: float = 0.0,
    features: Features | None = None,
    error_class: ErrorClass = ErrorClass.NO_EDGE,
) -> TradeRecord:
    return TradeRecord(
        key="solana:mint",
        chain=Chain.SOLANA,
        venue=VenueId.PAPER,
        opened_at_ms=now_ms() - 60_000,
        closed_at_ms=now_ms(),
        entry_price=entry_price,
        exit_price=entry_price * (1.0 + pnl / 100.0),
        signal_price=signal_price,
        size_usd=100.0,
        pnl_usd=pnl,
        fees_usd=1.0,
        exit_reason=exit_reason,
        error_class=error_class,
        features=features or Features(),
        weights_version=0,
        max_favorable_excursion=mfe,
        entry_slippage=slippage,
    )


class TestPostmortem(unittest.TestCase):
    def test_liquidity_drain_is_a_rug(self):
        trade = _trade(exit_reason=ExitReason.LIQUIDITY_DRAIN)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.RUG)

    def test_late_entry_detected_from_the_signal_price_gap(self):
        trade = _trade(entry_price=1.20, signal_price=1.0)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.LATE_ENTRY)

    def test_slippage_blowout_outranks_late_entry(self):
        trade = _trade(entry_price=1.20, signal_price=1.0, slippage=0.35)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.SLIPPAGE_BLOWOUT)

    def test_gave_back_a_reachable_gain(self):
        trade = _trade(mfe=0.30)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.EXIT_TOO_SLOW)

    def test_adverse_selection_from_launch_bundles(self):
        trade = _trade(features=Features(bundle_pct=0.40))
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.ADVERSE_SELECTION)

    def test_honest_bucket_when_nothing_else_explains_it(self):
        self.assertIs(learning.classify(_trade(), STRATEGY), ErrorClass.NO_EDGE)

    def test_a_win_that_gave_back_its_peak_is_still_flagged(self):
        trade = _trade(pnl=10.0, exit_reason=ExitReason.TRAILING_STOP, mfe=2.5)
        self.assertIs(learning.classify(trade, STRATEGY), ErrorClass.EXIT_TOO_FAST)


class TestSelfTuning(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_nudges_are_bounded_by_the_config_value(self):
        base = float(STRATEGY["gates.min_liquidity_usd"])
        nudge = learning.NUDGES[ErrorClass.RUG][0]
        for _ in range(40):
            learning._apply_nudge(self.store, STRATEGY, nudge, "rug", 1.0)
        final = self.store.param(nudge.param, base)
        self.assertLessEqual(
            final,
            base * learning.MAX_DRIFT_MULTIPLE + 1e-6,
            "the bot must not be able to redesign the strategy behind your back",
        )

    def test_postmortem_ignores_a_class_that_is_not_costing_money(self):
        for _ in range(10):
            self.store.record_trade(
                _trade(pnl=+40.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, error_class=ErrorClass.RUG)
            )
        report = learning.run_postmortem(self.store, STRATEGY)
        self.assertTrue(any("not costing money" in note for note in report.skipped))
        self.assertEqual(self.store.all_params(), {})

    def test_postmortem_acts_on_a_dominant_expensive_class(self):
        for _ in range(12):
            self.store.record_trade(
                _trade(pnl=-30.0, exit_reason=ExitReason.LIQUIDITY_DRAIN, error_class=ErrorClass.RUG)
            )
        learning.run_postmortem(self.store, STRATEGY)
        self.assertGreater(
            self.store.param("gates.min_liquidity_usd", 0.0),
            float(STRATEGY["gates.min_liquidity_usd"]),
        )

    def test_training_refuses_to_learn_from_too_little_data(self):
        for _ in range(5):
            self.store.record_trade(_trade())
        result = learning.train(self.store, STRATEGY)
        self.assertFalse(result.trained)
        self.assertIn("keeping prior weights", result.note)

    def test_filter_cost_surfaces_expensive_gates(self):
        from alphahound.models import Action, Decision

        for i in range(15):
            decision = Decision(
                candidate=Candidate(chain=Chain.SOLANA, address=f"m{i}", price_usd=1.0),
                features=Features(),
                score=Score(probability=0.4, expected_value=0.0),
                action=Action.REJECT_GATE,
                reason="liquidity: 9000 < 15000",
            )
            decision_id = self.store.record_decision(decision)
            self.store.resolve_shadow(decision_id, 0.80)
        costs = learning.filter_cost(self.store)
        self.assertTrue(costs)
        self.assertEqual(costs[0].gate, "liquidity")
        self.assertEqual(costs[0].would_have_won, 15)

        notes = learning.relax_costly_gates(self.store, STRATEGY)
        self.assertTrue(notes, "a gate whose rejections all won must be loosened")
        self.assertLess(
            self.store.param("gates.min_liquidity_usd", 0.0),
            float(STRATEGY["gates.min_liquidity_usd"]),
        )


class TestBacktest(unittest.TestCase):
    def test_simulated_exit_never_credits_perfect_timing(self):
        peak = 3.0
        banked = backtest.simulate_exit(peak, STRATEGY)
        self.assertLess(banked, peak)
        self.assertGreater(banked, 0.5)

    def test_a_position_that_only_fell_takes_the_stop(self):
        self.assertAlmostEqual(
            backtest.simulate_exit(-0.90, STRATEGY),
            -float(STRATEGY["exits.stop_loss_pct"]),
            places=9,
        )


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store, self._tmp = make_store()

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_trade_round_trip_preserves_features(self):
        features = Features(retail_share=0.33, bundle_pct=0.11, axiom_share=0.22)
        self.store.record_trade(_trade(features=features))
        loaded = self.store.trades()[0]
        self.assertAlmostEqual(loaded.features.retail_share, 0.33, places=9)
        self.assertAlmostEqual(loaded.features.axiom_share, 0.22, places=9)

    def test_consecutive_losses(self):
        self.store.record_trade(_trade(pnl=-10.0))
        self.store.record_trade(_trade(pnl=-10.0))
        self.assertEqual(self.store.consecutive_losses(), 2)
        self.store.record_trade(_trade(pnl=+10.0))
        self.assertEqual(self.store.consecutive_losses(), 0)

    def test_param_changes_leave_an_audit_trail(self):
        self.store.set_param("exits.stop_loss_pct", 0.20, "because")
        self.store.set_param("exits.stop_loss_pct", 0.24, "changed mind")
        history = self.store.param_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["reason"], "changed mind")
        self.assertAlmostEqual(history[0]["old_value"], 0.20, places=9)

    def test_weights_versioning_and_activation(self):
        v1 = self.store.save_weights({"weights": {"a": 1.0}, "bias": 0.0}, 100, 0.60, activate=True)
        v2 = self.store.save_weights({"weights": {"a": 2.0}, "bias": 0.0}, 120, 0.55)
        self.assertEqual(self.store.active_weights()[0], v1)
        self.store.activate_weights(v2)
        self.assertEqual(self.store.active_weights()[0], v2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
