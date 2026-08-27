"""The engine.

One loop, in a fixed order that encodes the priorities:

    1. manage open positions   money already at risk outranks money not yet at
                               risk, always
    2. update shadow tracking  measure what the filters cost
    3. discover                pull new candidates
    4. score and enter         spend the RPC budget on the best candidates only
    5. learn                   periodically, never inside the hot path

Open positions are persisted to state/positions.json on every change. A bot
that forgets its positions on restart is a bot that leaves bags on chain, and
restarts happen at exactly the moments you least want that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import asdict
from pathlib import Path

from . import learning
from .discovery import Discovery
from .execution import ExecutionError, Router, build_router
from .execution.relay import IntentRelay
from .log import get
from .models import (
    Action,
    Candidate,
    Chain,
    Decision,
    ErrorClass,
    ExitReason,
    Features,
    Position,
    TradeRecord,
    VenueId,
    now_ms,
)
from .net import Http
from .portfolio import ExitOrder, PositionManager
from .providers import Birdeye, Dexscreener, Helius
from .risk import RiskEngine
from .scoring import Model, Scorer
from .settings import Config, Settings, load_strategy, load_terminals
from .signals import Enricher
from .signals.solana import SolanaReader
from .signals.terminals import TerminalRegistry
from .store import Store, features_from_json

log = get("engine")


def build_registry(store: Store, terminals: Config) -> TerminalRegistry:
    entries = terminals.get("terminal", []) or []
    venues = terminals.section("venues")
    return TerminalRegistry(entries, venues, store.terminal_labels())


class Engine:
    def __init__(
        self,
        settings: Settings,
        strategy: Config | None = None,
        terminals: Config | None = None,
    ) -> None:
        self.settings = settings
        self.strategy = strategy or load_strategy()
        self.terminals = terminals or load_terminals()

        self.store = Store(settings.state_dir)
        self.http = Http()
        self.dex = Dexscreener(
            self.http,
            cache_seconds=0.8 * float(self.strategy.get("loop.tick_seconds", 3.0)),
        )
        self.helius = Helius(self.http, settings.helius_api_key)
        self.birdeye = Birdeye(self.http, settings.birdeye_api_key)
        self.solana = (
            SolanaReader(self.http, settings.solana_rpc_url)
            if settings.solana_rpc_url and Chain.SOLANA in settings.enabled_chains
            else None
        )

        self.router: Router = build_router(settings, self.strategy, self.dex, self.http)
        self.registry = build_registry(self.store, self.terminals)
        self.enricher = Enricher(
            store=self.store,
            strategy=self.strategy,
            dexscreener=self.dex,
            registry=self.registry,
            solana=self.solana,
            helius=self.helius,
            birdeye=self.birdeye,
            probe=self.router.round_trip,
        )
        self.scorer = Scorer(Model.load(self.store), self.strategy, self.store, live=settings.live)
        self.risk = RiskEngine(self.strategy, self.store)
        self.exits = PositionManager(self.strategy, self.store)
        self.relay = IntentRelay(self.http, settings)
        self.discovery = Discovery(settings, self.strategy, self.dex)

        self.positions: dict[str, Position] = {}
        self.watching: dict[str, Candidate] = {}
        self._positions_path = settings.state_dir / "positions.json"
        self._closed_since_learn = 0
        self._stop = asyncio.Event()
        self._load_positions()

    # -- lifecycle ---------------------------------------------------------
    async def run(self) -> None:
        problems = self.settings.validate()
        if problems:
            for problem in problems:
                log.error("configuration problem", extra={"problem": problem})
            raise SystemExit("refusing to start with an invalid live configuration")

        log.info(
            "starting",
            extra={
                "mode": self.settings.mode,
                "chains": [c.value for c in self.settings.enabled_chains],
                "weights_version": self.scorer.model.version,
                "equity_usd": self.risk.equity(),
                "open_positions": len(self.positions),
            },
        )
        if not self.registry.attributable_labels:
            log.warning(
                "no terminal fee accounts labeled: attribution features are inert. "
                "Run `alphahound discover-terminals` to populate them."
            )

        await self.discovery.start()
        tick_seconds = float(self.strategy.get("loop.tick_seconds", 3.0))
        try:
            while not self._stop.is_set():
                started = now_ms()
                try:
                    await self.tick()
                except Exception:  # noqa: BLE001 - a bad tick must not kill the bot
                    log.exception("tick failed")
                elapsed = (now_ms() - started) / 1000.0
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=max(0.1, tick_seconds - elapsed)
                    )
        finally:
            await self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        await self.discovery.stop()
        await self.http.aclose()
        self._save_positions()
        self.store.close()
        log.info("stopped")

    # -- the loop ----------------------------------------------------------
    async def tick(self) -> None:
        await self.manage_positions()
        await self.update_shadows()

        found = await self.discovery.poll()
        for candidate in found:
            self.watching[candidate.key] = candidate
        self.discovery.prune()
        self._prune_watching()

        await self.score_and_enter()
        await self.maybe_learn()

    # -- positions ---------------------------------------------------------
    async def manage_positions(self) -> None:
        if not self.positions:
            return
        halted, reason = self.risk.halted()

        for key, position in list(self.positions.items()):
            await self.enricher.refresh(position.candidate)
            price = position.candidate.price_usd
            if price <= 0:
                log.warning("no price for open position", extra={"key": key})
                continue

            orders = self.exits.evaluate(position, price, position.candidate.liquidity_usd)
            if halted and not orders:
                # The kill switch closes positions rather than merely stopping
                # new ones. Halting entries while holding open risk is the
                # worst of both states.
                orders = [ExitOrder(1.0, ExitReason.KILL_SWITCH, reason)]
            for order in orders:
                await self._exit(position, order.fraction, order.reason, order.note)
                if position.tokens_remaining <= 1e-12:
                    break

    async def _exit(self, position: Position, fraction: float, reason: ExitReason, note: str) -> None:
        tokens = position.tokens_remaining * max(0.0, min(1.0, fraction))
        if tokens <= 0:
            return
        try:
            fill = await self.router.sell(position, tokens)
        except ExecutionError as exc:
            log.error(
                "exit failed",
                extra={"key": position.candidate.key, "reason": reason.value, "error": str(exc)},
            )
            return

        cost_basis = position.size_usd * (tokens / position.tokens) if position.tokens else 0.0
        position.realized_usd += fill.amount_out - cost_basis
        position.fees_usd += fill.fee_usd
        position.tokens_remaining = max(0.0, position.tokens_remaining - tokens)
        position.last_exit_price = fill.price
        position.last_exit_reason = reason.value

        log.info(
            "exit",
            extra={
                "symbol": position.candidate.symbol or position.candidate.address,
                "reason": reason.value,
                "note": note,
                "tokens": tokens,
                "usd_out": round(fill.amount_out, 2),
                "realized_usd": round(position.realized_usd, 2),
                "remaining": position.tokens_remaining,
            },
        )

        if position.tokens_remaining <= 1e-12:
            self._close(position, reason)
        self._save_positions()

    def _close(self, position: Position, reason: ExitReason) -> None:
        mfe, mae = PositionManager.excursions(position)
        trade = TradeRecord(
            key=position.candidate.key,
            chain=position.candidate.chain,
            venue=position.venue,
            opened_at_ms=position.opened_at_ms,
            closed_at_ms=now_ms(),
            entry_price=position.entry_price,
            exit_price=position.last_exit_price or position.candidate.price_usd,
            signal_price=position.signal_price,
            size_usd=position.size_usd,
            pnl_usd=position.realized_usd,
            fees_usd=position.fees_usd,
            exit_reason=reason,
            error_class=ErrorClass.WIN,
            features=position.entry_features,
            weights_version=self.scorer.model.version,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            entry_slippage=0.0,
        )
        trade.error_class = learning.classify(trade, self.strategy)
        self.store.record_trade(trade)
        self.positions.pop(position.candidate.key, None)
        self.enricher.forget(position.candidate.key)
        self.risk.note_trade_closed(trade.won)
        self._closed_since_learn += 1
        log.info(
            "closed",
            extra={
                "symbol": position.candidate.symbol or position.candidate.address,
                "pnl_usd": round(trade.pnl_usd, 2),
                "pnl_pct": round(trade.pnl_pct, 4),
                "error_class": trade.error_class.value,
                "exit_reason": reason.value,
                "mfe": round(mfe, 3),
            },
        )

    # -- shadow tracking ---------------------------------------------------
    async def update_shadows(self) -> None:
        """Track rejected candidates so the cost of our own filters is a number
        rather than an opinion."""
        rows = self.store.open_shadows()
        if not rows:
            return
        horizon_ms = int(float(self.strategy.get("learning.shadow_track_minutes", 60)) * 60_000)
        stale = [r for r in rows if now_ms() - r["opened_at_ms"] >= horizon_ms]
        stale_ids = {r["decision_id"] for r in stale}
        live_rows = [r for r in rows if r["decision_id"] not in stale_ids][:30]

        for row in stale:
            entry = float(row["price_at_decision"]) or 0.0
            best = float(row["best_price"] or entry)
            self.store.resolve_shadow(
                row["decision_id"], (best / entry - 1.0) if entry > 0 else 0.0
            )

        if not live_rows:
            return
        addresses = []
        by_address: dict[str, list[int]] = {}
        for row in live_rows:
            _, _, address = row["key"].partition(":")
            if not address:
                continue
            by_address.setdefault(address, []).append(row["decision_id"])
            addresses.append(address)
        for start in range(0, len(addresses), 30):
            snaps = await self.dex.token_pairs(addresses[start : start + 30])
            for snap in snaps:
                for decision_id in by_address.get(snap.token_address, []):
                    self.store.update_shadow(decision_id, snap.price_usd)

    # -- scoring and entry -------------------------------------------------
    async def score_and_enter(self) -> None:
        if not self.watching:
            return
        halted, reason = self.risk.halted()
        if halted:
            log.debug("not entering", extra={"reason": reason})
            return

        limit = int(self.strategy.get("loop.max_candidates_per_tick", 40))
        # Cheapest possible pre-rank: liquidity times recent volume, both free
        # from the Dexscreener snapshot. Enrichment costs RPC calls, so it is
        # spent on the top of this list rather than on everything.
        ranked = sorted(
            self.watching.values(),
            key=lambda c: -(c.volume_5m_usd * min(c.liquidity_usd, 250_000.0)),
        )[:limit]

        probe_size = max(
            float(self.strategy.get("risk.min_position_usd", 15.0)),
            self.risk.equity() * float(self.strategy.get("risk.max_position_pct", 0.05)),
        )

        for candidate in ranked:
            if candidate.key in self.positions:
                continue
            if not self.router.has_venue(candidate.chain):
                continue
            try:
                enrichment = await self.enricher.enrich(candidate, probe_size)
            except Exception as exc:  # noqa: BLE001
                log.debug("enrich failed", extra={"key": candidate.key, "error": str(exc)})
                continue

            score = self.scorer.score(enrichment)
            ok, why = self.scorer.passes(score)
            if not ok:
                action = Action.REJECT_GATE if score.vetoed else Action.REJECT_SCORE
                self._record(candidate, enrichment.features, score, action, 0.0, why)
                continue

            sizing = self.risk.size(candidate, score, self.scorer.payoff, list(self.positions.values()))
            if not sizing.allowed:
                self._record(
                    candidate, enrichment.features, score, Action.REJECT_RISK, 0.0, sizing.reason
                )
                continue

            decision = Decision(
                candidate=candidate,
                features=enrichment.features,
                score=score,
                action=Action.ENTER,
                size_usd=sizing.size_usd,
                reason=sizing.reason,
                weights_version=self.scorer.model.version,
            )
            decision_id = self.store.record_decision(decision)
            await self.relay.publish(decision, sizing.size_usd, note=self.scorer.explain(score))
            await self._enter(decision, decision_id)

    def _record(
        self,
        candidate: Candidate,
        features: Features,
        score,
        action: Action,
        size_usd: float,
        reason: str,
    ) -> None:
        self.store.record_decision(
            Decision(
                candidate=candidate,
                features=features,
                score=score,
                action=action,
                size_usd=size_usd,
                reason=reason,
                weights_version=self.scorer.model.version,
            )
        )

    async def _enter(self, decision: Decision, decision_id: int) -> None:
        candidate = decision.candidate
        signal_price = candidate.price_usd
        try:
            fill = await self.router.buy(candidate, decision.size_usd)
        except ExecutionError as exc:
            # A failed submit still costs gas and, more importantly, is evidence
            # about execution quality. Recording it as a zero-size loss keeps it
            # in the error taxonomy instead of vanishing into the log.
            log.warning("entry failed", extra={"key": candidate.key, "error": str(exc)})
            self.store.record_trade(
                TradeRecord(
                    key=candidate.key,
                    chain=candidate.chain,
                    venue=VenueId.PAPER,
                    opened_at_ms=now_ms(),
                    closed_at_ms=now_ms(),
                    entry_price=signal_price,
                    exit_price=signal_price,
                    signal_price=signal_price,
                    size_usd=decision.size_usd,
                    pnl_usd=0.0,
                    fees_usd=0.0,
                    exit_reason=ExitReason.MANUAL,
                    error_class=ErrorClass.EXECUTION_FAIL,
                    features=decision.features,
                    weights_version=decision.weights_version,
                    notes=str(exc)[:400],
                )
            )
            return

        if fill.amount_out <= 0:
            log.error("entry filled zero tokens", extra={"key": candidate.key})
            return

        position = Position(
            candidate=candidate,
            venue=fill.venue,
            entry_price=fill.price,
            size_usd=decision.size_usd,
            tokens=fill.amount_out,
            tokens_remaining=fill.amount_out,
            fees_usd=fill.fee_usd,
            decision_id=decision_id,
            entry_features=decision.features,
            signal_price=signal_price,
        )
        self.positions[candidate.key] = position
        self._save_positions()
        log.info(
            "entered",
            extra={
                "symbol": candidate.symbol or candidate.address,
                "chain": candidate.chain.value,
                "venue": fill.venue.value,
                "size_usd": decision.size_usd,
                "entry_price": fill.price,
                "drift_from_signal": round(fill.price / signal_price - 1.0, 4)
                if signal_price
                else 0.0,
                "score": self.scorer.explain(decision.score),
                "sizing": decision.reason,
                "tx": fill.tx_id,
            },
        )

    # -- learning ----------------------------------------------------------
    async def maybe_learn(self) -> None:
        if not self.strategy.get("learning.enabled", True):
            return
        cadence = int(self.strategy.get("learning.retrain_every_closed_trades", 10))
        if self._closed_since_learn < cadence:
            return
        self._closed_since_learn = 0

        rolled_back = learning.check_rollback(self.store, self.strategy)
        report = learning.run_postmortem(self.store, self.strategy)
        relaxed = learning.relax_costly_gates(self.store, self.strategy)
        result = learning.train(self.store, self.strategy)

        if result.promoted or rolled_back:
            self.scorer.model = Model.load(self.store)
        log.info(
            "learning cycle",
            extra={
                "postmortem": report.counts,
                "applied": report.applied,
                "relaxed": relaxed,
                "training": result.note,
                "rollback": rolled_back,
                "active_weights": self.scorer.model.version,
            },
        )

    # -- housekeeping ------------------------------------------------------
    def _prune_watching(self) -> None:
        max_age = float(self.strategy.get("loop.max_candidate_age_minutes", 180))
        for key, candidate in list(self.watching.items()):
            if key in self.positions:
                continue
            age = candidate.age_minutes or (now_ms() - candidate.discovered_at_ms) / 60_000.0
            if age > max_age:
                del self.watching[key]
                self.enricher.forget(key)

    def _save_positions(self) -> None:
        payload = []
        for position in self.positions.values():
            data = asdict(position)
            data["candidate"]["chain"] = position.candidate.chain.value
            data["venue"] = position.venue.value
            data["entry_features"] = position.entry_features.as_dict()
            payload.append(data)
        tmp = self._positions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._positions_path)

    def _load_positions(self) -> None:
        if not self._positions_path.exists():
            return
        try:
            payload = json.loads(self._positions_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            log.error("could not read persisted positions", extra={"error": str(exc)})
            return
        for data in payload:
            try:
                candidate_data = dict(data.pop("candidate"))
                candidate_data["chain"] = Chain(candidate_data["chain"])
                candidate = Candidate(
                    **{
                        k: v
                        for k, v in candidate_data.items()
                        if k in Candidate.__dataclass_fields__
                    }
                )
                features = features_from_json(json.dumps(data.pop("entry_features", {})))
                position = Position(
                    candidate=candidate,
                    venue=VenueId(data.pop("venue")),
                    entry_features=features,
                    **{
                        k: v
                        for k, v in data.items()
                        if k in Position.__dataclass_fields__
                        and k not in {"candidate", "venue", "entry_features"}
                    },
                )
                self.positions[candidate.key] = position
                self.watching[candidate.key] = candidate
            except (KeyError, TypeError, ValueError) as exc:
                log.error("skipping unreadable position", extra={"error": str(exc)})
        if self.positions:
            log.info("restored positions", extra={"count": len(self.positions)})


async def run_forever(settings: Settings) -> None:
    engine = Engine(settings)
    await engine.run()
