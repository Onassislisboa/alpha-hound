"""SQLite persistence.

One file, WAL mode, no ORM. The write volume of a bot that takes a handful of
positions an hour does not justify a database server, and a single .db file is
the difference between "I can inspect last Tuesday" and "the logs rotated".

The one non-obvious table is `shadow`: every REJECTED candidate is tracked for
an hour so the counterfactual PnL of our own filters is measurable. A bot that
only records trades it took can never discover that its gates are too tight -
it will just get quieter and quieter and call that discipline.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import (
    Action,
    Chain,
    Decision,
    ErrorClass,
    ExitReason,
    Features,
    TradeRecord,
    VenueId,
    now_ms,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    key TEXT NOT NULL,
    chain TEXT NOT NULL,
    symbol TEXT,
    action TEXT NOT NULL,
    probability REAL NOT NULL,
    expected_value REAL NOT NULL,
    size_usd REAL NOT NULL,
    signal_price REAL NOT NULL,
    features TEXT NOT NULL,
    contributions TEXT NOT NULL,
    weights_version INTEGER NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_key ON decisions(key);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts_ms);

CREATE TABLE IF NOT EXISTS shadow (
    decision_id INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL,
    price_at_decision REAL NOT NULL,
    best_price REAL NOT NULL,
    worst_price REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    counterfactual_pct REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    chain TEXT NOT NULL,
    venue TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL,
    closed_at_ms INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    signal_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    fees_usd REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    error_class TEXT NOT NULL,
    mfe REAL NOT NULL,
    mae REAL NOT NULL,
    entry_slippage REAL NOT NULL,
    features TEXT NOT NULL,
    weights_version INTEGER NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at_ms);

CREATE TABLE IF NOT EXISTS weights (
    version INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    payload TEXT NOT NULL,
    samples INTEGER NOT NULL,
    holdout_logloss REAL,
    active INTEGER NOT NULL DEFAULT 0,
    note TEXT
);

-- Parameters the postmortem loop is allowed to nudge, with an audit trail of
-- why. Nothing changes a live tunable without a row here explaining itself.
CREATE TABLE IF NOT EXISTS param_overrides (
    name TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS param_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    name TEXT NOT NULL,
    old_value REAL,
    new_value REAL NOT NULL,
    reason TEXT
);

-- Learned terminal attribution: address -> (label, class).
CREATE TABLE IF NOT EXISTS terminal_map (
    address TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    class TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

-- Wallets we independently measured, rather than wallets someone tweeted.
CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    pnl_usd REAL NOT NULL DEFAULT 0,
    is_smart INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def features_from_json(payload: str) -> Features:
    data = json.loads(payload)
    known = set(Features.names())
    return Features(**{k: float(v) for k, v in data.items() if k in known})


class Store:
    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "alphahound.db"
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- decisions ---------------------------------------------------------
    def record_decision(self, decision: Decision) -> int:
        cur = self.conn.execute(
            """INSERT INTO decisions (ts_ms, key, chain, symbol, action, probability,
                   expected_value, size_usd, signal_price, features, contributions,
                   weights_version, reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision.ts_ms,
                decision.candidate.key,
                decision.candidate.chain.value,
                decision.candidate.symbol,
                decision.action.value,
                decision.score.probability,
                decision.score.expected_value,
                decision.size_usd,
                decision.candidate.price_usd,
                json.dumps(decision.features.as_dict()),
                json.dumps(decision.score.contributions),
                decision.weights_version,
                decision.reason,
            ),
        )
        decision_id = int(cur.lastrowid or 0)
        if decision.action is not Action.ENTER and decision.candidate.price_usd > 0:
            self.open_shadow(decision_id, decision.candidate.key, decision.candidate.price_usd)
        return decision_id

    def recent_decision_keys(self, since_ms: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT key FROM decisions WHERE ts_ms >= ?", (since_ms,)
        ).fetchall()
        return {r["key"] for r in rows}

    # -- shadow tracking ---------------------------------------------------
    def open_shadow(self, decision_id: int, key: str, price: float) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO shadow
               (decision_id, key, opened_at_ms, price_at_decision, best_price, worst_price)
               VALUES (?,?,?,?,?,?)""",
            (decision_id, key, now_ms(), price, price, price),
        )

    def open_shadows(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM shadow WHERE resolved = 0").fetchall()

    def update_shadow(self, decision_id: int, price: float) -> None:
        self.conn.execute(
            """UPDATE shadow
               SET best_price = MAX(best_price, ?), worst_price = MIN(worst_price, ?)
               WHERE decision_id = ?""",
            (price, price, decision_id),
        )

    def resolve_shadow(self, decision_id: int, counterfactual_pct: float) -> None:
        self.conn.execute(
            "UPDATE shadow SET resolved = 1, counterfactual_pct = ? WHERE decision_id = ?",
            (counterfactual_pct, decision_id),
        )

    def filter_cost_report(self, limit: int = 200) -> list[sqlite3.Row]:
        """Rejected candidates that would have won, most profitable first. This
        is the report that tells you which gate is quietly costing you money."""
        return self.conn.execute(
            """SELECT d.action, d.reason, d.probability, d.expected_value,
                      s.counterfactual_pct, d.key, d.symbol, d.features
               FROM shadow s JOIN decisions d ON d.id = s.decision_id
               WHERE s.resolved = 1
               ORDER BY s.counterfactual_pct DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    # -- trades ------------------------------------------------------------
    def record_trade(self, trade: TradeRecord) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (key, chain, venue, opened_at_ms, closed_at_ms,
                   entry_price, exit_price, signal_price, size_usd, pnl_usd, fees_usd,
                   exit_reason, error_class, mfe, mae, entry_slippage, features,
                   weights_version, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade.key,
                trade.chain.value,
                trade.venue.value,
                trade.opened_at_ms,
                trade.closed_at_ms,
                trade.entry_price,
                trade.exit_price,
                trade.signal_price,
                trade.size_usd,
                trade.pnl_usd,
                trade.fees_usd,
                trade.exit_reason.value,
                trade.error_class.value,
                trade.max_favorable_excursion,
                trade.max_adverse_excursion,
                trade.entry_slippage,
                json.dumps(trade.features.as_dict()),
                trade.weights_version,
                trade.notes,
            ),
        )
        return int(cur.lastrowid or 0)

    def trades(self, limit: int | None = None, since_ms: int = 0) -> list[TradeRecord]:
        sql = "SELECT * FROM trades WHERE closed_at_ms >= ? ORDER BY closed_at_ms"
        params: list[Any] = [since_ms]
        if limit:
            sql += " DESC LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        out = [self._row_to_trade(r) for r in rows]
        return list(reversed(out)) if limit else out

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            key=row["key"],
            chain=Chain(row["chain"]),
            venue=VenueId(row["venue"]),
            opened_at_ms=row["opened_at_ms"],
            closed_at_ms=row["closed_at_ms"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            signal_price=row["signal_price"],
            size_usd=row["size_usd"],
            pnl_usd=row["pnl_usd"],
            fees_usd=row["fees_usd"],
            exit_reason=ExitReason(row["exit_reason"]),
            error_class=ErrorClass(row["error_class"]),
            max_favorable_excursion=row["mfe"],
            max_adverse_excursion=row["mae"],
            entry_slippage=row["entry_slippage"],
            features=features_from_json(row["features"]),
            weights_version=row["weights_version"],
            notes=row["notes"] or "",
        )

    def trade_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()
        return int(row["n"])

    def realized_pnl(self, since_ms: int = 0) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS pnl FROM trades WHERE closed_at_ms >= ?",
            (since_ms,),
        ).fetchone()
        return float(row["pnl"])

    def consecutive_losses(self) -> int:
        rows = self.conn.execute(
            "SELECT pnl_usd FROM trades ORDER BY closed_at_ms DESC LIMIT 20"
        ).fetchall()
        n = 0
        for r in rows:
            if r["pnl_usd"] >= 0:
                break
            n += 1
        return n

    def error_class_counts(self, since_ms: int = 0) -> dict[str, int]:
        rows = self.conn.execute(
            """SELECT error_class, COUNT(*) AS n FROM trades
               WHERE closed_at_ms >= ? GROUP BY error_class""",
            (since_ms,),
        ).fetchall()
        return {r["error_class"]: int(r["n"]) for r in rows}

    # -- weights -----------------------------------------------------------
    def save_weights(
        self,
        payload: dict[str, Any],
        samples: int,
        holdout_logloss: float | None,
        note: str = "",
        activate: bool = False,
    ) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM weights").fetchone()
        version = int(row["v"]) + 1
        self.conn.execute(
            """INSERT INTO weights (version, ts_ms, payload, samples, holdout_logloss, active, note)
               VALUES (?,?,?,?,?,?,?)""",
            (version, now_ms(), json.dumps(payload), samples, holdout_logloss, 0, note),
        )
        if activate:
            self.activate_weights(version)
        return version

    def activate_weights(self, version: int) -> None:
        self.conn.execute("UPDATE weights SET active = 0")
        self.conn.execute("UPDATE weights SET active = 1 WHERE version = ?", (version,))

    def active_weights(self) -> tuple[int, dict[str, Any]] | None:
        row = self.conn.execute("SELECT version, payload FROM weights WHERE active = 1").fetchone()
        if not row:
            return None
        return int(row["version"]), json.loads(row["payload"])

    def weights_version_row(self, version: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM weights WHERE version = ?", (version,)).fetchone()

    def pnl_by_weights_version(self, version: int) -> tuple[int, float]:
        row = self.conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usd), 0) AS pnl
               FROM trades WHERE weights_version = ?""",
            (version,),
        ).fetchone()
        return int(row["n"]), float(row["pnl"])

    # -- tunable parameters ------------------------------------------------
    def param(self, name: str, default: float) -> float:
        row = self.conn.execute(
            "SELECT value FROM param_overrides WHERE name = ?", (name,)
        ).fetchone()
        return float(row["value"]) if row else default

    def set_param(self, name: str, value: float, reason: str) -> None:
        old = self.conn.execute(
            "SELECT value FROM param_overrides WHERE name = ?", (name,)
        ).fetchone()
        self.conn.execute(
            """INSERT INTO param_overrides (name, value, updated_at_ms, reason)
               VALUES (?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET value=excluded.value,
                   updated_at_ms=excluded.updated_at_ms, reason=excluded.reason""",
            (name, value, now_ms(), reason),
        )
        self.conn.execute(
            """INSERT INTO param_history (ts_ms, name, old_value, new_value, reason)
               VALUES (?,?,?,?,?)""",
            (now_ms(), name, float(old["value"]) if old else None, value, reason),
        )

    def all_params(self) -> dict[str, float]:
        rows = self.conn.execute("SELECT name, value FROM param_overrides").fetchall()
        return {r["name"]: float(r["value"]) for r in rows}

    def param_history(self, limit: int = 50) -> list[sqlite3.Row]:
        # id breaks ties: two changes inside the same millisecond are common
        # during a learning cycle, and ordering by timestamp alone makes their
        # order undefined.
        return self.conn.execute(
            "SELECT * FROM param_history ORDER BY ts_ms DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- terminal attribution ---------------------------------------------
    def terminal_labels(self) -> dict[str, tuple[str, str]]:
        rows = self.conn.execute("SELECT address, label, class FROM terminal_map").fetchall()
        return {r["address"]: (r["label"], r["class"]) for r in rows}

    def label_terminal(self, address: str, label: str, klass: str) -> None:
        self.conn.execute(
            """INSERT INTO terminal_map (address, label, class, hits, updated_at_ms)
               VALUES (?,?,?,0,?)
               ON CONFLICT(address) DO UPDATE SET label=excluded.label,
                   class=excluded.class, updated_at_ms=excluded.updated_at_ms""",
            (address, label, klass, now_ms()),
        )

    # -- smart money -------------------------------------------------------
    def smart_wallets(self, chain: Chain) -> set[str]:
        rows = self.conn.execute(
            "SELECT address FROM wallets WHERE chain = ? AND is_smart = 1", (chain.value,)
        ).fetchall()
        return {r["address"] for r in rows}

    def upsert_wallet(
        self, address: str, chain: Chain, trades: int, wins: int, pnl_usd: float, is_smart: bool
    ) -> None:
        self.conn.execute(
            """INSERT INTO wallets (address, chain, trades, wins, pnl_usd, is_smart, updated_at_ms)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(address) DO UPDATE SET trades=excluded.trades,
                   wins=excluded.wins, pnl_usd=excluded.pnl_usd,
                   is_smart=excluded.is_smart, updated_at_ms=excluded.updated_at_ms""",
            (address, chain.value, trades, wins, pnl_usd, int(is_smart), now_ms()),
        )

    # -- kv ----------------------------------------------------------------
    def get_kv(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        self.conn.execute(
            """INSERT INTO kv (key, value) VALUES (?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, value),
        )

    def executemany(self, sql: str, rows: Iterable[tuple]) -> None:
        self.conn.executemany(sql, rows)
