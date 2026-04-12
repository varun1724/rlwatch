"""SQLite-based metric persistence for rlwatch."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from rlwatch.config import RLWatchConfig

# Current on-disk schema version. Bump when the DDL below changes, and add a
# migration step in ``_migrate`` keyed on the previous version. CLAUDE.md
# cardinal rule #6: never break a user's ``.db`` without a migration.
#
# Version history
#   1: initial schema (runs, metrics, alerts)
#   2: adds metrics.grad_norm REAL for the gradient-norm spike detector
SCHEMA_VERSION = 2


class MetricStore:
    """Persists training metrics to a local SQLite database."""

    def __init__(self, config: RLWatchConfig):
        self.log_dir = Path(config.storage.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.log_dir / config.storage.db_name
        self.run_id = config.run_id
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                framework TEXT,
                started_at REAL,
                config_json TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                entropy REAL,
                kl_divergence REAL,
                reward_mean REAL,
                reward_std REAL,
                reward_min REAL,
                reward_max REAL,
                advantage_std REAL,
                loss REAL,
                learning_rate REAL,
                clip_fraction REAL,
                grad_norm REAL,
                extra_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_step
                ON metrics(run_id, step);

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                detector TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_values_json TEXT,
                recommendation TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_run
                ON alerts(run_id);

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
        """)
        self._conn.commit()
        self._migrate()

    def _migrate(self):
        """Apply any pending schema migrations.

        The current version is read from the ``schema_version`` table. For a
        brand-new database the table is empty, so we stamp it with
        ``SCHEMA_VERSION``. Future bumps add ``if current < N`` branches that
        run idempotent ``ALTER TABLE`` statements.
        """
        row = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        current = row[0] if row else 0

        if current == SCHEMA_VERSION:
            return

        if current == 0:
            # Fresh database — DDL above already created v2 layout. Stamp it.
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()
            return

        if current < 2:
            # v1 → v2: add grad_norm column. ``ADD COLUMN`` is idempotent
            # against repeat opens via the version check above.
            self._conn.execute(
                "ALTER TABLE metrics ADD COLUMN grad_norm REAL"
            )
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (2,)
            )
            self._conn.commit()
            current = 2

        if current == SCHEMA_VERSION:
            return

        raise RuntimeError(
            f"rlwatch database is at schema version {current}, but this "
            f"version of rlwatch expects {SCHEMA_VERSION}. Upgrade rlwatch "
            f"or migrate the database."
        )

    def register_run(self, config: RLWatchConfig):
        """Register a new training run."""
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, framework, started_at, config_json) VALUES (?, ?, ?, ?)",
            (config.run_id, config.framework, time.time(), json.dumps(config.to_dict())),
        )
        self._conn.commit()

    def log_metrics(
        self,
        step: int,
        *,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        advantage_std: Optional[float] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        grad_norm: Optional[float] = None,
        **extra,
    ):
        """Log metrics for a single training step."""
        extra_json = json.dumps(extra) if extra else None
        self._conn.execute(
            """INSERT INTO metrics
               (run_id, step, timestamp, entropy, kl_divergence,
                reward_mean, reward_std, reward_min, reward_max,
                advantage_std, loss, learning_rate, clip_fraction,
                grad_norm, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id, step, time.time(),
                entropy, kl_divergence,
                reward_mean, reward_std, reward_min, reward_max,
                advantage_std, loss, learning_rate, clip_fraction,
                grad_norm, extra_json,
            ),
        )
        # Commit every step for durability (WAL mode keeps this fast)
        self._conn.commit()

    def log_alert(
        self,
        step: int,
        detector: str,
        severity: str,
        message: str,
        metric_values: dict,
        recommendation: str,
    ):
        """Log an alert to the database."""
        self._conn.execute(
            """INSERT INTO alerts
               (run_id, step, timestamp, detector, severity, message, metric_values_json, recommendation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id, step, time.time(),
                detector, severity, message,
                json.dumps(metric_values), recommendation,
            ),
        )
        self._conn.commit()

    def get_metrics(self, run_id: Optional[str] = None) -> list[dict]:
        """Retrieve all metrics for a run.

        Returns an empty list if the store has been closed.
        """
        if self._conn is None:
            return []
        rid = run_id or self.run_id
        cursor = self._conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY step", (rid,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_alerts(self, run_id: Optional[str] = None) -> list[dict]:
        """Retrieve all alerts for a run.

        Returns an empty list if the store has been closed.
        """
        if self._conn is None:
            return []
        rid = run_id or self.run_id
        cursor = self._conn.execute(
            "SELECT * FROM alerts WHERE run_id = ? ORDER BY step", (rid,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_all_runs(self) -> list[dict]:
        """Retrieve all runs.

        Returns an empty list if the store has been closed.
        """
        if self._conn is None:
            return []
        cursor = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_latest_metrics(self, run_id: Optional[str] = None, limit: int = 1000) -> list[dict]:
        """Retrieve latest N metrics for a run.

        Returns an empty list if the store has been closed.
        """
        if self._conn is None:
            return []
        rid = run_id or self.run_id
        cursor = self._conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY step DESC LIMIT ?",
            (rid, limit),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        rows.reverse()
        return rows

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def open_store(log_dir: str, db_name: str = "metrics.db") -> sqlite3.Connection:
    """Open an existing metric store for read-only retrospective analysis."""
    db_path = Path(log_dir) / db_name
    if not db_path.exists():
        raise FileNotFoundError(f"No metric database found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn
