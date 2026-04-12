"""Unit tests for the SQLite schema migration path."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.storage import SCHEMA_VERSION, MetricStore, open_store


def _make_config(tmp_log_dir: str) -> RLWatchConfig:
    cfg = RLWatchConfig()
    cfg.storage.log_dir = tmp_log_dir
    cfg.run_id = "test_run"
    return cfg


class TestSchemaMigration:
    def test_fresh_db_stamps_current_version(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        try:
            row = store._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            assert row[0] == SCHEMA_VERSION
        finally:
            store.close()

    def test_grad_norm_column_exists_after_init(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        try:
            cols = [
                row[1]
                for row in store._conn.execute("PRAGMA table_info(metrics)").fetchall()
            ]
            assert "grad_norm" in cols
        finally:
            store.close()

    def test_v1_db_migrates_to_v2(self, tmp_log_dir):
        """Hand-build a v1 schema, then open via MetricStore and assert upgrade."""
        db_path = Path(tmp_log_dir) / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        # Recreate the v1 layout: schema_version table exists with version 1,
        # metrics table has every column EXCEPT grad_norm.
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                framework TEXT,
                started_at REAL,
                config_json TEXT
            );
            CREATE TABLE metrics (
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
                extra_json TEXT
            );
            CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                detector TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_values_json TEXT,
                recommendation TEXT
            );
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version (version) VALUES (1);
            """
        )
        conn.commit()
        conn.close()

        # Open via MetricStore — should migrate.
        store = MetricStore(_make_config(tmp_log_dir))
        try:
            cols = [
                row[1]
                for row in store._conn.execute("PRAGMA table_info(metrics)").fetchall()
            ]
            assert "grad_norm" in cols
            row = store._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            assert row[0] == 2
        finally:
            store.close()

    def test_second_open_is_idempotent(self, tmp_log_dir):
        store1 = MetricStore(_make_config(tmp_log_dir))
        store1.close()
        store2 = MetricStore(_make_config(tmp_log_dir))
        try:
            row = store2._conn.execute(
                "SELECT COUNT(*) FROM schema_version"
            ).fetchone()
            # Fresh init stamps once, second open does nothing.
            assert row[0] == 1
        finally:
            store2.close()

    def test_future_version_raises(self, tmp_log_dir):
        # Build a full v2-shape DB and then bump the schema_version row to a
        # value newer than this rlwatch knows about. The v2 DDL is needed so
        # the ``CREATE INDEX IF NOT EXISTS`` statements MetricStore re-runs on
        # open don't fail on missing columns.
        db_path = Path(tmp_log_dir) / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                framework TEXT,
                started_at REAL,
                config_json TEXT
            );
            CREATE TABLE metrics (
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
                extra_json TEXT
            );
            CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                detector TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_values_json TEXT,
                recommendation TEXT
            );
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version (version) VALUES (999);
            """
        )
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="schema version 999"):
            MetricStore(_make_config(tmp_log_dir))


class TestStoreAfterClose:
    def test_get_alerts_returns_empty_after_close(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        store.close()
        assert store.get_alerts() == []

    def test_get_metrics_returns_empty_after_close(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        store.close()
        assert store.get_metrics() == []

    def test_get_all_runs_returns_empty_after_close(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        store.close()
        assert store.get_all_runs() == []

    def test_get_latest_metrics_returns_empty_after_close(self, tmp_log_dir):
        store = MetricStore(_make_config(tmp_log_dir))
        store.close()
        assert store.get_latest_metrics() == []


class TestOpenStore:
    def test_open_existing_store(self, tmp_log_dir):
        # Initialize so the .db file exists.
        store = MetricStore(_make_config(tmp_log_dir))
        store.close()

        conn = open_store(tmp_log_dir)
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == SCHEMA_VERSION
        finally:
            conn.close()

    def test_open_missing_store_raises(self, tmp_log_dir):
        with pytest.raises(FileNotFoundError):
            open_store(tmp_log_dir, db_name="not_here.db")
