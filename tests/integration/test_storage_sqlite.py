"""Integration tests for the SQLite metric store on a real tmp_path DB."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.storage import SCHEMA_VERSION, MetricStore

pytestmark = pytest.mark.integration


def _cfg(log_dir: str, run_id: str = "int_test") -> RLWatchConfig:
    cfg = RLWatchConfig()
    cfg.storage.log_dir = log_dir
    cfg.run_id = run_id
    return cfg


class TestRealSQLite:
    def test_wal_mode_is_on(self, tmp_log_dir):
        store = MetricStore(_cfg(tmp_log_dir))
        try:
            row = store._conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal"
        finally:
            store.close()

    def test_bulk_write_2000_rows(self, tmp_log_dir):
        store = MetricStore(_cfg(tmp_log_dir))
        try:
            store.register_run(_cfg(tmp_log_dir))
            for step in range(2000):
                store.log_metrics(
                    step,
                    entropy=2.5,
                    kl_divergence=0.01,
                    grad_norm=1.0,
                )
            metrics = store.get_metrics()
            assert len(metrics) == 2000
            # grad_norm column round-trips.
            assert metrics[0]["grad_norm"] == 1.0
        finally:
            store.close()

    def test_close_reopen_persists_data(self, tmp_log_dir):
        cfg = _cfg(tmp_log_dir, run_id="persist")
        store1 = MetricStore(cfg)
        store1.register_run(cfg)
        store1.log_metrics(0, entropy=1.5, grad_norm=0.5)
        store1.close()

        store2 = MetricStore(cfg)
        try:
            metrics = store2.get_metrics(run_id="persist")
            assert len(metrics) == 1
            assert metrics[0]["entropy"] == 1.5
            assert metrics[0]["grad_norm"] == 0.5
            # Schema version still current after reopen.
            v = store2._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            assert v[0] == SCHEMA_VERSION
        finally:
            store2.close()

    def test_alerts_table_round_trip(self, tmp_log_dir):
        store = MetricStore(_cfg(tmp_log_dir))
        try:
            store.register_run(_cfg(tmp_log_dir))
            store.log_alert(
                step=10,
                detector="loss_nan",
                severity="critical",
                message="hi",
                metric_values={"loss": None},
                recommendation="rollback",
            )
            alerts = store.get_alerts()
            assert len(alerts) == 1
            assert alerts[0]["detector"] == "loss_nan"
            assert alerts[0]["severity"] == "critical"
        finally:
            store.close()
