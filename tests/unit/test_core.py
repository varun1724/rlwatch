"""Tests for rlwatch core API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch, attach


class TestRLWatch:
    def test_creates_run_id(self):
        config = RLWatchConfig()
        config.storage.log_dir = tempfile.mkdtemp()
        monitor = RLWatch(config)
        assert monitor.run_id != ""
        monitor.stop()

    def test_uses_provided_run_id(self):
        config = RLWatchConfig()
        config.run_id = "test_run_123"
        config.storage.log_dir = tempfile.mkdtemp()
        monitor = RLWatch(config)
        assert monitor.run_id == "test_run_123"
        monitor.stop()

    def test_log_step_stores_metrics(self):
        config = RLWatchConfig()
        config.storage.log_dir = tempfile.mkdtemp()
        monitor = RLWatch(config)
        monitor.start()

        monitor.log_step(0, entropy=2.5, kl_divergence=0.01)
        monitor.log_step(1, entropy=2.4, kl_divergence=0.02)

        metrics = monitor.store.get_metrics()
        assert len(metrics) == 2
        assert metrics[0]["entropy"] == 2.5
        assert metrics[1]["entropy"] == 2.4
        monitor.stop()

    def test_log_step_computes_reward_stats(self):
        config = RLWatchConfig()
        config.storage.log_dir = tempfile.mkdtemp()
        monitor = RLWatch(config)
        monitor.start()

        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        monitor.log_step(0, rewards=rewards)

        metrics = monitor.store.get_metrics()
        assert len(metrics) == 1
        assert abs(metrics[0]["reward_mean"] - 2.5) < 0.01
        monitor.stop()

    def test_detects_and_stores_alerts(self):
        config = RLWatchConfig()
        config.storage.log_dir = tempfile.mkdtemp()
        config.entropy_collapse.warmup_steps = 2
        config.entropy_collapse.consecutive_steps = 3
        monitor = RLWatch(config)
        monitor.start()

        # Warmup
        for step in range(5):
            monitor.log_step(step, entropy=2.5)

        # Collapse
        for step in range(5, 15):
            monitor.log_step(step, entropy=0.1)

        alerts = monitor.store.get_alerts()
        assert len(alerts) > 0
        assert any(a["detector"] == "entropy_collapse" for a in alerts)
        monitor.stop()


class TestAttach:
    def test_attach_returns_monitor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", str(tmp_path))
        monitor = attach(
            framework="manual",
            run_id="test_attach",
        )
        assert monitor is not None
        assert monitor.run_id == "test_attach"
        monitor.stop()
