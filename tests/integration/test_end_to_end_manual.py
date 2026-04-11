"""End-to-end integration test for the manual ``log_step`` path.

attach() → log_step loop → SQLite assertions → CLI diagnose round-trip.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import rlwatch
from rlwatch.cli import main

pytestmark = pytest.mark.integration


def test_manual_attach_full_pipeline(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
    monitor = rlwatch.attach(framework="manual", run_id="e2e_run")
    try:
        # Tighten thresholds so we get alerts inside 100 steps.
        monitor._detectors.entropy_detector.config.warmup_steps = 2
        monitor._detectors.entropy_detector.config.consecutive_steps = 5

        for step in range(5):
            monitor.log_step(step, entropy=2.5, kl_divergence=0.01, loss=0.5)
        for step in range(5, 100):
            monitor.log_step(
                step,
                entropy=0.05,
                kl_divergence=0.01,
                loss=0.4,
                grad_norm=1.0,
            )

        # SQLite contains all rows + at least one entropy alert.
        metrics = monitor.store.get_metrics()
        assert len(metrics) == 100
        alerts = monitor.store.get_alerts()
        assert any(a["detector"] == "entropy_collapse" for a in alerts)
    finally:
        monitor.stop()

    # CLI diagnose round-trips the same DB.
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "diagnose",
            "--log-dir",
            tmp_log_dir,
            "--run-id",
            "e2e_run",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["run_id"] == "e2e_run"
    assert data["total_steps"] == 100
    assert data["health"] == "critical"
    assert "entropy_collapse" in data["detectors_triggered"]
