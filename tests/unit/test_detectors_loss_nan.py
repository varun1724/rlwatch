"""Unit tests for the LossNaNDetector."""

from __future__ import annotations

import math

from rlwatch.config import LossNaNConfig
from rlwatch.detectors import LossNaNDetector


class TestLossNaNDetector:
    def test_finite_loss_no_alert(self):
        det = LossNaNDetector(LossNaNConfig())
        for step in range(100):
            assert det.check(step, loss=0.5) is None

    def test_nan_fires_critical(self):
        det = LossNaNDetector(LossNaNConfig())
        alert = det.check(10, loss=float("nan"))
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.detector == "loss_nan"
        assert alert.metric_values["kind"] == "NaN"

    def test_pos_inf_fires_critical(self):
        det = LossNaNDetector(LossNaNConfig())
        alert = det.check(5, loss=float("inf"))
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.metric_values["kind"] == "+Inf"

    def test_neg_inf_fires_critical(self):
        det = LossNaNDetector(LossNaNConfig())
        alert = det.check(7, loss=float("-inf"))
        assert alert is not None
        assert alert.metric_values["kind"] == "-Inf"

    def test_none_input_noop(self):
        det = LossNaNDetector(LossNaNConfig())
        assert det.check(0, loss=None) is None

    def test_disabled_noop(self):
        det = LossNaNDetector(LossNaNConfig(enabled=False))
        assert det.check(0, loss=float("nan")) is None

    def test_warmup_suppresses(self):
        det = LossNaNDetector(LossNaNConfig(warmup_steps=10))
        # First 9 steps (1..9) are inside warmup and should not alert.
        for step in range(9):
            assert det.check(step, loss=float("nan")) is None
        # 10th call passes the warmup gate.
        alert = det.check(9, loss=float("nan"))
        assert alert is not None

    def test_fires_every_step_no_internal_cooldown(self):
        """Cooldown is the AlertManager's job, not the detector's."""
        det = LossNaNDetector(LossNaNConfig())
        alerts = [det.check(i, loss=float("nan")) for i in range(5)]
        assert all(a is not None for a in alerts)

    def test_recommendation_present(self):
        det = LossNaNDetector(LossNaNConfig())
        alert = det.check(0, loss=float("nan"))
        assert alert is not None
        assert alert.recommendation  # cardinal rule #3
