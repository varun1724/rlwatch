"""Unit tests for the GradientNormSpikeDetector."""

from __future__ import annotations

import numpy as np

from rlwatch.config import GradientNormSpikeConfig
from rlwatch.detectors import GradientNormSpikeDetector


class TestGradientNormSpikeDetector:
    def test_no_alert_for_stable_grad_norm(self):
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(warmup_steps=5, sigma_multiplier=3.0)
        )
        for step in range(150):
            assert det.check(step, grad_norm=1.0 + 0.001 * (step % 3)) is None

    def test_alerts_on_spike(self):
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(
                warmup_steps=5, sigma_multiplier=2.0, rolling_window=50
            )
        )
        rng = np.random.default_rng(0)
        for step in range(60):
            det.check(step, grad_norm=1.0 + float(rng.normal(0, 0.05)))
        alert = det.check(61, grad_norm=10.0)
        assert alert is not None
        assert alert.detector == "gradient_norm_spike"
        assert alert.severity in {"warning", "critical"}

    def test_warmup_suppresses(self):
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(warmup_steps=10)
        )
        for step in range(10):
            assert det.check(step, grad_norm=100.0) is None

    def test_none_input_noop(self):
        det = GradientNormSpikeDetector(GradientNormSpikeConfig())
        assert det.check(0, grad_norm=None) is None

    def test_disabled_noop(self):
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(enabled=False)
        )
        for step in range(50):
            assert det.check(step, grad_norm=999.0) is None

    def test_frozen_baseline_does_not_drift(self):
        """Slow upward drift should still alert when baseline is frozen.

        With a rolling baseline this would silently follow the trend; with
        ``baseline_mode='frozen'`` (the default for grad-norm), the baseline
        locks in once the rolling window fills.
        """
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(
                warmup_steps=5, sigma_multiplier=2.0, rolling_window=30
            )
        )
        # Fill the window with low, stable values to lock the frozen baseline.
        for step in range(35):
            det.check(step, grad_norm=1.0)
        # Slow drift upward — by step 80 grad_norm has tripled. A rolling
        # baseline would follow; the frozen baseline catches it.
        fired = False
        for step in range(35, 80):
            grad_norm = 1.0 + (step - 35) * 0.05
            alert = det.check(step, grad_norm=grad_norm)
            if alert is not None:
                fired = True
                break
        assert fired, "frozen baseline failed to catch slow drift"

    def test_zero_baseline_guard(self):
        """A constant-zero history must not divide by zero."""
        det = GradientNormSpikeDetector(
            GradientNormSpikeConfig(warmup_steps=2, rolling_window=15)
        )
        for step in range(20):
            # Should never raise.
            det.check(step, grad_norm=0.0)
