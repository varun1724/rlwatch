"""Tests for rlwatch detectors."""

import numpy as np
import pytest

from rlwatch.config import (
    AdvantageVarianceConfig,
    EntropyCollapseConfig,
    KLExplosionConfig,
    RewardHackingConfig,
    RLWatchConfig,
)
from rlwatch.detectors import (
    AdvantageVarianceDetector,
    DetectorSuite,
    EntropyCollapseDetector,
    KLExplosionDetector,
    RewardHackingDetector,
)


class TestEntropyCollapseDetector:
    def test_no_alert_during_warmup(self):
        config = EntropyCollapseConfig(warmup_steps=10, threshold=1.0, consecutive_steps=5)
        detector = EntropyCollapseDetector(config)

        # Feed low entropy during warmup — should not alert
        for step in range(10):
            alert = detector.check(step, entropy=0.1)
            assert alert is None

    def test_no_alert_when_entropy_healthy(self):
        config = EntropyCollapseConfig(warmup_steps=5, threshold=1.0, consecutive_steps=5)
        detector = EntropyCollapseDetector(config)

        for step in range(100):
            alert = detector.check(step, entropy=2.5)
            assert alert is None

    def test_alerts_on_entropy_collapse(self):
        config = EntropyCollapseConfig(warmup_steps=5, threshold=1.0, consecutive_steps=10)
        detector = EntropyCollapseDetector(config)

        # Warmup with healthy entropy
        for step in range(10):
            detector.check(step, entropy=2.5)

        # Feed collapsing entropy
        alerts = []
        for step in range(10, 30):
            alert = detector.check(step, entropy=0.3)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert any(a.severity == "critical" for a in alerts)
        assert alerts[0].detector == "entropy_collapse"

    def test_no_alert_when_disabled(self):
        config = EntropyCollapseConfig(enabled=False)
        detector = EntropyCollapseDetector(config)

        for step in range(100):
            alert = detector.check(step, entropy=0.01)
            assert alert is None

    def test_none_entropy_ignored(self):
        config = EntropyCollapseConfig()
        detector = EntropyCollapseDetector(config)
        assert detector.check(0, entropy=None) is None

    def test_warning_and_critical_each_fire_once(self):
        """A sustained collapse should emit exactly one warning then one critical.

        Regression test for the prior behavior where the warning branch
        returned an Alert on every step from ``consecutive_steps // 2`` up to
        ``consecutive_steps``, relying on ``AlertManager`` cooldown to filter
        duplicates.
        """
        config = EntropyCollapseConfig(
            warmup_steps=5, threshold=1.0, consecutive_steps=10
        )
        detector = EntropyCollapseDetector(config)

        # Warmup with healthy entropy so the initial-entropy baseline exists.
        for step in range(5):
            assert detector.check(step, entropy=2.5) is None

        # Feed 15 consecutive steps below the threshold and collect every alert.
        alerts = []
        for step in range(5, 20):
            alert = detector.check(step, entropy=0.3)
            if alert:
                alerts.append(alert)

        warnings = [a for a in alerts if a.severity == "warning"]
        criticals = [a for a in alerts if a.severity == "critical"]
        assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"
        assert len(criticals) == 1, f"expected 1 critical, got {len(criticals)}"
        # Warning must come before critical.
        assert warnings[0].step < criticals[0].step

    def test_alerts_re_arm_after_recovery(self):
        """If entropy recovers and collapses again, both tiers should re-fire."""
        config = EntropyCollapseConfig(
            warmup_steps=2, threshold=1.0, consecutive_steps=6
        )
        detector = EntropyCollapseDetector(config)

        # Warmup + healthy baseline.
        for step in range(5):
            detector.check(step, entropy=2.5)

        # First collapse.
        first_alerts = []
        for step in range(5, 15):
            alert = detector.check(step, entropy=0.2)
            if alert:
                first_alerts.append(alert)

        # Recover.
        for step in range(15, 20):
            detector.check(step, entropy=2.5)

        # Second collapse.
        second_alerts = []
        for step in range(20, 30):
            alert = detector.check(step, entropy=0.2)
            if alert:
                second_alerts.append(alert)

        assert any(a.severity == "critical" for a in first_alerts)
        assert any(a.severity == "critical" for a in second_alerts)
        assert any(a.severity == "warning" for a in second_alerts)


class TestKLExplosionDetector:
    def test_no_alert_during_warmup(self):
        config = KLExplosionConfig(warmup_steps=10)
        detector = KLExplosionDetector(config)

        for step in range(10):
            alert = detector.check(step, kl_divergence=100.0)
            assert alert is None

    def test_alerts_on_kl_spike(self):
        config = KLExplosionConfig(warmup_steps=5, sigma_multiplier=2.0, rolling_window=50)
        detector = KLExplosionDetector(config)

        # Feed stable KL
        for step in range(30):
            detector.check(step, kl_divergence=0.01 + np.random.normal(0, 0.001))

        # Spike KL
        alert = detector.check(31, kl_divergence=5.0)
        assert alert is not None
        assert alert.detector == "kl_explosion"

    def test_no_alert_for_stable_kl(self):
        config = KLExplosionConfig(warmup_steps=5, sigma_multiplier=3.0)
        detector = KLExplosionDetector(config)

        for step in range(100):
            alert = detector.check(step, kl_divergence=0.01)
            assert alert is None


class TestRewardHackingDetector:
    def test_no_alert_during_warmup(self):
        config = RewardHackingConfig(warmup_steps=20)
        detector = RewardHackingDetector(config)

        for step in range(20):
            alert = detector.check(step, reward_std=100.0)
            assert alert is None

    def test_alerts_on_variance_explosion(self):
        config = RewardHackingConfig(warmup_steps=10, variance_multiplier=3.0, baseline_window=20)
        detector = RewardHackingDetector(config)

        # Establish baseline
        for step in range(30):
            detector.check(step, reward_std=0.5)

        # Explode variance
        alert = detector.check(31, reward_std=5.0)
        assert alert is not None
        assert alert.detector == "reward_hacking"

    def test_no_alert_for_stable_rewards(self):
        config = RewardHackingConfig(warmup_steps=10, variance_multiplier=3.0, baseline_window=20)
        detector = RewardHackingDetector(config)

        for step in range(100):
            alert = detector.check(step, reward_std=0.5)
            assert alert is None


class TestAdvantageVarianceDetector:
    def test_alerts_on_spike(self):
        config = AdvantageVarianceConfig(warmup_steps=5, std_multiplier=3.0, rolling_window=50)
        detector = AdvantageVarianceDetector(config)

        # Stable baseline
        for step in range(30):
            detector.check(step, advantage_std=1.0)

        # Spike
        alert = detector.check(31, advantage_std=10.0)
        assert alert is not None
        assert alert.detector == "advantage_variance"

    def test_no_alert_for_stable_advantages(self):
        config = AdvantageVarianceConfig(warmup_steps=5, std_multiplier=3.0)
        detector = AdvantageVarianceDetector(config)

        for step in range(100):
            alert = detector.check(step, advantage_std=1.0)
            assert alert is None


class TestBaselineMode:
    """The new ``baseline_mode='frozen'`` option on KL and advantage detectors.

    Default for both is ``rolling`` (no behavior change vs. legacy). The
    ``frozen`` mode catches slow drift that the rolling baseline silently
    follows.
    """

    def test_kl_default_is_rolling(self):
        cfg = KLExplosionConfig()
        assert cfg.baseline_mode == "rolling"

    def test_kl_frozen_catches_slow_drift(self):
        cfg = KLExplosionConfig(
            warmup_steps=2,
            sigma_multiplier=2.0,
            rolling_window=20,
            baseline_mode="frozen",
        )
        det = KLExplosionDetector(cfg)
        # Fill the window with low, stable KL — locks the frozen baseline.
        for step in range(25):
            det.check(step, kl_divergence=0.01 + 0.0001 * (step % 3))
        # Slow upward drift. A rolling baseline would follow; frozen catches
        # it because the baseline stays anchored at ~0.01.
        fired = False
        for step in range(25, 200):
            kl = 0.01 + (step - 25) * 0.005
            if det.check(step, kl_divergence=kl) is not None:
                fired = True
                break
        assert fired, "frozen baseline failed to catch slow KL drift"

    def test_advantage_frozen_catches_slow_drift(self):
        cfg = AdvantageVarianceConfig(
            warmup_steps=2,
            std_multiplier=2.0,
            rolling_window=20,
            baseline_mode="frozen",
        )
        det = AdvantageVarianceDetector(cfg)
        for step in range(25):
            det.check(step, advantage_std=1.0 + 0.001 * (step % 3))
        fired = False
        for step in range(25, 200):
            adv = 1.0 + (step - 25) * 0.05
            if det.check(step, advantage_std=adv) is not None:
                fired = True
                break
        assert fired, "frozen baseline failed to catch slow advantage drift"


class TestDetectorSuite:
    def test_runs_all_detectors(self):
        config = RLWatchConfig()
        config.entropy_collapse.warmup_steps = 2
        config.entropy_collapse.consecutive_steps = 3
        config.kl_explosion.warmup_steps = 2
        config.reward_hacking.warmup_steps = 2
        config.advantage_variance.warmup_steps = 2

        suite = DetectorSuite(config)

        # Feed healthy metrics
        for step in range(10):
            alerts = suite.check_step(
                step,
                entropy=2.5,
                kl_divergence=0.01,
                reward_std=0.5,
                advantage_std=1.0,
            )
        # Healthy run should have no alerts
        assert len(alerts) == 0

    def test_detects_entropy_collapse(self):
        config = RLWatchConfig()
        config.entropy_collapse.warmup_steps = 2
        config.entropy_collapse.consecutive_steps = 5

        suite = DetectorSuite(config)

        # Warmup
        for step in range(5):
            suite.check_step(step, entropy=2.5)

        # Collapse
        all_alerts = []
        for step in range(5, 15):
            alerts = suite.check_step(step, entropy=0.1)
            all_alerts.extend(alerts)

        assert any(a.detector == "entropy_collapse" for a in all_alerts)
