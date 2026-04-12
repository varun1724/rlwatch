"""Unit tests for the RewardMeanDriftDetector."""

from __future__ import annotations

from rlwatch.config import RewardMeanDriftConfig
from rlwatch.detectors import RewardMeanDriftDetector


class TestRewardMeanDriftDetector:
    def test_no_alert_during_warmup(self):
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(warmup_steps=20, consecutive_steps=5)
        )
        for step in range(20):
            assert det.check(step, reward_mean=-1.0 + 0.1 * step) is None

    def test_no_alert_for_oscillating_reward(self):
        """A healthy oscillating reward_mean should never fire."""
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(warmup_steps=5, consecutive_steps=10)
        )
        for step in range(200):
            # Oscillates between -0.5 and 0.5 — never monotone for 10 steps.
            reward = 0.5 if step % 2 == 0 else -0.5
            assert det.check(step, reward_mean=reward) is None

    def test_upward_drift_fires_warning(self):
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=5,
                consecutive_steps=10,
                min_drift_magnitude=0.05,
            )
        )
        # Warmup.
        for step in range(5):
            det.check(step, reward_mean=0.0)
        # Monotone upward drift.
        alerts = []
        for step in range(5, 30):
            alert = det.check(step, reward_mean=0.01 * step)
            if alert:
                alerts.append(alert)
        assert len(alerts) >= 1
        assert alerts[0].detector == "reward_mean_drift"
        assert alerts[0].severity == "warning"
        assert alerts[0].metric_values["direction"] == "up"

    def test_downward_drift_fires_warning(self):
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=5,
                consecutive_steps=10,
                min_drift_magnitude=0.05,
            )
        )
        for step in range(5):
            det.check(step, reward_mean=1.0)
        alerts = []
        for step in range(5, 30):
            alert = det.check(step, reward_mean=1.0 - 0.01 * step)
            if alert:
                alerts.append(alert)
        assert len(alerts) >= 1
        assert alerts[0].metric_values["direction"] == "down"

    def test_recovery_resets_counter(self):
        """If the drift reverses, the counter resets."""
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=2,
                consecutive_steps=10,
                min_drift_magnitude=0.01,
            )
        )
        # Warmup.
        for step in range(2):
            det.check(step, reward_mean=0.0)
        # 8 steps of upward drift (not enough to fire).
        for step in range(2, 10):
            det.check(step, reward_mean=0.01 * step)
        # Reverse direction — resets the counter.
        det.check(10, reward_mean=-1.0)
        # 8 more steps of upward drift (should NOT fire because we reset).
        alerts = []
        for step in range(11, 19):
            alert = det.check(step, reward_mean=0.01 * step)
            if alert:
                alerts.append(alert)
        assert len(alerts) == 0

    def test_magnitude_too_small_no_alert(self):
        """If drift is monotone but very small, don't fire."""
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=2,
                consecutive_steps=5,
                min_drift_magnitude=1.0,  # high threshold
            )
        )
        for step in range(2):
            det.check(step, reward_mean=0.0)
        # Monotone upward but magnitude < 1.0.
        for step in range(2, 20):
            alert = det.check(step, reward_mean=0.001 * step)
            assert alert is None

    def test_none_input_noop(self):
        det = RewardMeanDriftDetector(RewardMeanDriftConfig())
        assert det.check(0, reward_mean=None) is None

    def test_disabled_noop(self):
        det = RewardMeanDriftDetector(RewardMeanDriftConfig(enabled=False))
        for step in range(100):
            assert det.check(step, reward_mean=0.01 * step) is None

    def test_recommendation_present(self):
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=2, consecutive_steps=5, min_drift_magnitude=0.01
            )
        )
        for step in range(2):
            det.check(step, reward_mean=0.0)
        alert = None
        for step in range(2, 20):
            alert = det.check(step, reward_mean=0.01 * step)
            if alert:
                break
        assert alert is not None
        assert alert.recommendation  # cardinal rule #3

    def test_warning_fires_once_per_episode(self):
        """Should fire once, not every subsequent step."""
        det = RewardMeanDriftDetector(
            RewardMeanDriftConfig(
                warmup_steps=2, consecutive_steps=5, min_drift_magnitude=0.01
            )
        )
        for step in range(2):
            det.check(step, reward_mean=0.0)
        alerts = []
        for step in range(2, 50):
            alert = det.check(step, reward_mean=0.01 * step)
            if alert:
                alerts.append(alert)
        assert len(alerts) == 1
