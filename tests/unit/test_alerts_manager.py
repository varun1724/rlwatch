"""Unit tests for AlertManager — cooldown, rate limiting, preemption."""

from __future__ import annotations

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig, SlackConfig
from rlwatch.detectors import Alert


def _alert(detector="entropy_collapse", severity="warning", step=10):
    return Alert(
        detector=detector,
        severity=severity,
        step=step,
        message="test",
        metric_values={"x": 1.0},
        recommendation="test",
    )


class TestAlertManagerCooldown:
    def test_first_alert_passes(self):
        mgr = AlertManager(AlertConfig(cooldown_steps=100), run_id="r1")
        assert mgr.send(_alert(step=0)) is True

    def test_repeat_within_cooldown_suppressed(self):
        mgr = AlertManager(AlertConfig(cooldown_steps=50), run_id="r1")
        assert mgr.send(_alert(step=0)) is True
        assert mgr.send(_alert(step=10)) is False
        assert mgr.send(_alert(step=49)) is False

    def test_repeat_after_cooldown_passes(self):
        mgr = AlertManager(AlertConfig(cooldown_steps=50), run_id="r1")
        assert mgr.send(_alert(step=0)) is True
        assert mgr.send(_alert(step=51)) is True

    def test_different_detectors_independent(self):
        mgr = AlertManager(AlertConfig(cooldown_steps=100), run_id="r1")
        assert mgr.send(_alert(detector="entropy_collapse", step=0)) is True
        assert mgr.send(_alert(detector="kl_explosion", step=1)) is True

    def test_critical_preempts_warning_cooldown(self):
        """A critical must escalate even if a warning is mid-cooldown.

        Regression for the bug where ``_last_alert_step`` was indexed by
        detector only, causing a critical immediately after a warning to be
        silently dropped.
        """
        mgr = AlertManager(AlertConfig(cooldown_steps=100), run_id="r1")
        assert mgr.send(_alert(severity="warning", step=0)) is True
        # 5 steps later: a warning is still in cooldown but a critical must
        # be allowed through.
        assert mgr.send(_alert(severity="warning", step=5)) is False
        assert mgr.send(_alert(severity="critical", step=5)) is True

    def test_critical_has_its_own_cooldown(self):
        mgr = AlertManager(AlertConfig(cooldown_steps=50), run_id="r1")
        assert mgr.send(_alert(severity="critical", step=0)) is True
        assert mgr.send(_alert(severity="critical", step=10)) is False
        assert mgr.send(_alert(severity="critical", step=51)) is True


class TestAlertManagerRateLimit:
    def test_max_alerts_honored(self):
        mgr = AlertManager(
            AlertConfig(cooldown_steps=1, max_alerts_per_run=3), run_id="r1"
        )
        sent = []
        for step in range(10):
            sent.append(mgr.send(_alert(step=step * 5)))
        # Allowed up to 3 sends.
        assert sum(sent) == 3
        assert mgr.total_alerts_sent == 3


class TestAlertManagerChannels:
    def test_slack_disabled_when_no_url(self):
        mgr = AlertManager(AlertConfig(), run_id="r1")
        assert mgr._slack_client is None

    def test_slack_enabled_when_url_set(self):
        cfg = AlertConfig(
            slack=SlackConfig(enabled=True, webhook_url="https://hooks.slack.com/test")
        )
        mgr = AlertManager(cfg, run_id="r1")
        assert mgr._slack_client is not None
