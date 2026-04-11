"""Integration tests for the Slack alert sender.

We patch ``slack_sdk.webhook.WebhookClient`` directly because slack_sdk uses
``urllib`` under the hood and ``responses`` only intercepts the ``requests``
library. CLAUDE.md explicitly blesses ``unittest.mock.patch`` for this kind
of integration boundary.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig, SlackConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.integration

WEBHOOK = "https://hooks.slack.com/services/test/test/test"


def _alert(severity="critical"):
    return Alert(
        detector="entropy_collapse",
        severity=severity,
        step=42,
        message="Entropy collapsed.",
        metric_values={"current_entropy": 0.1, "threshold": 1.0},
        recommendation="Reduce LR.",
    )


def _wait_for_threads(timeout: float = 5.0) -> None:
    """Join every non-main thread so the daemon Slack worker can finish."""
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in list(threading.enumerate()):
        if t is main or not t.is_alive():
            continue
        t.join(timeout=max(0.0, deadline - time.monotonic()))


def _make_response(status_code: int = 200, body: str = "ok"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.body = body
    return resp


def test_slack_payload_contains_alert_fields():
    with patch("slack_sdk.webhook.WebhookClient") as webhook_cls:
        instance = MagicMock()
        instance.send.return_value = _make_response(200)
        webhook_cls.return_value = instance

        cfg = AlertConfig(
            slack=SlackConfig(enabled=True, webhook_url=WEBHOOK),
            cooldown_steps=0,
        )
        mgr = AlertManager(cfg, run_id="test_run")
        assert mgr.send(_alert()) is True
        _wait_for_threads()

        webhook_cls.assert_called_once_with(WEBHOOK)
        instance.send.assert_called_once()
        # Verify content via the kwargs passed to send().
        kwargs = instance.send.call_args.kwargs
        assert "blocks" in kwargs
        rendered = str(kwargs["blocks"])
        assert "entropy_collapse" in rendered
        assert "CRITICAL" in rendered
        assert "test_run" in rendered
        assert "Reduce LR" in rendered


def test_slack_failure_logged_not_raised(caplog):
    with patch("slack_sdk.webhook.WebhookClient") as webhook_cls:
        instance = MagicMock()
        instance.send.return_value = _make_response(400, "invalid_payload")
        webhook_cls.return_value = instance

        cfg = AlertConfig(
            slack=SlackConfig(enabled=True, webhook_url=WEBHOOK),
            cooldown_steps=0,
        )
        mgr = AlertManager(cfg, run_id="test_run")
        assert mgr.send(_alert()) is True
        _wait_for_threads()

        assert any(
            "Slack webhook returned 400" in r.message for r in caplog.records
        )


def test_slack_exception_logged_not_raised(caplog):
    with patch("slack_sdk.webhook.WebhookClient") as webhook_cls:
        instance = MagicMock()
        instance.send.side_effect = ConnectionError("nope")
        webhook_cls.return_value = instance

        cfg = AlertConfig(
            slack=SlackConfig(enabled=True, webhook_url=WEBHOOK),
            cooldown_steps=0,
        )
        mgr = AlertManager(cfg, run_id="r")
        # Must not raise; daemon thread swallows.
        assert mgr.send(_alert()) is True
        _wait_for_threads()
        assert any("Failed to send Slack alert" in r.message for r in caplog.records)


def test_slack_emoji_differs_by_severity():
    with patch("slack_sdk.webhook.WebhookClient") as webhook_cls:
        instance = MagicMock()
        instance.send.return_value = _make_response(200)
        webhook_cls.return_value = instance

        cfg = AlertConfig(
            slack=SlackConfig(enabled=True, webhook_url=WEBHOOK),
            cooldown_steps=0,
        )
        mgr = AlertManager(cfg, run_id="r")
        mgr.send(_alert(severity="critical"))
        _wait_for_threads()

        rendered = str(instance.send.call_args.kwargs["blocks"])
        assert "rotating_light" in rendered  # critical emoji
