"""Integration tests for the Discord webhook alert sender.

We patch ``urllib.request.urlopen`` directly because ``_DiscordSender`` uses
stdlib HTTP (project rule: zero new network deps in core; the existing CI
forbidden-pattern grep already carves out ``alerts.py``). Same approach as
``test_alerts_email.py``: catch the ``Request`` object that gets passed to
``urlopen``, decode its body, and assert on the JSON payload.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig, DiscordConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.integration

WEBHOOK = "https://discord.com/api/webhooks/test/test"


def _alert(severity="critical"):
    return Alert(
        detector="entropy_collapse",
        severity=severity,
        step=42,
        message="Entropy collapsed.",
        metric_values={"current_entropy": 0.1, "threshold": 1.0},
        recommendation="Reduce LR by 5x.",
    )


def _wait_for_threads(timeout: float = 5.0) -> None:
    """Join every non-main thread so the daemon Discord worker can finish."""
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in list(threading.enumerate()):
        if t is main or not t.is_alive():
            continue
        t.join(timeout=max(0.0, deadline - time.monotonic()))


def _make_response(status: int = 204):
    """Mimic the ``urlopen`` context-manager response."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@contextmanager
def _patched_urlopen(status: int = 204):
    """Patch the urlopen used by _DiscordSender.send (imported lazily)."""
    with patch("urllib.request.urlopen") as urlopen_mock:
        urlopen_mock.return_value = _make_response(status=status)
        yield urlopen_mock


def _payload_from_call(urlopen_mock) -> dict:
    """Pull the JSON body out of the ``Request`` object the sender posted."""
    req = urlopen_mock.call_args.args[0]
    return json.loads(req.data.decode("utf-8"))


def _make_manager(**discord_kwargs) -> AlertManager:
    cfg = AlertConfig(
        discord=DiscordConfig(enabled=True, webhook_url=WEBHOOK, **discord_kwargs),
        cooldown_steps=0,
    )
    return AlertManager(cfg, run_id="discord_test")


class TestDiscordPayload:
    def test_critical_payload_contains_alert_fields(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            assert mgr.send(_alert(severity="critical")) is True
            _wait_for_threads()

            urlopen_mock.assert_called_once()
            payload = _payload_from_call(urlopen_mock)

            assert payload["username"] == "rlwatch"
            assert len(payload["embeds"]) == 1
            embed = payload["embeds"][0]
            assert "entropy_collapse" in embed["title"]
            assert "CRITICAL" in embed["title"]
            assert "🚨" in embed["title"]
            assert embed["color"] == 0xFF0000  # red for critical
            assert embed["description"] == "Entropy collapsed."

            field_names = [f["name"] for f in embed["fields"]]
            assert "Run" in field_names
            assert "Step" in field_names
            assert "Recommended action" in field_names
            # Per-metric fields use backticked keys.
            assert "`current_entropy`" in field_names

    def test_warning_uses_orange_color_and_warning_emoji(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(severity="warning"))
            _wait_for_threads()

            embed = _payload_from_call(urlopen_mock)["embeds"][0]
            assert embed["color"] == 0xFFA500  # orange for warning
            assert "⚠️" in embed["title"]
            assert "WARNING" in embed["title"]

    def test_request_uses_post_method_and_json_content_type(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert())
            _wait_for_threads()

            req = urlopen_mock.call_args.args[0]
            assert req.get_method() == "POST"
            # Header keys are normalized to title-case by urllib.
            assert req.headers.get("Content-type") == "application/json"


class TestDiscordMentions:
    def test_critical_mentions_role(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(mention_role_ids=["123456"])
            mgr.send(_alert(severity="critical"))
            _wait_for_threads()

            payload = _payload_from_call(urlopen_mock)
            assert payload.get("content") == "<@&123456>"

    def test_warning_does_not_mention_role(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(mention_role_ids=["123456"])
            mgr.send(_alert(severity="warning"))
            _wait_for_threads()

            payload = _payload_from_call(urlopen_mock)
            assert "content" not in payload  # nothing to mention on warning

    def test_no_mention_when_no_role_ids_configured(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(severity="critical"))
            _wait_for_threads()

            payload = _payload_from_call(urlopen_mock)
            assert "content" not in payload


class TestDiscordErrorPaths:
    def test_5xx_response_logged_not_raised(self, caplog):
        with _patched_urlopen(status=500):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True  # AlertManager.send still True
            _wait_for_threads()
            assert any(
                "Discord webhook returned 500" in r.message for r in caplog.records
            )

    def test_url_error_logged_not_raised(self, caplog):
        with patch("urllib.request.urlopen", side_effect=URLError("nope")):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()
            assert any(
                "Failed to send Discord alert" in r.message for r in caplog.records
            )

    def test_unexpected_exception_logged_not_raised(self, caplog):
        with patch("urllib.request.urlopen", side_effect=ValueError("weird")):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()
            assert any(
                "Unexpected Discord send error" in r.message for r in caplog.records
            )


class TestDiscordChannelWiring:
    def test_discord_disabled_when_no_url(self):
        cfg = AlertConfig()  # default DiscordConfig.enabled=False
        mgr = AlertManager(cfg, run_id="r")
        assert mgr._discord_client is None

    def test_discord_enabled_when_url_set(self):
        cfg = AlertConfig(
            discord=DiscordConfig(enabled=True, webhook_url=WEBHOOK)
        )
        mgr = AlertManager(cfg, run_id="r")
        assert mgr._discord_client is not None
