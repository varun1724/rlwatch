"""Integration tests for the generic HTTP webhook alert sender.

The webhook sender uses ``string.Template`` ``${field}`` substitution into a
JSON body, then validates the substituted body with ``json.loads`` before
sending. The tests cover the default template, custom templates, JSON-escape
correctness for tricky message content, the invalid-template-after-substitution
fail-safe, custom headers, timeout propagation, and the cardinal-rule
"never raises into the training loop" guarantee.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from rlwatch.alerts import AlertManager, _DEFAULT_WEBHOOK_TEMPLATE, _json_escape
from rlwatch.config import AlertConfig, WebhookConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.integration

URL = "https://example.invalid/rlwatch"


def _alert(**overrides):
    base = dict(
        detector="kl_explosion",
        severity="critical",
        step=99,
        message="KL exploded.",
        metric_values={"kl_divergence": 5.0, "z_score": 12.3},
        recommendation="Lower LR.",
    )
    base.update(overrides)
    return Alert(**base)


def _wait_for_threads(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in list(threading.enumerate()):
        if t is main or not t.is_alive():
            continue
        t.join(timeout=max(0.0, deadline - time.monotonic()))


def _make_response(status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@contextmanager
def _patched_urlopen(status: int = 200):
    with patch("urllib.request.urlopen") as m:
        m.return_value = _make_response(status=status)
        yield m


def _body_from_call(urlopen_mock) -> str:
    req = urlopen_mock.call_args.args[0]
    return req.data.decode("utf-8")


def _make_manager(**webhook_kwargs) -> AlertManager:
    cfg = AlertConfig(
        webhook=WebhookConfig(enabled=True, url=URL, **webhook_kwargs),
        cooldown_steps=0,
    )
    return AlertManager(cfg, run_id="webhook_test_run")


class TestDefaultTemplate:
    def test_default_template_produces_valid_json(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()

            body = _body_from_call(urlopen_mock)
            parsed = json.loads(body)  # must parse
            assert parsed["detector"] == "kl_explosion"
            assert parsed["severity"] == "critical"
            assert parsed["step"] == 99
            assert parsed["run_id"] == "webhook_test_run"
            assert parsed["message"] == "KL exploded."
            assert parsed["recommendation"] == "Lower LR."
            assert parsed["metrics"]["kl_divergence"] == 5.0
            assert "timestamp" in parsed  # ISO8601 string

    def test_default_template_constant_uses_iso8601_timestamp(self):
        """The default template's ${timestamp} field should ISO-8601-format."""
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert())
            _wait_for_threads()

            parsed = json.loads(_body_from_call(urlopen_mock))
            from datetime import datetime
            # Should not raise.
            datetime.fromisoformat(parsed["timestamp"])


class TestCustomTemplates:
    def test_custom_template_substitution(self):
        custom = '{"alert": "${detector}-${severity_upper}", "at": ${step}}'
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(template_json=custom)
            mgr.send(_alert())
            _wait_for_threads()

            parsed = json.loads(_body_from_call(urlopen_mock))
            assert parsed == {"alert": "kl_explosion-CRITICAL", "at": 99}

    def test_invalid_template_after_substitution_logged_not_sent(self, caplog):
        # ${step} is unquoted; if a custom template uses it inside string
        # quotes, the result is still valid JSON. But if a template uses
        # ${metrics_json} inside string quotes, the result is NOT valid JSON
        # (the object literal collides with the surrounding quotes).
        broken = '{"x": "${metrics_json}"}'
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(template_json=broken)
            assert mgr.send(_alert()) is True
            _wait_for_threads()

            urlopen_mock.assert_not_called()
            assert any(
                "Webhook template produced invalid JSON" in r.message
                for r in caplog.records
            )


class TestJSONEscapeSafety:
    def test_message_with_double_quotes_escaped(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(message='He said "hi" and left.'))
            _wait_for_threads()

            body = _body_from_call(urlopen_mock)
            parsed = json.loads(body)  # must still parse
            assert parsed["message"] == 'He said "hi" and left.'

    def test_message_with_backslashes_escaped(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(message="path: C:\\Users\\foo\\bar"))
            _wait_for_threads()

            parsed = json.loads(_body_from_call(urlopen_mock))
            assert parsed["message"] == "path: C:\\Users\\foo\\bar"

    def test_message_with_newlines_escaped(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(message="line one\nline two"))
            _wait_for_threads()

            parsed = json.loads(_body_from_call(urlopen_mock))
            assert parsed["message"] == "line one\nline two"

    def test_message_with_unicode_escaped(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert(recommendation="résumé — naïveté"))
            _wait_for_threads()

            parsed = json.loads(_body_from_call(urlopen_mock))
            assert parsed["recommendation"] == "résumé — naïveté"

    def test_json_escape_helper_directly(self):
        # Direct unit-test of the _json_escape helper since it's the
        # load-bearing piece of the substitution-safety guarantee.
        assert _json_escape('he said "hi"') == 'he said \\"hi\\"'
        assert _json_escape("a\\b") == "a\\\\b"
        assert _json_escape("line\nbreak") == "line\\nbreak"
        assert _json_escape(None) == ""


class TestRequestShape:
    def test_custom_headers_sent(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(headers={"Authorization": "Bearer secret-token"})
            mgr.send(_alert())
            _wait_for_threads()

            req = urlopen_mock.call_args.args[0]
            # urllib title-cases header keys.
            assert req.headers.get("Authorization") == "Bearer secret-token"
            assert req.headers.get("Content-type") == "application/json"

    def test_method_defaults_to_post(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager()
            mgr.send(_alert())
            _wait_for_threads()

            req = urlopen_mock.call_args.args[0]
            assert req.get_method() == "POST"

    def test_method_can_be_put(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(method="PUT")
            mgr.send(_alert())
            _wait_for_threads()

            req = urlopen_mock.call_args.args[0]
            assert req.get_method() == "PUT"

    def test_timeout_propagated(self):
        with _patched_urlopen() as urlopen_mock:
            mgr = _make_manager(timeout_seconds=42)
            mgr.send(_alert())
            _wait_for_threads()

            # urlopen called with timeout=42 in kwargs.
            kwargs = urlopen_mock.call_args.kwargs
            assert kwargs.get("timeout") == 42


class TestErrorPaths:
    def test_url_error_logged_not_raised(self, caplog):
        with patch("urllib.request.urlopen", side_effect=URLError("nope")):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()
            assert any(
                "Failed to send webhook alert" in r.message for r in caplog.records
            )

    def test_5xx_response_logged_not_raised(self, caplog):
        with _patched_urlopen(status=500):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()
            assert any(
                "Webhook returned 500" in r.message for r in caplog.records
            )

    def test_unexpected_exception_logged_not_raised(self, caplog):
        with patch("urllib.request.urlopen", side_effect=ValueError("weird")):
            mgr = _make_manager()
            assert mgr.send(_alert()) is True
            _wait_for_threads()
            assert any(
                "Unexpected webhook send error" in r.message for r in caplog.records
            )


class TestWebhookChannelWiring:
    def test_webhook_disabled_when_no_url(self):
        cfg = AlertConfig()  # default WebhookConfig.enabled=False
        mgr = AlertManager(cfg, run_id="r")
        assert mgr._webhook_client is None

    def test_webhook_enabled_when_url_set(self):
        cfg = AlertConfig(webhook=WebhookConfig(enabled=True, url=URL))
        mgr = AlertManager(cfg, run_id="r")
        assert mgr._webhook_client is not None

    def test_default_template_constant_is_valid_json_with_placeholders(self):
        """The default template, before substitution, must look like JSON
        with `${...}` substitution markers in plausible places."""
        # Quick smoke check on the constant itself.
        assert "${detector}" in _DEFAULT_WEBHOOK_TEMPLATE
        assert "${step}" in _DEFAULT_WEBHOOK_TEMPLATE
        assert "${metrics_json}" in _DEFAULT_WEBHOOK_TEMPLATE
