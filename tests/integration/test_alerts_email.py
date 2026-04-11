"""Integration tests for the email alert sender via mocked smtplib.SMTP.

We use ``unittest.mock.patch`` rather than ``aiosmtpd`` per the plan's risk
mitigation — patching is faster, has zero install weight, and CLAUDE.md
explicitly blesses it for this purpose.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig, EmailConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.integration


def _wait_for_threads(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    main = threading.main_thread()
    for t in list(threading.enumerate()):
        if t is main or not t.is_alive():
            continue
        t.join(timeout=max(0.0, deadline - time.monotonic()))


def _alert():
    return Alert(
        detector="kl_explosion",
        severity="critical",
        step=99,
        message="KL exploded.",
        metric_values={"kl_divergence": 5.0},
        recommendation="Lower LR.",
    )


def _make_manager() -> AlertManager:
    cfg = AlertConfig(
        email=EmailConfig(
            enabled=True,
            smtp_host="smtp.test.local",
            smtp_port=25,
            from_addr="bot@test",
            to_addrs=["alice@test"],
        ),
        cooldown_steps=0,
    )
    return AlertManager(cfg, run_id="email_test_run")


def test_email_send_calls_smtp():
    with patch("rlwatch.alerts.smtplib.SMTP") as smtp_cls:
        instance = MagicMock()
        smtp_cls.return_value.__enter__.return_value = instance

        mgr = _make_manager()
        assert mgr.send(_alert()) is True
        _wait_for_threads()

        smtp_cls.assert_called_once_with("smtp.test.local", 25)
        instance.starttls.assert_called_once()
        instance.sendmail.assert_called_once()
        args, _ = instance.sendmail.call_args
        from_addr, to_addrs, body = args
        assert from_addr == "bot@test"
        assert to_addrs == ["alice@test"]
        assert "kl_explosion" in body
        assert "CRITICAL" in body
        assert "Lower LR" in body


def test_email_failure_logged_not_raised(caplog):
    with patch(
        "rlwatch.alerts.smtplib.SMTP", side_effect=ConnectionRefusedError("nope")
    ):
        mgr = _make_manager()
        # Must not raise; daemon thread swallows and logs.
        assert mgr.send(_alert()) is True
        _wait_for_threads()
        assert any("Failed to send email" in r.message for r in caplog.records)
