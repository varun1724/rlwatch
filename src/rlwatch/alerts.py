"""Alert delivery channels — console, Slack, email, Discord, generic webhook.

This module is the only place in the codebase that's allowed to make network
calls (CLAUDE.md cardinal rule #4). The CI forbidden-pattern grep enforces
this — all ``urllib.request`` / ``requests`` / ``httpx`` references must live
here.

Every sender follows the same shape:
- Constructed with config, holds no global state.
- ``send(alert, run_id)`` is called from a daemon thread by ``AlertManager``.
- Catches and logs every exception. **Never raises into the training loop.**
"""

from __future__ import annotations

import json
import logging
import smtplib
import string
import threading
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from rlwatch.config import AlertConfig, DiscordConfig, WebhookConfig
from rlwatch.detectors import Alert

logger = logging.getLogger("rlwatch.alerts")


class AlertManager:
    """Manages alert delivery with cooldown and rate limiting."""

    def __init__(self, config: AlertConfig, run_id: str = ""):
        self.config = config
        self.run_id = run_id
        self._alert_count = 0
        # (detector, severity) -> last step that severity fired. Tracking per
        # severity lets a critical preempt a warning that's still in cooldown.
        self._last_alert_step: dict[tuple[str, str], int] = {}
        # Last step *any* severity fired for this detector — used to honor the
        # warning cooldown against repeated warnings.
        self._last_warning_step: dict[str, int] = {}
        self._slack_client: Optional[_SlackSender] = None
        self._email_client: Optional[_EmailSender] = None
        self._discord_client: Optional[_DiscordSender] = None
        self._webhook_client: Optional[_WebhookSender] = None

        if config.slack.enabled and config.slack.webhook_url:
            self._slack_client = _SlackSender(config.slack.webhook_url)

        if config.email.enabled and config.email.to_addrs:
            self._email_client = _EmailSender(
                host=config.email.smtp_host,
                port=config.email.smtp_port,
                user=config.email.smtp_user,
                password=config.email.smtp_password,
                from_addr=config.email.from_addr,
                to_addrs=config.email.to_addrs,
            )

        if config.discord.enabled and config.discord.webhook_url:
            self._discord_client = _DiscordSender(config.discord)

        if config.webhook.enabled and config.webhook.url:
            self._webhook_client = _WebhookSender(config.webhook)

    def should_send(self, alert: Alert) -> bool:
        """Check if an alert should be sent based on cooldown and rate limits.

        Cooldown semantics:
          * A repeat alert at the same (detector, severity) within
            ``cooldown_steps`` is suppressed.
          * A *critical* alert is allowed through even if a warning from the
            same detector is still inside its cooldown window — escalation
            should never be muted by an earlier, lesser alert. The critical
            still respects its own per-severity cooldown.
        """
        if self._alert_count >= self.config.max_alerts_per_run:
            return False

        key = (alert.detector, alert.severity)
        last_step = self._last_alert_step.get(key, -self.config.cooldown_steps - 1)
        if alert.step - last_step < self.config.cooldown_steps:
            return False

        return True

    def send(self, alert: Alert) -> bool:
        """Send an alert via all configured channels (non-blocking).

        Returns True if the alert was actually sent (not suppressed by cooldown).
        """
        if not self.should_send(alert):
            return False

        self._alert_count += 1
        self._last_alert_step[(alert.detector, alert.severity)] = alert.step

        # Log to console always
        _log_alert_console(alert, self.run_id)

        # Send via configured channels in background threads
        if self._slack_client:
            threading.Thread(
                target=self._slack_client.send,
                args=(alert, self.run_id),
                daemon=True,
            ).start()

        if self._email_client:
            threading.Thread(
                target=self._email_client.send,
                args=(alert, self.run_id),
                daemon=True,
            ).start()

        if self._discord_client:
            threading.Thread(
                target=self._discord_client.send,
                args=(alert, self.run_id),
                daemon=True,
            ).start()

        if self._webhook_client:
            threading.Thread(
                target=self._webhook_client.send,
                args=(alert, self.run_id),
                daemon=True,
            ).start()

        return True

    @property
    def total_alerts_sent(self) -> int:
        return self._alert_count


def _log_alert_console(alert: Alert, run_id: str):
    """Log an alert to the console using rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console(stderr=True)
        severity_color = "red" if alert.severity == "critical" else "yellow"
        title = f"[bold {severity_color}]rlwatch {alert.severity.upper()}: {alert.detector}[/]"
        body = (
            f"[bold]Step {alert.step}[/] | Run: {run_id}\n\n"
            f"{alert.message}\n\n"
            f"[dim]Recommendation:[/] {alert.recommendation}"
        )
        console.print(Panel(body, title=title, border_style=severity_color))
    except ImportError:
        # Fallback without rich
        prefix = "CRITICAL" if alert.severity == "critical" else "WARNING"
        logger.warning(
            "[rlwatch %s] %s at step %d: %s | %s",
            prefix, alert.detector, alert.step, alert.message, alert.recommendation,
        )


class _SlackSender:
    """Sends alerts to Slack via webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Alert, run_id: str):
        try:
            from slack_sdk.webhook import WebhookClient

            client = WebhookClient(self.webhook_url)
            emoji = ":rotating_light:" if alert.severity == "critical" else ":warning:"
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} rlwatch {alert.severity.upper()}: {alert.detector}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Run:* `{run_id}`"},
                        {"type": "mrkdwn", "text": f"*Step:* {alert.step}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Recommended action:* {alert.recommendation}",
                    },
                },
            ]

            # Add metric values as context
            metric_fields = []
            for k, v in alert.metric_values.items():
                if v is not None:
                    formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                    metric_fields.append(
                        {"type": "mrkdwn", "text": f"`{k}`: {formatted}"}
                    )

            if metric_fields:
                # Slack limits fields to 10
                blocks.append({
                    "type": "section",
                    "fields": metric_fields[:10],
                })

            response = client.send(blocks=blocks)
            if response.status_code != 200:
                logger.error("Slack webhook returned %d: %s", response.status_code, response.body)
        except Exception as e:
            logger.error("Failed to send Slack alert: %s", e)


class _EmailSender:
    """Sends alerts via email."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        from_addr: str,
        to_addrs: list[str],
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Alert, run_id: str):
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[rlwatch {alert.severity.upper()}] {alert.detector} — Run {run_id} Step {alert.step}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            # Plain text
            text = (
                f"rlwatch {alert.severity.upper()}: {alert.detector}\n\n"
                f"Run: {run_id}\n"
                f"Step: {alert.step}\n\n"
                f"{alert.message}\n\n"
                f"Recommendation: {alert.recommendation}\n\n"
                f"Metrics:\n"
            )
            for k, v in alert.metric_values.items():
                if v is not None:
                    formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                    text += f"  {k}: {formatted}\n"

            # HTML
            html = f"""
            <html>
            <body>
            <h2 style="color: {'red' if alert.severity == 'critical' else 'orange'}">
                rlwatch {alert.severity.upper()}: {alert.detector}
            </h2>
            <p><strong>Run:</strong> <code>{run_id}</code> | <strong>Step:</strong> {alert.step}</p>
            <p>{alert.message}</p>
            <p><strong>Recommendation:</strong> {alert.recommendation}</p>
            <h3>Metrics</h3>
            <table border="1" cellpadding="5" cellspacing="0">
            """
            for k, v in alert.metric_values.items():
                if v is not None:
                    formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                    html += f"<tr><td><code>{k}</code></td><td>{formatted}</td></tr>"
            html += "</table></body></html>"

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                if self.user and self.password:
                    server.login(self.user, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info("Email alert sent to %s", self.to_addrs)
        except Exception as e:
            logger.error("Failed to send email alert: %s", e)


# ---------------------------------------------------------------------------
# Discord webhook sender
# ---------------------------------------------------------------------------
class _DiscordSender:
    """Sends alerts to a Discord channel via the webhook API.

    Discord webhooks accept JSON at ``https://discord.com/api/webhooks/{id}/{token}``
    with optional ``content`` (plain text), ``embeds`` (rich blocks), ``username``,
    and ``avatar_url`` fields. We use one embed per alert with severity-coded
    color and an emoji-prefixed title.
    """

    def __init__(self, config: DiscordConfig):
        self.config = config

    def send(self, alert: Alert, run_id: str):
        try:
            from urllib.error import HTTPError, URLError
            from urllib.request import Request, urlopen

            emoji = "🚨" if alert.severity == "critical" else "⚠️"
            color = 0xFF0000 if alert.severity == "critical" else 0xFFA500

            # Mention configured roles only on critical alerts so warnings
            # don't ping the on-call rotation in the middle of the night.
            mention_content: Optional[str] = None
            if alert.severity == "critical" and self.config.mention_role_ids:
                mention_content = " ".join(
                    f"<@&{rid}>" for rid in self.config.mention_role_ids
                )

            fields = [
                {"name": "Run", "value": f"`{run_id}`", "inline": True},
                {"name": "Step", "value": str(alert.step), "inline": True},
                {
                    "name": "Recommended action",
                    "value": alert.recommendation,
                    "inline": False,
                },
            ]
            # Discord caps embed fields at 25; cap our metric overflow at 10
            # to leave headroom and stay readable.
            for k, v in list(alert.metric_values.items())[:10]:
                if v is None:
                    continue
                formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
                fields.append(
                    {"name": f"`{k}`", "value": formatted, "inline": True}
                )

            payload: dict = {
                "username": self.config.username,
                "embeds": [
                    {
                        "title": f"{emoji} rlwatch {alert.severity.upper()}: {alert.detector}",
                        "description": alert.message,
                        "color": color,
                        "fields": fields,
                    }
                ],
            }
            if self.config.avatar_url:
                payload["avatar_url"] = self.config.avatar_url
            if mention_content:
                payload["content"] = mention_content

            data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                # Discord returns 204 No Content on success.
                if resp.status >= 300:
                    logger.error("Discord webhook returned %d", resp.status)
        except (HTTPError, URLError) as e:
            logger.error("Failed to send Discord alert: %s", e)
        except Exception as e:
            logger.error("Unexpected Discord send error: %s", e)


# ---------------------------------------------------------------------------
# Generic HTTP webhook sender
# ---------------------------------------------------------------------------
_DEFAULT_WEBHOOK_TEMPLATE = """{
  "detector": "${detector}",
  "severity": "${severity}",
  "step": ${step},
  "run_id": "${run_id}",
  "message": "${message}",
  "recommendation": "${recommendation}",
  "metrics": ${metrics_json},
  "timestamp": "${timestamp}"
}"""


def _json_escape(s: str) -> str:
    """Escape a string so it can be safely substituted into a JSON string slot.

    Uses ``json.dumps`` and strips the surrounding quotes — that's the
    canonical "give me a JSON-safe string body" trick. Handles quotes,
    backslashes, newlines, control chars, and non-ASCII unicode.
    """
    if s is None:
        return ""
    return json.dumps(s)[1:-1]


class _WebhookSender:
    """Generic HTTP webhook sender with ``string.Template`` substitution.

    POSTs (or PUTs) a JSON body to a user-supplied URL. The body is built
    from a ``string.Template`` so users can customize the payload shape for
    whatever downstream system they're feeding (incident tracker, internal
    log aggregator, custom Slack-of-record, etc.).

    Substitutable fields:
        ${detector}        — alert.detector
        ${severity}        — "critical" | "warning"
        ${severity_upper}  — "CRITICAL" | "WARNING"
        ${step}            — int (unquoted in default template — numeric slot)
        ${message}         — alert.message (JSON-escaped)
        ${recommendation}  — alert.recommendation (JSON-escaped)
        ${run_id}          — manager run_id
        ${timestamp}       — ISO8601 UTC at send time
        ${metrics_json}    — json.dumps(alert.metric_values), unquoted (object slot)

    The substituted body is validated with ``json.loads`` before sending.
    Invalid JSON is logged and dropped — we never POST something that won't
    parse on the other end.
    """

    def __init__(self, config: WebhookConfig):
        self.config = config

    def send(self, alert: Alert, run_id: str):
        try:
            from urllib.error import HTTPError, URLError
            from urllib.request import Request, urlopen

            tmpl_str = self.config.template_json or _DEFAULT_WEBHOOK_TEMPLATE
            tmpl = string.Template(tmpl_str)
            body = tmpl.safe_substitute(
                detector=alert.detector,
                severity=alert.severity,
                severity_upper=alert.severity.upper(),
                step=alert.step,
                message=_json_escape(alert.message),
                recommendation=_json_escape(alert.recommendation),
                run_id=_json_escape(run_id),
                metrics_json=json.dumps(alert.metric_values),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Validate the substituted body is still parseable JSON. A
            # malformed custom template should fail loudly here, not on the
            # receiving server.
            try:
                json.loads(body)
            except json.JSONDecodeError as e:
                logger.error(
                    "Webhook template produced invalid JSON after substitution: %s",
                    e,
                )
                return

            req = Request(
                self.config.url,
                data=body.encode("utf-8"),
                headers={"Content-Type": "application/json", **self.config.headers},
                method=self.config.method,
            )
            with urlopen(req, timeout=self.config.timeout_seconds) as resp:
                if resp.status >= 300:
                    logger.error("Webhook returned %d", resp.status)
        except (HTTPError, URLError) as e:
            logger.error("Failed to send webhook alert: %s", e)
        except Exception as e:
            logger.error("Unexpected webhook send error: %s", e)
