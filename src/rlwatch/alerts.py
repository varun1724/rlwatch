"""Alert delivery via Slack webhook and email."""

from __future__ import annotations

import asyncio
import logging
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from rlwatch.config import AlertConfig
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
