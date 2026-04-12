# Alerts

When a detector fires, rlwatch builds an `Alert` object and dispatches it through every configured channel. Alerts always go to the console; everything else is opt-in.

## Channels

- [**Console**](#console) — always on, rich-formatted panels in stderr
- [**Slack**](slack.md) — webhook via `slack_sdk`
- [**Email**](email.md) — SMTP via `smtplib`
- [**Discord**](discord.md) — webhook via stdlib HTTP
- [**Generic webhook**](webhook.md) — POST/PUT JSON to any URL

All non-console channels run in **daemon threads** so they never block the training loop. Failures in delivery are logged but never raised — if rlwatch crashes, training shouldn't.

## Cardinal rules

These are the contracts every channel honors. They're enforced in CI by the test harness and the forbidden-pattern grep.

1. **Every alert ships with a recommendation.** A bare metric value is not an alert. The alert message tells you what's wrong; the `recommendation` field tells you what to *do* about it.
2. **No network calls in core.** Only `src/rlwatch/alerts.py` is allowed to import network libraries (urllib, requests, slack_sdk, smtplib). The default install works offline, in air-gapped GPU clusters, with zero telemetry.
3. **Delivery never blocks training.** Daemon threads, bounded timeouts (10s default for HTTP), exceptions caught and logged.
4. **Cooldown can be preempted by escalation.** A critical alert that follows a warning from the same detector inside the cooldown window is allowed through. Escalation is never muted by an earlier lesser alert.

## Alert structure

Every alert has the same fields:

```python
@dataclass
class Alert:
    detector: str            # e.g., "entropy_collapse"
    severity: str            # "warning" or "critical"
    step: int                # training step at which it fired
    message: str             # one-sentence human-readable explanation
    metric_values: dict      # the relevant numeric values
    recommendation: str      # what to actually do about it
```

Channels render these fields differently — Slack uses block kit, Discord uses embeds, email has both plain-text and HTML versions, the generic webhook lets you template the payload yourself. The underlying data is identical.

## Cooldown and rate limiting

- **`alerts.cooldown_steps`** (default `100`): the same `(detector, severity)` won't re-fire within this many steps.
- **`alerts.max_alerts_per_run`** (default `50`): a single run is capped at this many total alerts. Defends against pager-bombing if a detector goes haywire.

Cooldown is tracked per `(detector, severity)` pair, which is why a critical can preempt a recent warning — they're different keys in the cooldown table.

## Console

Always on. Uses [Rich](https://rich.readthedocs.io/) to print color-coded panels to stderr. Critical alerts get a red border; warnings get yellow.

The console alert is what you see when you run `python examples/simulate_grpo_run.py` — it's the demo experience, and it works without configuring anything.
