# Slack

rlwatch posts alerts to Slack via incoming webhooks. The webhook URL is the only required setting.

## Setup

1. Create an incoming webhook in your Slack workspace: **Apps → Incoming Webhooks → Add to Slack**.
2. Copy the webhook URL (looks like `https://hooks.slack.com/services/T.../B.../...`).
3. Configure rlwatch one of two ways:

**Environment variable** (recommended for CI/CD):
```bash
export RLWATCH_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

**`rlwatch.yaml`:**
```yaml
alerts:
  slack:
    webhook_url: "https://hooks.slack.com/services/..."
```

You don't need to set `enabled: true` explicitly — rlwatch flips the flag automatically when the webhook URL is set.

## Payload format

Each alert becomes a Slack message with [block kit](https://api.slack.com/block-kit) blocks:

- **Header** with severity emoji (`🚨` for critical, `⚠️` for warning) and detector name
- **Section** with run id and step
- **Section** with the alert message
- **Section** with the recommended action
- **Section** with up to 10 metric values from the alert

## Mentioning users

```yaml
alerts:
  slack:
    webhook_url: "..."
    channel: "#ml-alerts"
    mention_users: ["U12345", "U67890"]
```

`mention_users` takes Slack member IDs (the ones that look like `U...`), not display names. You can find a member ID by clicking a user's profile and copying it from "More → Copy member ID".

## Failure handling

If the webhook returns a non-200, the failure is logged at error level. If the connection fails, the failure is logged. Either way, training is not interrupted — Slack delivery runs in a daemon thread and exceptions are caught.
