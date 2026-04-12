# Discord

rlwatch posts alerts to Discord via webhooks. The webhook URL is the only required setting. Uses stdlib `urllib.request` — no extra dependencies.

## Setup

1. In your Discord server, **right-click a channel → Edit Channel → Integrations → Webhooks → New Webhook**.
2. Copy the webhook URL (looks like `https://discord.com/api/webhooks/{id}/{token}`).
3. Configure rlwatch:

**Environment variable:**
```bash
export RLWATCH_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

**`rlwatch.yaml`:**
```yaml
alerts:
  discord:
    webhook_url: "https://discord.com/api/webhooks/..."
    username: "rlwatch"          # optional override of the bot display name
    avatar_url: ""               # optional avatar override
    mention_role_ids: []         # role IDs to @-mention on critical alerts only
```

You don't need to set `enabled: true` — rlwatch flips it automatically when the webhook URL is set.

## Payload format

Each alert becomes a Discord message with one embed:

- **Title** with severity emoji (`🚨` for critical, `⚠️` for warning) and the detector name
- **Color** — red (`#FF0000`) for critical, orange (`#FFA500`) for warning
- **Description** = the alert message
- **Fields** for run id, step, the recommended action, and up to 10 metric values

## Mentioning roles

Discord roles can be `@`-mentioned by ID. To get a role ID:

1. Enable **Developer Mode** in **User Settings → Advanced**.
2. Right-click a role in your server settings → **Copy Role ID**.

Then:

```yaml
alerts:
  discord:
    webhook_url: "..."
    mention_role_ids: ["123456789012345678"]
```

**Role mentions only fire on critical alerts.** Warnings never page anyone — that's the contract. The reasoning: warnings are for awareness, criticals are for action. If a warning could ping the on-call rotation at 3am, you'd mute the channel.

## Failure handling

Discord webhooks return `204 No Content` on success. Non-204 status codes are logged at error level. Connection failures and unexpected exceptions are also caught and logged. Training is never interrupted — delivery runs in a daemon thread with a 10-second timeout.
