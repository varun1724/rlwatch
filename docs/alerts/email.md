# Email

rlwatch sends alerts over SMTP using the standard library `smtplib`. Works with Gmail (with an app password), AWS SES, SendGrid, Mailgun, or any other SMTP relay.

## Setup

```yaml
alerts:
  email:
    smtp_host: smtp.gmail.com
    smtp_port: 587
    smtp_user: alerts@yourcompany.com
    smtp_password: "..."          # use an app password for Gmail
    from_addr: alerts@yourcompany.com
    to_addrs:
      - oncall@yourcompany.com
      - you@yourcompany.com
```

Or via environment variables:

```bash
export RLWATCH_SMTP_HOST=smtp.gmail.com
export RLWATCH_SMTP_PORT=587
export RLWATCH_SMTP_USER=alerts@yourcompany.com
export RLWATCH_SMTP_PASSWORD="..."
export RLWATCH_EMAIL_FROM=alerts@yourcompany.com
export RLWATCH_EMAIL_TO="oncall@yourcompany.com,you@yourcompany.com"
```

`RLWATCH_EMAIL_TO` is comma-separated and auto-enables the email channel when set.

## Message format

Each alert is sent as a multipart message with both plain-text and HTML parts:

- **Subject:** `[rlwatch CRITICAL] entropy_collapse — Run grpo_v3_exp12 Step 320`
- **Plain text:** the alert message, the recommended action, and a key/value list of metric values
- **HTML:** the same content with severity-coded headers (red for critical, orange for warning) and a metric table

Most modern email clients show the HTML part. Plain-text is the fallback for terminal mail clients.

## TLS

The sender always calls `STARTTLS` after connecting. Plain unencrypted SMTP is not supported — for SMTP relays that don't require auth, leave `smtp_user` and `smtp_password` empty and TLS still happens.

## Failure handling

SMTP errors are logged at error level. The send runs in a daemon thread, so SMTP timeouts or auth failures don't block the training loop. If your SMTP relay is down for hours, you'll see error log lines but training keeps running.

## Gmail-specific notes

Gmail requires an [app password](https://support.google.com/accounts/answer/185833) — your regular Google account password won't work over SMTP. Generate one under **Google Account → Security → 2-Step Verification → App passwords**.
