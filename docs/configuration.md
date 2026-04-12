# Configuration

rlwatch is configured via a layered system: defaults, then a YAML file, then environment variables, then keyword arguments to `attach()`. Later layers override earlier ones.

## Resolution order

```
defaults → ./rlwatch.yaml → environment variables → attach(**kwargs)
```

If two layers set the same field, the later one wins. The YAML file is optional — without it, the defaults are sensible for most GRPO/PPO runs.

## Generating a starter YAML

```bash
rlwatch init
```

Writes `rlwatch.yaml` with every default, every detector, every threshold, and every alert channel. Edit it to taste.

## Full YAML schema

```yaml
# Run identifier (auto-generated if empty)
run_id: ""

# Framework: auto, trl, verl, openrlhf, manual
framework: auto

# --- Detectors ---

entropy_collapse:
  enabled: true
  threshold: 1.0
  consecutive_steps: 50
  warmup_steps: 20

kl_explosion:
  enabled: true
  sigma_multiplier: 3.0
  rolling_window: 100
  warmup_steps: 20
  baseline_mode: rolling     # or "frozen"
  clip_region: 0.2           # PPO clip region, surfaced in alert metrics

reward_hacking:
  enabled: true
  variance_multiplier: 3.0
  dip_test_significance: 0.05
  baseline_window: 100
  warmup_steps: 50

advantage_variance:
  enabled: true
  std_multiplier: 3.0
  rolling_window: 100
  warmup_steps: 20
  baseline_mode: rolling     # or "frozen"

loss_nan:
  enabled: true
  warmup_steps: 0

gradient_norm_spike:
  enabled: true
  sigma_multiplier: 3.0
  rolling_window: 100
  warmup_steps: 20
  baseline_mode: frozen      # default for grad norm — see detectors/gradient-norm

# --- Alert delivery ---

alerts:
  cooldown_steps: 100
  max_alerts_per_run: 50

  slack:
    enabled: false
    webhook_url: ""
    channel: ""
    mention_users: []

  email:
    enabled: false
    smtp_host: smtp.gmail.com
    smtp_port: 587
    smtp_user: ""
    smtp_password: ""
    from_addr: ""
    to_addrs: []

  discord:
    enabled: false
    webhook_url: ""
    username: rlwatch
    avatar_url: ""
    mention_role_ids: []

  webhook:
    enabled: false
    url: ""
    method: POST
    headers: {}
    template_json: ""
    timeout_seconds: 10

# --- Storage ---

storage:
  log_dir: ./rlwatch_logs
  db_name: metrics.db

# --- Dashboard ---

dashboard:
  port: 8501
  host: 0.0.0.0
```

## Environment variables

Every alert channel can be wired up via env vars without touching the YAML at all. This is the recommended path for CI/CD and containerized environments.

| Env var | Sets | Auto-enables |
|---|---|---|
| `RLWATCH_RUN_ID` | `run_id` | — |
| `RLWATCH_FRAMEWORK` | `framework` | — |
| `RLWATCH_LOG_DIR` | `storage.log_dir` | — |
| `RLWATCH_SLACK_WEBHOOK_URL` | `alerts.slack.webhook_url` | Slack |
| `RLWATCH_SLACK_CHANNEL` | `alerts.slack.channel` | — |
| `RLWATCH_SMTP_HOST` | `alerts.email.smtp_host` | — |
| `RLWATCH_SMTP_PORT` | `alerts.email.smtp_port` | — |
| `RLWATCH_SMTP_USER` | `alerts.email.smtp_user` | — |
| `RLWATCH_SMTP_PASSWORD` | `alerts.email.smtp_password` | — |
| `RLWATCH_EMAIL_FROM` | `alerts.email.from_addr` | — |
| `RLWATCH_EMAIL_TO` | `alerts.email.to_addrs` (comma-separated) | Email |
| `RLWATCH_DISCORD_WEBHOOK_URL` | `alerts.discord.webhook_url` | Discord |
| `RLWATCH_WEBHOOK_URL` | `alerts.webhook.url` | Webhook |
| `RLWATCH_WEBHOOK_TEMPLATE` | `alerts.webhook.template_json` | — |

"Auto-enables" means: setting that env var also flips the channel's `enabled: true` flag, so you don't have to write a YAML file just to turn it on.

## Keyword overrides

Anything in the YAML can be overridden when you call `attach()`:

```python
import rlwatch

monitor = rlwatch.attach(
    framework="manual",
    run_id="my_run_v3",
    entropy_collapse={"threshold": 0.5, "consecutive_steps": 30},  # nested deep-merge
)
```

Nested-dict kwargs are deep-merged into the corresponding sub-dataclass — passing `entropy_collapse={"threshold": 0.5}` does **not** wipe out the other entropy collapse fields. Unknown nested keys raise a `ValueError` so typos fail loudly.
