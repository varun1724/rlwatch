# Generic webhook

The universal escape hatch. POSTs (or PUTs) a JSON body to any URL. Use this for any system rlwatch doesn't have a dedicated channel for: PagerDuty's events API, Mattermost, Rocket.Chat, an internal log aggregator, a custom incident tracker, etc.

Stdlib HTTP — no extra dependencies.

## Setup

```yaml
alerts:
  webhook:
    url: "https://your-service.example.com/rlwatch"
    method: POST                # or PUT
    headers:
      Authorization: "Bearer your-token-here"
    template_json: ""           # empty = use the default template
    timeout_seconds: 10
```

Or via environment variables:

```bash
export RLWATCH_WEBHOOK_URL="https://your-service.example.com/rlwatch"
# Optional custom template:
export RLWATCH_WEBHOOK_TEMPLATE='{"alert": "${detector}-${severity_upper}", "step": ${step}}'
```

`RLWATCH_WEBHOOK_URL` auto-enables the channel when set. `RLWATCH_WEBHOOK_TEMPLATE` is optional — without it, the default template is used.

## Default payload

```json
{
  "detector": "entropy_collapse",
  "severity": "critical",
  "step": 320,
  "run_id": "grpo_v3_exp12",
  "message": "Entropy collapse detected — policy entropy dropped from 2.78 to 0.21 over 50 consecutive steps (threshold: 1.0).",
  "recommendation": "Reduce learning rate by 5x or increase KL penalty coefficient. Consider increasing entropy bonus if available.",
  "metrics": {
    "current_entropy": 0.21,
    "initial_entropy": 2.78,
    "consecutive_steps_below": 50,
    "threshold": 1.0
  },
  "timestamp": "2026-04-11T09:32:14.123456+00:00"
}
```

This is a sensible default for most consumers. If you need a different shape, write your own template (see below).

## Custom templates

The body is built from a [`string.Template`](https://docs.python.org/3/library/string.html#template-strings) with `${field}` substitution. After substitution, the result must still parse as JSON — rlwatch validates with `json.loads` before sending and refuses to POST something invalid.

### Substitutable fields

| Placeholder | Type | Notes |
|---|---|---|
| `${detector}` | string | The detector identifier (e.g., `entropy_collapse`) |
| `${severity}` | string | `critical` or `warning` |
| `${severity_upper}` | string | `CRITICAL` or `WARNING` |
| `${step}` | int | **unquoted** — numeric slot in JSON |
| `${message}` | string | JSON-escaped (quotes/backslashes/newlines/unicode handled) |
| `${recommendation}` | string | JSON-escaped |
| `${run_id}` | string | JSON-escaped |
| `${timestamp}` | string | ISO 8601 UTC at send time |
| `${metrics_json}` | object | **unquoted** — `json.dumps(alert.metric_values)`, an object literal |

### Quoting rules (the footgun)

`${step}` and `${metrics_json}` are **not quoted** in the default template — they're numeric and object slots respectively. If you put them inside string quotes in a custom template, the substituted result will be invalid JSON and rlwatch will log an error and drop the alert. Examples:

✅ **Correct** — `${step}` in a numeric slot:
```json
{"alert_step": ${step}}
```

❌ **Wrong** — quoted, will substitute to `{"alert_step": "320"}` which is type-mismatched (string instead of int) but actually still parses. The bigger problem is the `${metrics_json}` case:

❌ **Wrong** — `${metrics_json}` inside string quotes, produces invalid JSON:
```json
{"x": "${metrics_json}"}
```
After substitution this becomes `{"x": "{"current_entropy": 0.21, ...}"}` which is invalid JSON because the inner `"` ends the outer string. rlwatch detects this and logs an error.

✅ **Correct** — `${metrics_json}` in an object slot:
```json
{"all_metrics": ${metrics_json}}
```

### String fields are JSON-escaped automatically

`${message}`, `${recommendation}`, and `${run_id}` are passed through `_json_escape()` before substitution. Quotes, backslashes, newlines, and non-ASCII unicode all work without breaking the body.

## Custom headers

Useful for authentication:

```yaml
alerts:
  webhook:
    url: "https://api.example.com/incidents"
    headers:
      Authorization: "Bearer secret-token"
      X-Service-Name: "rlwatch"
```

Headers are merged with the default `Content-Type: application/json` — your custom headers are added but don't replace the content type.

## Failure handling

The same contract as every other channel: HTTP errors logged at error level, connection failures logged, unexpected exceptions caught. Training never blocks. Delivery runs in a daemon thread with the configured `timeout_seconds` (default 10).
