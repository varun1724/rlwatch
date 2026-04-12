# Detectors

rlwatch ships with six detectors. Each one watches a specific failure mode that GRPO/PPO runs hit in practice. Every detector follows the same contract: configurable warmup, configurable severity thresholds, fires once per episode (not per step), and every alert ships with a recommended action.

## The lineup

| Detector | In plain English | Default trip-wire |
|---|---|---|
| [**Entropy collapse**](entropy-collapse.md) | The model stopped exploring — it's now just repeating itself. | Entropy < 1.0 for 50 steps in a row |
| [**KL divergence explosion**](kl-explosion.md) | The policy is running away from the reference model (usually the prelude to reward hacking). | KL > 3σ above the rolling mean |
| [**Reward hacking proxy**](reward-hacking.md) | Rewards suddenly got weird — either way more variance than before, or split into two clusters. | Variance > 3× baseline, **or** Hartigan dip test p < 0.05 |
| [**Advantage variance spike**](advantage-variance.md) | The value function estimates just became unstable. | Advantage std > 3× rolling baseline |
| [**Loss NaN / Inf**](loss-nan.md) | The optimizer has blown up; any further updates corrupt the policy. | Loss is non-finite (one step is enough) |
| [**Gradient norm spike**](gradient-norm.md) | Gradients exploded — usually the precursor to a loss NaN. | Grad norm > 3σ above frozen baseline |

Every detector has two severity levels (**warning** and **critical**), a configurable warmup period so it doesn't fire at step 3, and a cooldown so you don't get spammed.

## Common contract

Every detector exposes the same shape of config (see [configuration](../configuration.md)):

- `enabled: bool` — turn the detector off without removing it from your YAML.
- `warmup_steps: int` — number of steps to ignore at the start of training. Use a longer warmup for noisy initial steps.
- One or more numeric thresholds (the trip-wires).

And every detector emits an `Alert` with the same fields:

- `detector` — the detector's identifier (e.g., `"entropy_collapse"`)
- `severity` — `"warning"` or `"critical"`
- `step` — the training step at which the alert fired
- `message` — a one-sentence human-readable explanation
- `metric_values` — a dict of the relevant numeric values
- `recommendation` — what to actually do about it

## Severity tiers

Most detectors have two tiers:

- **Warning** — "this might be becoming a problem; check it out next time you look at the dashboard."
- **Critical** — "this is a problem right now; you should probably stop the run."

Critical alerts can preempt warnings within the cooldown window — if you see a warning and then a critical from the same detector five steps later, the critical is allowed through. The reverse is not true (a warning will not preempt an earlier critical).

## Cooldown and rate limiting

The `AlertManager` tracks `(detector, severity)` pairs against a cooldown window. By default, the same detector at the same severity won't re-fire within 100 steps, and a single run is capped at 50 total alerts. Both are configurable in `alerts.cooldown_steps` and `alerts.max_alerts_per_run`.

## Frozen vs rolling baselines

Two detectors (KL explosion, advantage variance spike) use a z-score model against a rolling baseline by default — this catches sharp spikes reliably but silently follows slow drift. The `baseline_mode: "rolling" | "frozen"` option freezes the baseline once the rolling window first fills, mirroring how `RewardHackingDetector` already works.

The new gradient norm spike detector defaults to `"frozen"` because grad norms drift slowly on healthy runs. KL and advantage default to `"rolling"` for backwards compatibility — if you want the frozen behavior, opt in explicitly:

```yaml
kl_explosion:
  baseline_mode: frozen
advantage_variance:
  baseline_mode: frozen
```

See [BUILD_DECISIONS.md on GitHub](https://github.com/varun1724/rlwatch/blob/main/BUILD_DECISIONS.md) for the full rationale.
