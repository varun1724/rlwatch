# Reward mean drift

**Watches:** the reward mean over consecutive steps.
**Failure mode:** the reward function is being gamed *slowly*. The variance doesn't spike (so the [reward hacking detector](reward-hacking.md) doesn't fire), but the mean is drifting monotonically in one direction for an extended period. This is the signature of distribution-shift reward hacking — the model has found a way to consistently push the reward higher (or lower) without producing the variance explosion that the faster failure mode shows.

## How it fires

Counter-based: tracks the direction of reward-mean movement on every step. When reward_mean moves in the same direction (up or down) for `consecutive_steps` consecutive steps **and** the total magnitude of the drift exceeds `min_drift_magnitude`, fires a **warning**.

There is no critical tier — monotone drift is suspicious but not catastrophic. It could be legitimate improvement on a well-designed reward function (healthy training *should* increase reward monotonically in the early stages). The recommendation tells the user to investigate rather than stop immediately.

The warning fires **once per drift episode**. If the drift direction reverses, the counter resets and both warning flags re-arm.

## Configuration

```yaml
reward_mean_drift:
  enabled: true
  consecutive_steps: 50    # how many monotone steps before alerting
  warmup_steps: 50         # skip early training (reward IS expected to climb)
  min_drift_magnitude: 0.1 # ignore flat-but-technically-monotone curves
```

## When to tune it

- **Raise `warmup_steps`** if your reward legitimately improves monotonically for the first 100+ steps. `warmup_steps: 100` or `warmup_steps: 200` is reasonable for well-calibrated rewards.
- **Raise `min_drift_magnitude`** if your reward function has a naturally wide range and small monotone movements are expected.
- **Lower `consecutive_steps`** if you want faster detection at the cost of more false positives. `consecutive_steps: 30` is aggressive.
- **Disable entirely** (`enabled: false`) if your reward is designed to be monotonically increasing (e.g., a curriculum-based reward that always goes up as the model gets better).

## Why warning-only (no critical)

Monotone reward drift is genuinely ambiguous. Unlike entropy collapse (where entropy < 1.0 for 50 steps is always bad) or loss NaN (which is never acceptable), a drifting reward mean might be:

- **Reward hacking** — the model found a distribution-shift exploit
- **Legitimate learning** — the model is just getting better at the task

The detector can't tell the difference. Its job is to flag the pattern so you investigate — not to declare the run broken.

## Recommended action when it fires

> Reward mean has been drifting [up/down] monotonically for N steps. Check for reward hacking via distribution shift rather than variance explosion. Inspect the completions to check for exploitation patterns.

Look at the actual completions the model is producing. If they're getting better (more correct, more coherent, more aligned), the drift is legitimate. If they're getting weirder or gaming a loophole, that's reward hacking.
