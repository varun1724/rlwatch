# Reward hacking proxy

**Watches:** per-step reward variance (and, optionally, the per-sample reward distribution).
**Failure mode:** the policy has discovered an adversarial way to game the reward function. This typically manifests as either (a) reward variance suddenly exploding because *some* samples are exploiting the hack and others aren't, or (b) the reward distribution becoming bimodal — two clusters, one of "normal" rewards and one of "hacked" rewards.

## How it fires

Two parallel checks every step:

1. **Variance explosion.** Establishes a baseline variance from the first 20 post-warmup steps and freezes it. When the current variance exceeds `variance_multiplier × baseline_variance`, fires a **warning**. When it exceeds `2 × variance_multiplier × baseline`, fires a **critical**.

2. **Bimodal distribution.** If you pass `rewards=ndarray` to `log_step` (so the detector can see individual sample rewards, not just the std), it accumulates them in a buffer of up to 200 samples and runs Hartigan's dip test for unimodality. When the dip test p-value is below `dip_test_significance`, fires a **warning**.

The dip test uses the real `diptest` package when installed (`pip install "rlwatch[monitoring]"`), which provides the actual Hartigan algorithm with precomputed critical-value tables. When `diptest` isn't installed, it falls back to a simplified home-rolled approximation that is good enough for strongly bimodal distributions but unreliable for borderline cases. The variance explosion check is the primary signal in practice; the bimodal check is complementary.

## Configuration

```yaml
reward_hacking:
  enabled: true
  variance_multiplier: 3.0     # warning at 3x baseline, critical at 6x
  dip_test_significance: 0.05  # Hartigan dip test p-value threshold
  baseline_window: 100         # rolling buffer for variance history
  warmup_steps: 50             # longer warmup — reward variance is noisy early
```

## When to tune it

- **Higher `variance_multiplier`** if your reward function is intrinsically high-variance (e.g., a sparse 0/1 reward where any per-step variation looks like a 5x change).
- **Lower `dip_test_significance`** (0.01 or 0.001) for very strict bimodal detection. The simplified implementation is conservative; lower thresholds mean fewer false positives but also fewer true positives.
- **Longer `baseline_window`** if your baseline period is too noisy with the default of 100 steps.

## How to feed the dip test

Pass a numpy array to `log_step`:

```python
monitor.log_step(
    step,
    rewards=per_sample_rewards,  # numpy array
    # ... other metrics ...
)
```

If you pass `reward_std` instead, only the variance explosion check runs (no dip test).

## Recommended action when it fires

> Investigate reward model outputs for exploitation patterns. Consider adding reward model regularization or capping rewards.

The fix is rarely "tune the LR." Reward hacking means the *reward function* is broken — the policy found a hole in it. The right move is to look at the actual completions the model is producing, identify the hack, and either patch the reward function or add a verifier that rejects the bad pattern.
