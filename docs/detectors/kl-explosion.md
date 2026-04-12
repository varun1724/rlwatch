# KL divergence explosion

**Watches:** KL divergence between the policy and the reference model.
**Failure mode:** the policy is running away from the reference model. This is usually the prelude to reward hacking — the model has discovered an out-of-distribution behavior that the reward model rates highly but that's also nonsense.

## How it fires

Z-score model: rolling mean and std over the last `rolling_window` steps. When the current KL is more than `sigma_multiplier` standard deviations above the mean, it fires a **warning**. When it's more than `1.5 × sigma_multiplier` σ above, it fires a **critical**.

## Configuration

```yaml
kl_explosion:
  enabled: true
  sigma_multiplier: 3.0   # warning at 3σ, critical at 4.5σ
  rolling_window: 100     # baseline window
  warmup_steps: 20
  baseline_mode: rolling  # or "frozen" — see below
  clip_region: 0.2        # PPO clip region, surfaced in alert metrics for context
```

## Rolling vs frozen baseline

Default `baseline_mode: rolling` updates the baseline every step from the deque. This catches sharp spikes reliably. **It will silently follow slow drift** — if KL creeps up over several hundred steps, the rolling baseline creeps with it and the z-score never crosses the threshold.

Set `baseline_mode: frozen` to lock the baseline once the rolling window first fills (mirroring `RewardHackingDetector`). This catches drift, at the cost of being more sensitive late in long runs. The default stays `rolling` for backwards compatibility — flip to `frozen` if you've seen drift go undetected on a real run.

## When to tune it

- **Higher `sigma_multiplier`** (`5.0`) if you're getting false positives on legitimately bursty KL. PPO clip regions naturally produce some KL spikes.
- **`baseline_mode: frozen`** if you suspect slow KL drift on long runs.
- **Larger `rolling_window`** for noisier KL curves where 100 steps is too short to characterize the baseline.

## Recommended action when it fires

> Immediately reduce learning rate or increase KL penalty. Consider reverting to a previous checkpoint.

If KL has critically exploded, the run is usually unrecoverable from where it is — the reference distribution is too far away. Revert to a checkpoint from before the explosion and tighten the KL coefficient.
