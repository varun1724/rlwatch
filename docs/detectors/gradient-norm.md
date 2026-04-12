# Gradient norm spike

**Watches:** the L2 norm of gradients passed back through the policy network.
**Failure mode:** gradient explosion. The most common precursor to a loss NaN — by the time you see this firing, the optimizer is one bad step away from blowing up entirely.

## How it fires

Z-score model identical to KL explosion. Maintains a rolling window of recent grad norms. Fires a **warning** when the current grad norm is more than `sigma_multiplier` σ above the baseline mean, and a **critical** when it's more than `1.5 × sigma_multiplier` σ above.

**Default `baseline_mode` is `"frozen"`**, not rolling. Gradient norms drift slowly on healthy runs, and a rolling baseline silently follows the trend — by the time something is "anomalously high" against the rolling baseline, it's already been climbing for hundreds of steps. The frozen baseline locks in once the rolling window first fills, mirroring how `RewardHackingDetector` already works.

## Configuration

```yaml
gradient_norm_spike:
  enabled: true
  sigma_multiplier: 3.0
  rolling_window: 100
  warmup_steps: 20
  baseline_mode: frozen   # default — see above
```

## Why frozen by default

This is the only detector where the v0.3 default differs from KL explosion and advantage variance. The reasoning:

- KL and advantage have noisy step-to-step values where the rolling baseline is genuinely informative.
- Grad norm is monotonically growing on most healthy runs (the policy explores wider regions of weight space). A rolling baseline tracks that growth and stops alerting on real spikes.
- The cost of frozen baseline is "alerts late in long runs once the model has legitimately moved." That's a mild false-positive, much milder than missing a real explosion.

## When to tune it

- **Higher `sigma_multiplier`** if you're getting late-run false positives. Long runs naturally widen the gradient distribution.
- **Switch to `baseline_mode: rolling`** if your model has a known, large grad-norm step-change (e.g., from a curriculum stage transition) that you want the detector to absorb instead of alert on.
- **Larger `rolling_window`** to baseline against more steps and reduce sensitivity.

## Recommended action when it fires

> Clip gradients (max_grad_norm), reduce learning rate, or check for exploding activations in the loss path.

In practice, **gradient clipping is the fix 9 times out of 10**. If `max_grad_norm` isn't already set in your training config, add it (1.0 is a sensible default). If it's already set and the alert still fires, the LR is probably too high — cut it 2-5× and restart from the last good checkpoint.
