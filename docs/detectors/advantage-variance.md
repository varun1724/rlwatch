# Advantage variance spike

**Watches:** the standard deviation of GAE advantages (or equivalent).
**Failure mode:** the value function estimates have become unstable. This often shows up just before training starts producing garbage gradients — the value head is no longer giving a useful baseline, so policy gradient updates have wildly inconsistent signs.

## How it fires

Same z-score model as KL explosion. Maintains a rolling window over `advantage_std`. Fires a **warning** when the current value is more than `std_multiplier` times the rolling mean, and a **critical** when it's more than `2 × std_multiplier` times the mean.

## Configuration

```yaml
advantage_variance:
  enabled: true
  std_multiplier: 3.0
  rolling_window: 100
  warmup_steps: 20
  baseline_mode: rolling   # or "frozen"
```

## Rolling vs frozen baseline

Same tradeoff as KL explosion. Default `rolling` catches sharp spikes; `frozen` catches slow drift. Default stays `rolling` for backwards compatibility.

## When to tune it

- **Higher `std_multiplier`** for noisier value functions (small batch sizes amplify advantage variance).
- **`baseline_mode: frozen`** if you suspect slow drift in advantage statistics.
- **Larger `rolling_window`** for long runs where the default 100-step window is too short.

## Recommended action when it fires

> Value function estimates are unstable. Reduce learning rate, increase batch size, or add advantage normalization.

In order of preference: (1) enable advantage normalization in your framework if it's not already on; (2) increase the batch size to smooth out per-step variance; (3) reduce LR. If none of those help, the value function may need more capacity or a longer warmup before being trusted as a baseline.
