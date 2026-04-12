# Entropy collapse

**Watches:** policy entropy.
**Failure mode:** the model has stopped exploring and is producing near-deterministic outputs. In a language model this looks like the policy converging on a single token or short repetitive phrase. The reward signal goes flat shortly after.

## How it fires

The detector keeps a running counter of consecutive steps where entropy is below `threshold`. When the counter reaches `consecutive_steps // 2`, it fires a **warning** (once). When the counter reaches `consecutive_steps`, it fires a **critical** (once). Both flags re-arm if entropy recovers above the threshold so a second collapse alerts again.

The `metric_values` on the alert include the current entropy, the "initial" entropy from a stable baseline collected during early healthy steps (so the comparison stays meaningful even after 500+ steps), and the consecutive-below counter at fire time.

## Configuration

```yaml
entropy_collapse:
  enabled: true
  threshold: 1.0          # entropy below this counts as "collapsed"
  consecutive_steps: 50   # ...for this many steps in a row
  warmup_steps: 20        # ignore the first 20 steps
```

## When to tune it

- **Lower the threshold** (`threshold: 0.5`) if your model legitimately runs at low entropy and you're getting false positives. Watch the dashboard for a few runs to find a baseline.
- **Lower `consecutive_steps`** (`consecutive_steps: 20`) if you want faster alerting at the cost of more false positives on noisy entropy curves.
- **Raise `warmup_steps`** if your first several steps have unusually low entropy (e.g., from a constrained decoding setup).

## Recommended action when it fires

> Reduce learning rate by 5x or increase KL penalty coefficient. Consider increasing entropy bonus if available.

The most common fix: the LR is too high and the policy is sliding into a degenerate solution. A 5x LR cut almost always rescues the run. If your framework supports an entropy bonus (e.g., `--entropy_coeff` in TRL), nudging it up is a softer fix.
