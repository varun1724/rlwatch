# Loss NaN / Inf

**Watches:** the training loss.
**Failure mode:** the optimizer has blown up. Loss is `NaN`, `+Inf`, or `-Inf`. Once this happens, **every subsequent gradient update corrupts the policy further** — there's no recovery from where you are.

## How it fires

There is no rolling state and no warning tier. The instant `loss` is non-finite, the detector fires a **critical** alert. The alert's `metric_values["kind"]` is one of `"NaN"`, `"+Inf"`, or `"-Inf"` so you can tell which path blew up.

If you set `warmup_steps > 0`, the detector ignores non-finite losses inside the warmup window. The default is 0 — there's no good reason to wait on this one.

## Configuration

```yaml
loss_nan:
  enabled: true
  warmup_steps: 0
```

## Recommended action when it fires

> Stop the run immediately. Revert to the last good checkpoint. Reduce learning rate, clip gradients, and check for divide-by-zero or log-of-zero in the loss computation.

Things to check, in order:

1. **Was there a precursor gradient norm spike?** If so, the fix is gradient clipping (`max_grad_norm`). The [gradient norm spike detector](gradient-norm.md) usually fires first if both are enabled.
2. **Is the LR too high?** A 5–10× LR cut + restart from the last good checkpoint usually rescues the run.
3. **Numerical instability in the loss computation?** Common culprits: `log(0)` from a poorly-initialized policy, divide-by-zero in advantage normalization with a constant batch, mixed-precision overflow in fp16. Adding small epsilons or moving the relevant op to fp32 fixes most of these.

## Why no warning tier

There's no "loss is *almost* NaN." Either it's finite or it isn't. By the time you'd see a warning, the run is already broken.
