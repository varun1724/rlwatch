# Playbook: what to do when rlwatch fires

You got an alert. This page tells you what it means and what to do about it, in priority order, for every detector. Bookmark this page.

---

## Entropy collapse

**The alert says:** *"Policy entropy dropped from X to Y over N consecutive steps."*

**What's happening:** Your model stopped exploring. It's producing near-deterministic outputs — the same tokens, the same patterns. Reward signal will go flat shortly after if it hasn't already.

**Fix priority:**

1. **Reduce learning rate by 5–10x.** This is the fix 90% of the time. The policy is sliding into a degenerate solution because the LR is too aggressive.
2. **Increase KL penalty coefficient.** If your framework supports a KL penalty term (most GRPO/PPO implementations do), bump it 2–3x. This pulls the policy back toward the reference model.
3. **Add or increase entropy bonus.** Some frameworks (TRL, OpenRLHF) support an explicit entropy regularization term. Nudging it up encourages exploration.
4. **Revert to a checkpoint from before the collapse.** If entropy is already < 0.5, the current policy may be unrecoverable. Roll back to the last checkpoint where entropy was healthy (> 1.5) and apply fix #1 before resuming.

**Common mistake:** ignoring the warning tier. If you see a warning at step 200 and do nothing, the critical at step 250 is much harder to recover from. Act on warnings.

---

## KL divergence explosion

**The alert says:** *"KL divergence = X, Y σ above rolling mean."*

**What's happening:** Your policy has drifted too far from the reference model. This is usually the prelude to reward hacking — the model has discovered an out-of-distribution behavior that the reward model rates highly but is actually nonsense.

**Fix priority:**

1. **Increase KL penalty coefficient immediately.** This is the direct control for KL. Double or triple it.
2. **Reduce learning rate.** Slower updates = less drift per step.
3. **Revert to a checkpoint.** If KL has critically exploded (> 4.5σ), the run is usually unrecoverable from where it is. Roll back and tighten the KL coefficient before resuming.
4. **Check your reward model.** If this happens early in training, the reward model may have a distributional blind spot that the policy found immediately. Inspect the completions at the step where KL spiked.

**Common mistake:** only looking at the KL number in isolation. Check entropy too — if entropy is also dropping, you have a compounding problem (the model is both collapsing AND drifting). Fix entropy first, then KL.

---

## Reward hacking (variance explosion)

**The alert says:** *"Reward variance exploded Nx above baseline."*

**What's happening:** Per-sample reward variance suddenly spiked. This means some completions are getting very different rewards than others — a signature of exploitation. Some samples found the hack; others didn't.

**Fix priority:**

1. **Inspect the completions.** This is the one alert where "look at the data" is step 1, not step 3. Open the actual model outputs at the alert step and look for patterns: repeated tokens, format gaming, length exploitation, keyword stuffing.
2. **Cap rewards.** Clamp the reward to a reasonable range (e.g., [-2, 2]) to limit the gradient signal from outlier samples.
3. **Add reward model regularization.** If you're using a learned reward model, add weight decay or dropout to the RM. A more uncertain RM is harder to exploit.
4. **Consider reward model ensembling.** Use 2–3 reward models and take the minimum score. The hack that fools one RM is less likely to fool all three.

**Common mistake:** reducing the LR. LR reduction fixes entropy collapse and KL explosion, but it doesn't fix reward hacking — a slower-moving policy will still converge on the exploit, just more slowly.

---

## Reward hacking (bimodal distribution)

**The alert says:** *"Bimodal reward distribution detected — Hartigan dip test p-value = X."*

**What's happening:** The per-sample rewards have split into two clusters. One cluster is "normal" rewards; the other is the hack. This is a more subtle signal than variance explosion — it can appear even when mean and variance look okay.

**Fix priority:** Same as variance explosion above. The bimodal alert is a different detection path for the same failure mode.

**Note:** Install `pip install "rlwatch[monitoring]"` for the real Hartigan dip test. The default simplified implementation is conservative and may miss borderline cases.

---

## Reward mean drift

**The alert says:** *"Reward mean has been drifting [up/down] monotonically for N steps."*

**What's happening:** The reward mean is moving in one direction for an extended period without any oscillation. This *might* be legitimate learning (reward should increase during healthy training), or it might be slow, sustained reward hacking where the variance doesn't spike but the distribution is shifting.

**Fix priority:**

1. **Inspect the completions.** Are the model's outputs actually getting better (more correct, more coherent), or are they getting weirder?
2. **If the outputs look good:** this is legitimate learning. Consider raising `warmup_steps` or `consecutive_steps` in the config to reduce future false positives.
3. **If the outputs look suspicious:** same fixes as reward hacking above. The model found a slow exploit.

**Common mistake:** panicking. This is a warning-only alert for a reason — monotone reward drift is genuinely ambiguous. Investigate, don't stop.

---

## Advantage variance spike

**The alert says:** *"Advantage std = X, Nx above rolling baseline."*

**What's happening:** The value function estimates just became unstable. The value head is no longer giving a useful baseline for the policy gradient, so updates have wildly inconsistent signs.

**Fix priority:**

1. **Enable advantage normalization.** If your framework has a `normalize_advantages` flag, turn it on. This is the single most common fix.
2. **Increase batch size.** Larger batches smooth out per-step advantage variance.
3. **Reduce learning rate.** Slower updates give the value head time to catch up.
4. **Add more capacity to the value head.** If the value function is too small relative to the policy, it can't keep up. This is a model architecture change, not a hyperparameter tweak.

**Common mistake:** ignoring this alert because "advantages are noisy anyway." Large spikes in advantage std often precede gradient explosions or entropy collapse by 50–100 steps. This is an early warning.

---

## Loss NaN / Inf

**The alert says:** *"Loss is non-finite (NaN / +Inf / -Inf)."*

**What's happening:** The optimizer has blown up. **Every further gradient update corrupts the policy.** This is not recoverable from the current state.

**Fix priority:**

1. **Stop the run immediately.** Not "after this epoch" — now. Every additional step makes the corruption worse.
2. **Revert to the last good checkpoint.** The last checkpoint before this step is your recovery point.
3. **Check if there was a gradient norm spike first.** The [gradient norm spike detector](#gradient-norm-spike) usually fires 10–50 steps before loss goes NaN. If you see both alerts, the fix is gradient clipping.
4. **Add gradient clipping** (`max_grad_norm=1.0`). If it wasn't set before, set it. If it was set, tighten it.
5. **Reduce learning rate** by 5–10x before resuming from the checkpoint.
6. **Check for numerical instability in the loss computation.** Common culprits: `log(0)` from a poorly-initialized policy, divide-by-zero in advantage normalization with a constant batch, mixed-precision overflow in fp16.

**Common mistake:** trying to "resume" training from a NaN state. You can't. The weights are corrupted. Roll back.

---

## Gradient norm spike

**The alert says:** *"Gradient norm = X, Yσ above baseline."*

**What's happening:** Gradients just exploded. This is almost always the immediate precursor to loss NaN — you have 10–50 steps to act before the optimizer blows up.

**Fix priority:**

1. **Add or tighten gradient clipping.** `max_grad_norm=1.0` is the standard default. If it's already set and you're still spiking, lower it to 0.5 or 0.1.
2. **Reduce learning rate.** The combination of high LR + large gradients is what produces the explosion.
3. **Check for exploding activations.** If a specific layer is producing huge gradients, it may need re-initialization or a lower per-layer LR.
4. **If using mixed precision (fp16):** switch to bf16 if your hardware supports it, or add a loss scaler. fp16 has a narrower dynamic range and is more prone to overflow.

**Common mistake:** waiting to see if the spike is transient. Gradient norm spikes are rarely transient — they're the leading edge of an exponential divergence. Act on the first spike.

---

## General principles

1. **Act on warnings, not just criticals.** A warning at step 200 is fixable. A critical at step 250 may not be.
2. **Check multiple detectors together.** Entropy collapse + KL explosion = compounding problem. Fix entropy first. Gradient spike + loss NaN = the spike caused the NaN. Fix the spike.
3. **The LR knob fixes most things.** When in doubt, cut the LR by 5x and restart from a checkpoint. It's the most universal fix.
4. **Always inspect the completions.** Numbers tell you *what* is happening; completions tell you *why*.
5. **Keep your checkpoints.** rlwatch can tell you when things went wrong, but you need checkpoints to actually recover. Save every N steps, not just at the end.
