# TRL + GRPO end-to-end (CPU, ~5 minutes)

This tutorial runs a real GPT-2 model through TRL's `GRPOTrainer` with rlwatch attached. The learning rate is **deliberately misconfigured** to be an order of magnitude too high, which causes a real entropy collapse within the first ~150 steps. rlwatch catches it and fires a critical alert.

It runs on CPU. No GPU. No API keys. ~5 minutes start to finish.

## What you'll see

By the end of this tutorial you'll have:

- A working `pip install "rlwatch[trl,tutorial]"` environment
- A real (deliberately broken) TRL GRPO training run
- A real `entropy_collapse` alert from the real detector
- A `rlwatch diagnose` report showing the collapse
- A clear sense of how rlwatch fits into a real training workflow

## Install

```bash
pip install "rlwatch[trl,tutorial]"
```

The `[tutorial]` extra pins exact known-working versions of `trl`, `transformers`, `torch`, and `datasets`. The tutorial CI cron runs against these pinned versions every month, so if a future TRL release silently breaks the path, we catch it before you do.

## Run it

```bash
python examples/trl_grpo_tutorial.py
```

(Or download just the tutorial script: [`examples/trl_grpo_tutorial.py`](https://github.com/varun1724/rlwatch/blob/main/examples/trl_grpo_tutorial.py).)

## What the script does

1. **Sets three random seeds.** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`. The tutorial is deterministic across runs on the same machine.

2. **Loads GPT-2 (124M params).** The smallest causal LM that ships with `transformers` and the smallest practical model for CPU GRPO.

3. **Builds a 20-prompt synthetic dataset.** The "task" is: respond with the word "YES". Reward function returns `1.0` if the completion starts with YES (case-insensitive) and `0.0` otherwise. Trivially easy for a real model — but with a deliberately too-high LR, GRPO collapses entropy long before it learns to do it reliably.

4. **Builds a `GRPOConfig` with `learning_rate=1e-3`.** That's an order of magnitude above what's safe for a 124M model. A healthy LR for this setup is around `5e-6`. The high LR is the bug we want rlwatch to catch.

5. **Calls `rlwatch.attach(trainer=trainer)`.** Two-line attach. The TRL deep-integration callback is registered directly on the trainer.

6. **Calls `trainer.train()`.** TRL runs ~200 GRPO steps. rlwatch reads metrics out of `on_log` and runs the detector suite on each one.

## Expected output

You'll see TRL's training output interleaved with rlwatch's console panels. Around step 150, you'll see something like this fly by:

```
╭───────────────────── rlwatch CRITICAL: entropy_collapse ──────────────────────╮
│ Step 150 | Run: trl_grpo_tutorial                                              │
│                                                                                │
│ Entropy collapse detected — policy entropy dropped from 4.21 to 0.34 over 15  │
│ consecutive steps (threshold: 1.0).                                            │
│                                                                                │
│ Recommendation: Reduce learning rate by 5x or increase KL penalty coefficient. │
╰────────────────────────────────────────────────────────────────────────────────╯
```

When training finishes, the script prints a summary:

```
================================================================
Tutorial complete. 1 alert(s) fired.
================================================================

✅ rlwatch caught the entropy collapse caused by the
   deliberately-too-high learning rate.

Next steps:
  1. Run `rlwatch diagnose` to see the full report.
  2. Re-run with `learning_rate=5e-6` and watch the alert NOT fire.
  3. Read the tutorial walkthrough at
     https://varun1724.github.io/rlwatch/tutorials/trl-grpo-end-to-end/
```

## Reproducibility caveats

The tutorial pins TRL/transformers/torch versions in the `[tutorial]` extra. **Exact alert step numbers may drift by ±50 steps** as future TRL minor releases change details of the GRPO loop. The shape of the failure (entropy collapse) is robust; the precise step is not.

If your run finishes without the alert firing, check that you installed the extras (not just core rlwatch) and that your TRL version matches the pin.

## Fix the bug and re-run

To prove the alert fires for the *right* reason and not because rlwatch is overeager, edit the script:

```python
args = GRPOConfig(
    output_dir="./_rlwatch_tutorial_output",
    learning_rate=5e-6,                  # was 1e-3 — now safe
    # ... everything else the same ...
)
```

Re-run. You should see TRL training go to completion with **zero alerts** from rlwatch. That's the no-false-positive guarantee in action.

## What you just learned

- **Two-line attach is real.** `rlwatch.attach(trainer=trainer)` is the entire integration. No callback boilerplate, no metric mapping.
- **Detection is fast.** rlwatch caught the collapse 50 steps before TRL would have produced any visibly broken output.
- **Recommendations are actionable.** "Reduce learning rate by 5x" is exactly what you need to do.
- **It's deterministic.** Run it twice; the alert fires both times.

## Next steps

- Read the [detectors overview](../detectors/index.md) to see what other failure modes rlwatch catches.
- Read [configuration](../configuration.md) to learn how to tune thresholds for your model and dataset.
- Set up a real alert channel ([Slack](../alerts/slack.md), [Discord](../alerts/discord.md), or [generic webhook](../alerts/webhook.md)) so you don't have to be at your terminal when something breaks.
