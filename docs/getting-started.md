# Getting started

This page takes you from zero to "I just watched rlwatch fire its first alert" in about three minutes.

## Install

```bash
pip install rlwatch
```

That gives you the core library — six detectors, the CLI, console + Slack + email + Discord + webhook alerts, and the SQLite metric store. About 20 MB of dependencies.

For the Streamlit dashboard:
```bash
pip install "rlwatch[dashboard]"
```

For the HuggingFace TRL deep integration:
```bash
pip install "rlwatch[trl]"
```

You can combine extras: `pip install "rlwatch[dashboard,trl]"`.

## Smoke test: see rlwatch fire in 30 seconds

The repo ships with a simulation that deliberately collapses entropy around step 280. Run it without any framework — it generates synthetic metrics and feeds them to rlwatch:

```bash
git clone https://github.com/varun1724/rlwatch
cd rlwatch
python examples/simulate_grpo_run.py
```

You'll see something like:

```
Step    0 | entropy=2.776 kl=0.0102 reward=-1.498 adv_std=0.991
Step   50 | entropy=2.762 kl=0.0151 reward=-1.298 adv_std=1.012
...
Step  300 | entropy=0.439 kl=0.1601 reward=0.012 adv_std=1.402
╭───────────────────── rlwatch CRITICAL: entropy_collapse ───────────────────────╮
│ Step 320 | Run: grpo_v3_exp12                                                  │
│                                                                                │
│ Entropy collapse detected — policy entropy dropped from 2.78 to 0.21 over 50   │
│ consecutive steps (threshold: 1.0).                                            │
│                                                                                │
│ Recommendation: Reduce learning rate by 5x or increase KL penalty coefficient. │
╰────────────────────────────────────────────────────────────────────────────────╯
```

Then look at the diagnosis report:
```bash
rlwatch diagnose
```

That's the full demo. You just watched the entire pipeline — detector → alert → console panel → SQLite store → CLI report — in one shot.

## Two-line attach in your real training loop

```python
import rlwatch
rlwatch.attach()
```

That's it. rlwatch will detect HuggingFace TRL automatically and register a `TrainerCallback` that reads entropy, KL, reward, advantage std, loss, and gradient norm from the trainer's `on_log` callback. If you're using veRL, OpenRLHF, or a custom training loop, see the manual mode below.

## Two-line attach with TRL (recommended)

```python
import rlwatch
from trl import GRPOTrainer, GRPOConfig

# Pass the trainer in directly when you construct rlwatch:
trainer = GRPOTrainer(model=..., args=GRPOConfig(...), ...)
monitor = rlwatch.attach(trainer=trainer)

trainer.train()
```

`attach(trainer=trainer)` is the recommended TRL path — it registers the rlwatch callback directly. If you've already called `rlwatch.attach()` before constructing the trainer, use `monitor.attach_to_trainer(trainer)` after.

For a full end-to-end walkthrough that fires a real `entropy_collapse` alert on real TRL GRPO training in ~5 minutes on a CPU, see the [TRL + GRPO tutorial](tutorials/trl-grpo-end-to-end.md).

## veRL integration (auto-detected)

```python
import rlwatch
rlwatch.attach()  # auto-detects veRL and registers a tracking backend
```

rlwatch registers a custom tracking backend with veRL's `Tracking` class. Metrics are mapped automatically: `actor/entropy` → `entropy`, `rewards/mean` → `reward_mean`, etc. If auto-registration doesn't work with your veRL version, see the fallback instructions in the log output.

## Manual mode (OpenRLHF / custom training loops)

```python
import rlwatch

monitor = rlwatch.attach(framework="manual", run_id="my_run")

for step in range(num_steps):
    # ... your training step ...

    monitor.log_step(
        step,
        entropy=policy_entropy,
        kl_divergence=kl,
        reward_mean=rewards.mean(),
        reward_std=rewards.std(),
        advantage_std=advantages.std(),
        loss=loss.item(),
        grad_norm=grad_norm.item(),
    )
```

Every metric is optional — pass whichever ones your framework exposes. Detectors that don't get their input simply don't fire.

## Setting up alerts

By default, alerts go to the console (always). To get pinged elsewhere:

- **Slack:** export `RLWATCH_SLACK_WEBHOOK_URL` or set it in `rlwatch.yaml`. See [alerts/slack](alerts/slack.md).
- **Discord:** export `RLWATCH_DISCORD_WEBHOOK_URL` or set it in `rlwatch.yaml`. See [alerts/discord](alerts/discord.md).
- **Email:** configure `alerts.email` in `rlwatch.yaml`. See [alerts/email](alerts/email.md).
- **Generic webhook:** export `RLWATCH_WEBHOOK_URL` or set `alerts.webhook.url`. See [alerts/webhook](alerts/webhook.md).

## What to read next

- The [detectors overview](detectors/index.md) for what each detector watches and how to tune it.
- The [configuration guide](configuration.md) for the YAML schema, env vars, and resolution order.
- The [TRL + GRPO end-to-end tutorial](tutorials/trl-grpo-end-to-end.md) for a real, deterministic, CPU-friendly example.
