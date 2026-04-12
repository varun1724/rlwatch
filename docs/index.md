# rlwatch

**Catch broken RL training runs before they waste your GPU budget.**

If you train language models with GRPO or PPO, you already know the pain: you kick off a run on 8 H100s, go to sleep, and wake up to find the policy collapsed into repeating the same token 12 hours ago. Nobody saw it. Nothing paged. The run just quietly rotted.

rlwatch is a tiny Python library that watches your training metrics in real time and pings you on Slack, Discord, email, or any HTTP endpoint the moment things start going wrong — *before* the run is ruined.

---

## The 30-second pitch

1. `pip install rlwatch`
2. Add two lines to your training script:
   ```python
   import rlwatch
   rlwatch.attach()
   ```
3. Keep training. If something breaks, you get a message like:

   > 🚨 **rlwatch CRITICAL: entropy_collapse**
   > Run: `grpo_v3_exp12` | Step: 340
   > Policy entropy dropped from 2.8 to 0.4 over 50 steps (threshold: 1.0).
   > **Recommended action:** reduce learning rate by 5× or increase KL penalty.

You open the dashboard, confirm the curve, kill the run, fix the config, and you've just saved ~30 GPU-hours.

---

## What's in these docs

| Page | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, two-line attach, see your first alert fire |
| [Detectors](detectors/index.md) | Every detector — what it watches for, default thresholds, when to tune them |
| [Configuration](configuration.md) | YAML schema, environment variables, resolution order |
| [CLI](cli.md) | `rlwatch init / runs / diagnose / dashboard` |
| [Alerts](alerts/index.md) | Slack, email, Discord, generic webhook — setup and payload formats |
| [TRL + GRPO end-to-end tutorial](tutorials/trl-grpo-end-to-end.md) | Catch a real entropy collapse on a real GPT-2 + TRL GRPO run in under 5 minutes on a laptop CPU |
| [FAQ](faq.md) | Does it work offline? Does it upload anything? Why no telemetry? |
| [Contributing](contributing.md) | The development workflow and the testing harness contract |

---

## Project direction

rlwatch is heading toward a hosted, team-oriented product. The local-first open-source library will stay free and useful on its own. See [`ROADMAP.md`](https://github.com/varun1724/rlwatch/blob/main/ROADMAP.md) on GitHub for the full plan.
