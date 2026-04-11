# rlwatch

**Catch broken RL training runs before they waste your GPU budget.**

If you train language models with GRPO or PPO, you already know the pain: you kick off a run on 8 H100s, go to sleep, and wake up to find the policy collapsed into repeating the same token 12 hours ago. Nobody saw it. Nothing paged. The run just quietly rotted.

rlwatch is a tiny Python library that watches your training metrics in real time and pings you on Slack (or email) the moment things start going wrong — *before* the run is ruined.

---

## The 30-second pitch

1. `pip install rlwatch`
2. Add two lines to your training script:
   ```python
   import rlwatch
   rlwatch.attach()
   ```
3. Keep training. If something breaks, you get a Slack message like:

   > 🚨 **rlwatch CRITICAL: entropy_collapse**
   > Run: `grpo_v3_exp12` | Step: 340
   > Policy entropy dropped from 2.8 to 0.4 over 50 steps (threshold: 1.0).
   > **Recommended action:** reduce learning rate by 5× or increase KL penalty.

You open the dashboard, confirm the curve, kill the run, fix the config, and you've just saved ~30 GPU-hours.

---

## What it watches for

These are the most common ways GRPO/PPO runs go sideways. rlwatch runs a dedicated detector for each one on every training step.

| Detector | In plain English | Default trip-wire |
|---|---|---|
| **Entropy collapse** | The model stopped exploring — it's now just repeating itself. | Entropy < 1.0 for 50 steps in a row |
| **KL divergence explosion** | The policy is running away from the reference model (usually the prelude to reward hacking). | KL > 3σ above the rolling mean |
| **Reward hacking proxy** | Rewards suddenly got weird — either way more variance than before, or split into two clusters (some samples hacked, some didn't). | Variance > 3× baseline, **or** Hartigan dip test p < 0.05 |
| **Advantage variance spike** | The value function estimates just became unstable. | Advantage std > 3× rolling baseline |
| **Loss NaN / Inf** | The optimizer has blown up; any further updates corrupt the policy. | Loss is non-finite (one step is enough) |
| **Gradient norm spike** | Gradients exploded — usually the precursor to a loss NaN. | Grad norm > 3σ above frozen baseline |

Every detector has two severity levels (**warning** and **critical**), a configurable warmup period so it doesn't fire at step 3, and a cooldown so you don't get spammed.

---

## Quick start

```bash
pip install -e .
```

### Option A: two-line attach (easiest)

```python
import rlwatch
rlwatch.attach()   # auto-detects TRL / veRL / OpenRLHF

# ...your normal training code...
```

rlwatch auto-detects HuggingFace TRL and registers a `TrainerCallback`. For veRL, OpenRLHF, or any custom loop, use Option B.

### Option B: manual metric logging

```python
import rlwatch

monitor = rlwatch.attach(framework="manual", run_id="grpo_v3_exp12")

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
    )
```

### See it fire

The repo ships with a simulated GRPO run that deliberately collapses entropy:

```bash
python examples/simulate_grpo_run.py   # run the simulation
rlwatch diagnose                         # get a retrospective report
rlwatch dashboard                        # open the live dashboard at localhost:8501
```

---

## Setting up alerts

### Slack
```bash
export RLWATCH_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```
Or put it in `rlwatch.yaml`:
```yaml
alerts:
  slack:
    webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### Email
```yaml
alerts:
  email:
    smtp_host: smtp.gmail.com
    to_addrs:
      - you@yourcompany.com
```

### Console
Always on. Rich-formatted panels show up in stderr regardless of other channels.

---

## Configuration

Generate a starter config:
```bash
rlwatch init
```
This writes `rlwatch.yaml` with every threshold at its default. Tweak to taste.

Resolution order: **defaults → YAML file → environment variables → `attach()` kwargs**. Later values win.

---

## CLI reference

| Command | What it does |
|---|---|
| `rlwatch init` | Write a starter `rlwatch.yaml` |
| `rlwatch runs` | List every monitored run in the local SQLite store |
| `rlwatch diagnose [--run-id ID]` | Print a retrospective report on a completed run |
| `rlwatch dashboard` | Launch the Streamlit dashboard at `localhost:8501` |

---

## How it stores data

Everything lives in a single SQLite file at `./rlwatch_logs/metrics.db`. Three tables: `runs`, `metrics`, `alerts`. WAL mode is on so the training loop writes and the dashboard reads concurrently without locking. Copy that `.db` file and you've copied the entire history of every run.

---

## Supported frameworks

- **HuggingFace TRL** — auto-detected, registers a `TrainerCallback` that reads entropy, KL, reward, advantage, loss from `on_log`.
- **veRL** — `framework="manual"` + `monitor.log_step()`. Deep integration on the roadmap.
- **OpenRLHF** — `framework="manual"` + `monitor.log_step()`. Deep integration on the roadmap.
- **Anything else** — same as above.

---

## Docker

```bash
docker build -t rlwatch .
docker run -p 8501:8501 rlwatch
```

---

## Project direction

rlwatch is heading toward a hosted, team-oriented product. The local-first open-source library will stay free and useful on its own. See `ROADMAP.md` for the full plan.

## Contributing & testing

rlwatch is a monitoring library — if it has bugs, it costs someone a GPU
budget. The test harness is the most load-bearing part of the repo.

```bash
pip install -e ".[dev]"
pytest -v                                    # all five tiers
pytest --cov=rlwatch --cov-fail-under=90    # coverage gate (must pass to merge)
```

The suite is organized into five tiers (unit / property / simulation /
integration / performance). See **[`TESTING.md`](TESTING.md)** for the
practical "how to run, write, and debug tests" guide and **[`CLAUDE.md`](CLAUDE.md)**
for the authoritative contract every PR has to meet.

## License

MIT
