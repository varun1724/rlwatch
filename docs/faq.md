# FAQ

## Does rlwatch upload my training data anywhere?

**No.** Cardinal rule #4 of the project: zero network calls in core. Slack, email, Discord, and the generic webhook are opt-in via config. The default install works offline, in air-gapped GPU clusters, with zero telemetry. CI enforces this with a forbidden-pattern grep that fails the build if any network library appears outside `src/rlwatch/alerts.py`.

## Where is my data stored?

In a single SQLite file at `./rlwatch_logs/metrics.db` (configurable via `storage.log_dir`). Three tables: `runs`, `metrics`, `alerts`. WAL mode is on so the training process can write while the dashboard reads concurrently. Copy the `.db` file and you've copied the entire history of every run.

## Will rlwatch slow down my training?

It shouldn't. The hot path is asserted in CI to be **under 1 ms** per `check_step` call (currently ~35µs on a 2020-era laptop), and the full `log_step` pipeline (detect + write to SQLite) is **under 3 ms** (currently ~70µs). Compared to a single forward pass of a 7B model, rlwatch is in the noise.

If you're running an inner loop where 70µs matters, consider only logging every N steps.

## What happens if rlwatch itself crashes?

Nothing visible to your training loop. Cardinal rule #2: rlwatch never raises into the user's code. Detector exceptions are caught at the boundary, alert delivery exceptions are caught in the daemon thread, and everything is logged at error level. Worst case: the rlwatch monitor stops doing its job, but training keeps running.

## Does it support PyTorch 1.x / TensorFlow / JAX?

rlwatch is framework-agnostic. The TRL integration is the deepest one (auto-registered `TrainerCallback`), but for any other framework you call `monitor.log_step(step, entropy=..., kl_divergence=..., ...)` yourself in your training loop. Every metric is optional — pass whatever your framework exposes.

## Why no PagerDuty channel yet?

PagerDuty is on the v0.4 roadmap. The reasoning: PagerDuty is an enterprise channel, and enterprise users have specific routing/escalation requirements that need a real customer to design against. v0.3 ships Slack, Discord, email, and the generic webhook — all four cover the vast majority of small-team and OSS use cases. The generic webhook can target PagerDuty's events API today if you write the template yourself.

## Why a simplified Hartigan dip test instead of the real one?

The real Hartigan dip test is provided by the [`diptest`](https://pypi.org/project/diptest/) PyPI package, which is a C extension. Adding a C extension as a runtime dependency violates the "default install works anywhere" goal — wheels aren't guaranteed for every architecture. The current home-rolled implementation is documented as approximate and is good enough for the variance-explosion check (which is the primary signal). Real `diptest` integration is on the v0.4 roadmap.

## Can I run multiple monitors in one process?

Not currently. `rlwatch.attach()` sets a global singleton. If you need multiple concurrent monitors (e.g., monitoring two trainers in the same process), you can construct `RLWatch` instances directly:

```python
from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch

cfg_a = RLWatchConfig(run_id="trainer_a")
cfg_b = RLWatchConfig(run_id="trainer_b")
mon_a = RLWatch(cfg_a)
mon_b = RLWatch(cfg_b)
mon_a.start(); mon_b.start()
```

But this is a power-user path and not officially supported. The two-line attach API assumes one monitor per process.

## Why no support for distributed/multi-node metrics?

rlwatch monitors a single process. In a distributed training setup, you typically attach it to the rank-0 process, which reflects the global training state for the metrics rlwatch cares about (entropy, KL, reward, advantage, loss, gradient norm). Cross-node metric aggregation is a separate problem that needs a network metric collection layer — it's explicitly out of scope for the OSS library and belongs to the future hosted service (Phase 2 of the roadmap).

## How is rlwatch different from W&B?

W&B logs metrics. rlwatch *watches* them and pings you when they're broken. Different jobs:

- **W&B is for record-keeping and post-hoc analysis** — what did my run look like? did this experiment beat the baseline? rlwatch doesn't try to compete with W&B on metric storage or visualization.
- **rlwatch is for active monitoring** — is my run going off the rails *right now*, before I've wasted 12 GPU-hours? W&B's dashboard requires you to look at it; rlwatch sends you a Slack message.

You can run both together. rlwatch's SQLite store is independent of any external service.

## How do I contribute?

Read [contributing](contributing.md), [CLAUDE.md](https://github.com/varun1724/rlwatch/blob/main/CLAUDE.md) for the development conventions, and [TESTING.md](https://github.com/varun1724/rlwatch/blob/main/TESTING.md) for the test harness contract. Every change ships with tests across the relevant tier(s).
