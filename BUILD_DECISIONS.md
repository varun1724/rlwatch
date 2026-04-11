# Build Decisions

## Tech Stack Choices

### Python + pip (no compiled extensions)
rlwatch is a pure Python library with no C extensions or compiled dependencies beyond scipy. This makes it installable via `pip install` on any system with Python 3.10+ without needing a GPU, CUDA, or build tools. The target audience (ML researchers) already has Python environments; adding a compiled dependency would create unnecessary friction.

### SQLite for metric storage
Chose SQLite over alternatives (InfluxDB, TimescaleDB, file-based JSON):
- **Zero setup** — sqlite3 is in the Python standard library. No database server to install.
- **WAL mode** provides fast concurrent reads/writes from the training process and dashboard simultaneously.
- **Portable** — the entire history of a run is a single `.db` file that can be copied, shared, or attached to a bug report.
- Tradeoff: not suitable for distributed multi-node metric aggregation. This is explicitly out of scope for MVP.

### Streamlit for dashboard
Chose Streamlit over Grafana, custom React, or Jupyter:
- **One command** — `rlwatch dashboard` launches a fully interactive dashboard. No external service to configure.
- **Python-native** — the entire stack stays in the ML researcher's Python environment.
- **Plotly integration** — interactive charts with zoom, pan, and hover without custom JS.
- Tradeoff: Streamlit adds ~100MB of dependencies. Acceptable for a monitoring tool that runs alongside multi-GB ML frameworks.

### Slack SDK + smtplib for alerts
- Slack: Used the official `slack_sdk` webhook client rather than raw HTTP. More reliable error handling and follows Slack's rate limiting guidelines.
- Email: Used stdlib `smtplib` to avoid adding another dependency. SMTP is universal and works with Gmail, AWS SES, SendGrid, etc.
- Alerts are sent in background threads to avoid blocking the training loop.

### scipy for Hartigan's dip test
The dip test for detecting bimodal reward distributions requires statistical routines. scipy is already present in nearly every ML environment, so it adds no new dependency burden.

### Rich for CLI output
Rich provides colored, formatted terminal output that makes alert messages and diagnostic reports readable at a glance. It's lightweight (~3MB) and has no native dependencies.

### Click for CLI
Click provides a clean, composable CLI framework with automatic help generation. Preferred over argparse for developer experience.

## Tradeoffs Made

### Detection heuristics are statistical, not ML-based
The four detectors use simple statistical rules (rolling means, z-scores, variance ratios, dip tests) rather than learned anomaly detection models. This is intentional:
- **Interpretable** — researchers can understand exactly why an alert fired.
- **No training data needed** — works on the first run without a historical dataset.
- **Low overhead** — no GPU memory consumed by the monitoring system itself.
- Tradeoff: may produce false positives in unusual but healthy training dynamics. Configurable thresholds and warmup periods mitigate this.

### Single-process monitoring only
The MVP monitors one training process. It does not aggregate metrics across distributed workers. This keeps the implementation simple and avoids the complexity of network-based metric collection. The researcher still gets value because they typically monitor the rank-0 process which reflects the global training state.

### Framework integration depth varies
TRL integration is the deepest (automatic TrainerCallback registration). veRL and OpenRLHF integration is more manual (user calls `log_step()`). This reflects TRL's mature callback API vs. veRL/OpenRLHF's younger plugin systems. As those frameworks stabilize, deeper integrations can be added.

### Rolling-window baselines can drift with the signal
Both `KLExplosionDetector` and `AdvantageVarianceDetector` compute their reference statistics from a rolling window (default 100 steps). This catches sharp spikes reliably but will *silently miss slow, sustained drift*: if KL or advantage std creeps up over several hundred steps, the rolling baseline creeps with it and the z-score never crosses the threshold. `RewardHackingDetector` avoids this by freezing its baseline variance from the first 20 post-warmup steps.

**v0.2.0 update:** both detectors now expose `baseline_mode: "rolling" | "frozen"`. The default stays `"rolling"` so existing users see no behavior change, but the option is there for runs where slow drift is the actual failure. The new `GradientNormSpikeDetector` defaults to `"frozen"` because gradient norms drift slowly on healthy runs and the rolling baseline silently follows them. A future v0.3 may flip the KL/advantage defaults too — that would be a real behavior change and ships behind a CHANGELOG entry, not silently.

### TRL auto-attach no longer scans gc.get_objects()
The original `_attach_trl()` walked `gc.get_objects()` looking for a live `Trainer` instance to bind the callback to. This was slow on large processes, fragile (depended on `attach()` being called *after* `Trainer(...)`), and untestable in any reasonable way. As of v0.2.0:

- `attach(trainer=...)` is the canonical path — pass the Trainer in and we register the callback directly.
- `monitor.attach_to_trainer(trainer)` is the helper for the case where attach() must run before the Trainer is constructed.
- The fallback is a clear log message telling the user which call to make. We do not scan the object graph anymore.

### Alert cooldown is per-detector, not global
If entropy collapses AND KL explodes simultaneously, you get two alerts. This is intentional — these are distinct failure modes requiring different remediation. The cooldown prevents repeated alerts for the same detector.

## What Was Intentionally NOT Built (Out of Scope)

1. **Multi-cloud metric aggregation across distributed runs on different cloud providers** — MVP monitors a single training process only. Cross-node aggregation requires a network metric collection layer that would double the implementation complexity.

2. **Reward model quality auditing and verifier calibration analysis** — This is a distinct product requiring separate framework integration, reward model access, and different statistical methods. It's a potential future product, not a feature of training instability detection.

3. **Integration with proprietary or closed-source RL frameworks** — MVP supports veRL, OpenRLHF, and HuggingFace TRL only. These are the three most-used open-source RL post-training frameworks. Proprietary framework support would require NDAs and custom development.

## How to Test the Risky Assumption

> **Risky assumption:** Teams experiencing GRPO instability will modify their training workflow to install a monitoring library rather than responding to instability empirically by shortening experiments or lowering learning rates.

### Testing approach:

1. **Run the simulation demo** (`python examples/simulate_grpo_run.py`) and observe whether the alert at step ~340 provides actionable information that would have been missed by empirical observation alone (e.g., the specific metric values, recommended actions).

2. **Measure time-to-detection**: In the simulation, entropy begins collapsing at step 280 but a human reviewing W&B charts might not notice until step 400+. rlwatch alerts at step ~340. This 60+ step gap (potentially hours of GPU time on a real run) is the value proposition.

3. **Track adoption signals**:
   - Count `pip install rlwatch` downloads from PyPI
   - Monitor GitHub stars and issues
   - Track how many users complete the full journey: install -> attach -> receive alert -> take action
   - The SQLite database stores all run data locally; a future opt-in telemetry feature could measure whether users who receive alerts actually stop and restart runs vs. ignoring the alert

4. **Customer development**: Post on veRL GitHub Issue #1677 and OpenRLHF Discord asking: "Would you install a 2-line library that sends you a Slack alert when entropy collapses during your GRPO run?" If fewer than 3 out of 10 contacted engineers express interest, the assumption is falsified.

5. **A/B comparison**: Ask beta users to run two similar training experiments — one with rlwatch, one without — and measure whether rlwatch-monitored runs result in fewer wasted GPU-hours from undetected instabilities.
