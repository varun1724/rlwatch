# CLAUDE.md

This file tells Claude Code (and any other AI agent) how to work productively in this repository. Humans should read it too — it encodes the conventions that keep rlwatch trustworthy.

---

## What this project is

rlwatch is a Python library that detects GRPO/PPO training instabilities in real time and alerts ML teams before they waste multi-day GPU runs. It will grow into a paid hosted service — see `ROADMAP.md`.

The top-level goal: **make rlwatch boring, reliable infrastructure that researchers forget is there until it saves them.**

---

## Repository layout

```
.
├── src/rlwatch/
│   ├── __init__.py       # Public API re-exports (attach, log_step, RLWatch, load_config)
│   ├── core.py           # RLWatch class, attach(), framework auto-detection
│   ├── detectors.py      # Four statistical detectors + DetectorSuite + Alert dataclass
│   ├── config.py         # Dataclass-based YAML/env/kwarg config loader
│   ├── storage.py        # MetricStore — SQLite WAL-mode persistence
│   ├── alerts.py         # AlertManager + Slack/email/console senders
│   ├── cli.py            # Click CLI: init / runs / diagnose / dashboard
│   └── dashboard.py      # Streamlit dashboard
├── tests/                # Pytest suite
├── examples/             # Runnable demos (including simulate_grpo_run.py)
├── rlwatch.yaml          # Example config
├── pyproject.toml
├── Dockerfile
├── README.md             # User-facing
├── ROADMAP.md            # Where we're going
└── CLAUDE.md             # This file — how we work
```

---

## Cardinal rules

1. **The two-line attach API is sacred.** `import rlwatch; rlwatch.attach()` must keep working. Any change that breaks it is a blocker.
2. **Never silently swallow exceptions in the monitoring path.** If rlwatch crashes, training shouldn't. Catch errors at the boundary and log them; never let them propagate into the user's training loop. But *do* log them — silent failure is worse than loud failure.
3. **Alerts must always include a recommended action.** A bare metric value is not an alert.
4. **Zero network calls in the core library.** Slack/email are opt-in via config. The default install must work offline, in an air-gapped GPU cluster, with no telemetry.
5. **Every detector change ships with tests.** No exceptions. See the testing section below.
6. **Don't break SQLite schema without a migration.** Users have `.db` files they care about.

---

## Development setup

```bash
pip install -e ".[dev]"
pip install -e ".[trl]"      # only needed for TRL integration tests
```

Run the test suite:
```bash
pytest -v
pytest --cov=rlwatch --cov-report=term-missing
```

Run a single test:
```bash
pytest tests/test_detectors.py::TestEntropyCollapseDetector::test_alerts_on_entropy_collapse -v
```

Run the simulation end-to-end to sanity check:
```bash
python examples/simulate_grpo_run.py
rlwatch diagnose
```

---

## Testing harness — the non-negotiable part

rlwatch is a monitoring library. If it is buggy, it causes silent, expensive failures. The testing harness is therefore the most important part of the repo. Follow this every time you add or change code.

### Test tiers

We use five tiers. Every change should extend the tier(s) it touches.

**Tier 1 — Unit tests** (`tests/test_*.py`)
- One test class per detector or component.
- Cover every state transition: warmup → healthy → warning → critical → cooldown → reset.
- Cover the `None`/missing-input branches (every detector must no-op on `None`).
- Cover the `enabled=False` branches.
- Fast: each test < 50ms. The whole unit suite should run in under 10 seconds.

**Tier 2 — Property-based tests** (`tests/property/` — add as needed)
- Use `hypothesis` to generate random but valid metric sequences.
- Assert invariants, not exact outputs:
  - No alert fires during warmup, ever.
  - Within a cooldown window, no two alerts from the same detector.
  - The total alert count never exceeds `max_alerts_per_run`.
  - A detector with constant healthy input never fires.
- These catch edge cases unit tests miss.

**Tier 3 — Simulation / golden tests** (`tests/simulations/`)
- Checked-in metric traces (JSON or Parquet) replayed through the full `DetectorSuite` + `MetricStore` + `AlertManager` pipeline.
- Each fixture has an expected alert set (detector, severity, approximate step).
- This is where we test the *whole pipeline behaves as advertised*.
- Use `examples/simulate_grpo_run.py` as the source for the canonical "entropy collapse" fixture.
- Add a new fixture every time we fix a real-world bug.

**Tier 4 — Integration tests** (`tests/integration/`)
- Real SQLite on a `tmp_path`.
- Real YAML parsing from a temp file.
- Real CLI invocation via `click.testing.CliRunner`.
- Real alert delivery with mocked transports:
  - Slack: `responses` library to stub the webhook URL
  - Email: `unittest.mock` patch of `smtplib.SMTP`, or `aiosmtpd` as a local SMTP catcher
- Real TRL `Trainer` on a tiny toy dataset for the TRL integration path. Gated on the `[trl]` extra so the default CI job doesn't need transformers.

**Tier 5 — Performance / regression tests** (`tests/perf/`)
- `pytest-benchmark` around `DetectorSuite.check_step()`.
- Hard assertion: < 1ms per step on a 2020-era laptop CPU. rlwatch must never be the bottleneck.
- Benchmark the SQLite write path: > 1000 steps/sec sustained.

### Coverage expectations

- Target ≥ 90% line coverage on `src/rlwatch/`.
- Enforced in CI via `pytest --cov=rlwatch --cov-fail-under=90`.
- **`dashboard.py` is excluded** from the gate (`omit` in
  `pyproject.toml`). Streamlit dashboards are not in-process testable
  without a headless browser; the dashboard is covered by manual smoke tests
  and the CLI smoke test in CI. Don't add it back to the coverage source
  unless you're also bringing a real way to exercise it.
- New code that drops coverage below the threshold does not merge.

See **`TESTING.md`** for the practical "how to run/write/debug tests" guide.
This file (CLAUDE.md) stays the authoritative spec for tier definitions and
the contract every PR must meet.

### What every PR must include

- [ ] Unit tests for any new branch in `detectors.py`, `core.py`, `config.py`, `storage.py`, or `alerts.py`.
- [ ] An integration test if the change touches CLI, alert delivery, storage schema, or framework integration.
- [ ] A simulation fixture if the change fixes or introduces a failure-mode detection.
- [ ] A benchmark if the change touches the hot path (`check_step`, `log_step`, `log_metrics`).
- [ ] An update to `CHANGELOG.md` (once it exists) and, if behavior changes, `README.md`.

### Anti-patterns to refuse

- ❌ Mocking `DetectorSuite` in a test for `RLWatch.log_step`. Test the real thing; it's fast.
- ❌ Asserting on exact alert *messages* (strings). Assert on `alert.detector`, `alert.severity`, and `alert.metric_values` instead — messages will change.
- ❌ Tests that touch the real filesystem outside `tmp_path`.
- ❌ Tests that hit the real network. Ever. Even "just for dev".
- ❌ Tests that depend on wall-clock time. Use injected clocks or counter-based warmup.
- ❌ Flaky tests "fixed" with retries. If it's flaky, it's broken — find the race.

---

## Conventions

### Code style
- Python 3.10+. We use modern type hints (`list[int]`, `X | None`).
- `from __future__ import annotations` at the top of every module.
- Dataclasses for all config; no dicts floating around the API surface.
- `Optional[T]` for nullable parameters in public APIs, `T | None` in internal code — this is a legacy inconsistency we'll clean up, but don't change public signatures without a reason.
- No single-letter variable names except in tight numerical loops.
- Docstrings on every public class and function. One-line summary, then details, then an example if non-obvious.

### Imports
- stdlib → third-party → first-party, separated by blank lines.
- Prefer `from rlwatch.x import Y` over `import rlwatch.x`.

### Logging
- Use `logging.getLogger("rlwatch")` or a child logger. Never `print()` except in CLI output paths that go through `rich.console`.
- `logger.info` for lifecycle (start/stop/attach).
- `logger.warning` for degraded but recoverable states (framework not found, falling back to manual).
- `logger.error` for real failures (alert delivery failed). Never raise into the training loop.

### Config
- Every configurable value lives in a dataclass in `config.py` with a sensible default.
- If you add a new config field, also add it to the YAML example in `README.md` and a test in `test_config.py`.
- Environment variable overrides go in the `env_map` in `config.py::load_config`.

### Detectors
- Every detector subclasses the conceptual contract: `check(step, *metrics) -> Alert | None`.
- Every detector supports `enabled`, `warmup_steps`, and at least one numeric threshold.
- Every detector with a rolling window uses `collections.deque(maxlen=...)`, never a list.
- Every detector's alert includes `detector`, `severity`, `step`, `message`, `metric_values` (dict), and `recommendation`.
- **`baseline_mode`** (KL, advantage, gradient norm): pick `"frozen"` for
  metrics that drift slowly on healthy runs (gradient norms, advantage std)
  and `"rolling"` for metrics where drift IS the failure (rare). The frozen
  baseline locks in once the rolling window first fills, mirroring
  `RewardHackingDetector`. KL and advantage default to `"rolling"` for
  backwards compatibility; gradient norm defaults to `"frozen"`.

### Alerts
- The cooldown / rate-limiting logic lives in `AlertManager`, not in individual detectors. Don't duplicate it.
- Delivery channels run in daemon threads so they never block `log_step`.
- Failures in delivery get logged, never raised.

### Storage
- WAL mode stays on. Do not change it.
- Schema changes require a migration. For now: bump a `schema_version` table, detect old versions on open, and run `ALTER TABLE` steps idempotently.
- Every `execute` is followed by a `commit` for durability. Don't batch unless you add an explicit flush-on-stop path.

---

## How Claude Code should operate in this repo

These are instructions specifically for AI agents working on rlwatch. They override default behavior when in conflict.

1. **Read before you write.** Always `Read` the file you're about to edit. Do not guess at line numbers or surrounding context.
2. **Run the tests after every non-trivial change.** `pytest -v`. If anything goes red, stop and investigate — don't pile on more changes.
3. **Prefer small, focused diffs.** One detector fix or one integration at a time. Do not opportunistically refactor while fixing a bug.
4. **Never delete a test to make CI pass.** If a test is wrong, explain why in the PR and write its replacement.
5. **Never change the public API without being asked.** `attach`, `log_step`, `RLWatch`, `RLWatchConfig`, `load_config` are contracts with users.
6. **When adding a detector:** write the config dataclass, the detector class, wire it into `DetectorSuite`, write unit tests, write a simulation fixture, update `README.md`'s detector table. All in the same PR.
7. **When adding a framework integration:** write the `_attach_<name>` helper in `core.py`, add a detection branch in `_detect_framework`, write an integration test, add an example script in `examples/`, update the README framework list.
8. **When adding an alert channel:** new sender class in `alerts.py`, new config dataclass section, wire it into `AlertManager.__init__`, add env var overrides to `config.py::load_config`, write a mocked-transport integration test.
9. **Prefer the dedicated tools** (Read/Edit/Write/Grep/Glob) over Bash for the tasks they cover.
10. **If you're unsure whether a change is in scope, ask.** The user would rather answer a question than undo a surprise.

---

## Quick health check commands

Before calling any significant change done, run:

```bash
pytest -v                                                     # all tests green
pytest --cov=rlwatch --cov-report=term-missing | tail -20    # coverage holding
python examples/simulate_grpo_run.py                          # end-to-end sanity
rlwatch diagnose                                              # CLI sanity
```

If any of these fail, fix that first.
