# Testing rlwatch

rlwatch is a monitoring library. If it has bugs, it causes silent, expensive
failures on someone else's GPU run. The test harness is therefore the most
load-bearing part of the repo. This document is the practical "how do I run,
write, and debug tests" guide; **`CLAUDE.md` is the authoritative spec** for
the five-tier model and the contract every PR has to meet.

If you're an AI agent reading this, also read `CLAUDE.md` — the cardinal
rules and "anti-patterns to refuse" sections override anything below.

---

## The five tiers, briefly

| Tier | Directory | What it does | Runtime budget |
|---|---|---|---|
| 1 — Unit | `tests/unit/` | One test class per detector or component, exhaustive state transitions | < 10 s |
| 2 — Property | `tests/property/` | Hypothesis invariants (no warmup alerts, cooldown honored, etc.) | < 60 s |
| 3 — Simulation | `tests/simulations/` | Replay canned metric traces through the full pipeline; assert the fired-alert set | < 30 s |
| 4 — Integration | `tests/integration/` | Real SQLite, real YAML, real CLI, mocked Slack/SMTP, optional TRL | < 60 s |
| 5 — Performance | `tests/perf/` | `pytest-benchmark` with hard latency/throughput assertions | < 30 s |

The full suite (Tiers 1–5) should land under three minutes locally.

---

## Setup

```bash
pip install -e ".[dev]"
# Optional: pulls transformers + trl, only needed for the TRL Tier 4 test
pip install -e ".[trl]"
```

If `pytest` says "no module named rlwatch", you skipped the editable install.
The `tests/conftest.py` puts `src/` on `sys.path` as a fallback, but the
editable install is what's documented in CI and in `CLAUDE.md`.

---

## Running tests

```bash
# Everything (excludes nothing — runs all five tiers)
pytest -v

# Coverage gate (must report >= 90% to merge)
pytest --cov=rlwatch --cov-fail-under=90

# One tier in isolation
pytest tests/unit/
pytest tests/property/
pytest tests/simulations/
pytest tests/integration/
pytest tests/perf/ --benchmark-only

# Skip the TRL test if you don't have transformers installed (it auto-skips)
pytest -m "not trl"

# Just one file
pytest tests/unit/test_detectors_loss_nan.py -v

# Just one test
pytest tests/unit/test_detectors.py::TestEntropyCollapseDetector::test_alerts_re_arm_after_recovery -v

# See where coverage is lacking
pytest --cov=rlwatch --cov-report=term-missing
```

---

## Writing a new detector test (Tier 1)

Every detector ships with a state-machine test file at
`tests/unit/test_detectors_<name>.py` that covers, at minimum:

```
warmup → healthy → warning → critical → recovery → re-arm → noop branches
```

The minimum checklist for a new detector PR:

- [ ] `test_no_alert_during_warmup`
- [ ] `test_no_alert_when_input_healthy`
- [ ] `test_alerts_on_<failure_mode>` (warning + critical separately)
- [ ] `test_<recovery_path>` if the detector tracks consecutive state
- [ ] `test_none_input_noop`
- [ ] `test_disabled_noop`
- [ ] If the detector uses a frozen baseline: `test_frozen_baseline_does_not_drift`

Then add the detector to **at least one Tier 3 fixture** (`tests/simulations/generators.py`)
and update the expected alert set in `tests/simulations/test_replay.py::FIXTURES`.

---

## Adding a regression fixture (Tier 3) when fixing a bug

The Tier 3 fixtures are the regression moat. Every fix that wasn't caught by
the existing harness gets a new fixture so it stays caught.

1. Reproduce the bug in a unit test first.
2. Add a new generator function in `tests/simulations/generators.py` that
   produces a metric trace exhibiting the bug. Make it deterministic — use
   `_seeded(seed)`, never `random.random()`.
3. Add a tuple to `FIXTURES` in `tests/simulations/test_replay.py` with the
   expected `(detector, severity)` set.
4. Run `pytest tests/simulations/` — the test should fail (proving it
   reproduces the bug).
5. Fix the bug.
6. Run again — both the unit test and the new fixture should pass.

This is the workflow that makes the fixture set monotonically more useful.

---

## Hypothesis tips (Tier 2)

- The CI Hypothesis profile uses `derandomize=True, deadline=1000ms`. Set
  `CI=true` in your shell to run the same profile locally.
- The `.hypothesis/` database is cached in CI (keyed by file hashes) so
  shrinking results persist between runs.
- **Never** suppress a Hypothesis failure with `@example` or `@settings(suppress_health_check=...)`.
  If Hypothesis found a counterexample, that's a real bug. Find the root
  cause; if the strategy itself is too liberal, narrow the strategy.
- Strategies live in `tests/property/strategies.py`. Reuse, don't redefine.

---

## pytest-benchmark gotchas (Tier 5)

- Perf tests run **only on Linux** in CI. macOS runners are too noisy for
  sub-millisecond assertions.
- Every assertion is multiplied by `BENCHMARK_CI_SLACK` (default 1.0
  locally, 2.0 in CI). Don't override locally — if your machine is too slow
  to hit 1.0, the test is signaling a real regression.
- Use `--benchmark-min-rounds=50 --benchmark-warmup=on` for stable results.
- When a perf test fails, *first* re-run on a quiet machine; perf flakes
  are a smell, not a fix excuse.

---

## Flaky test policy

If a test is flaky, it is **broken**. Find the race or the time-dependent
assertion and fix it. The CLAUDE.md anti-pattern list explicitly forbids:

- Sleeping in tests instead of using injected clocks or counter-based warmup
- Re-running a flaky test until it passes
- Adding `@pytest.mark.flaky`
- Adding `@example` to make a Hypothesis failure go away

The right responses are: convert wall-clock waits to deterministic step
counts, mock time, narrow the input range, or fix the actual race.

---

## Production readiness gates (before tagging a release)

The full checklist lives in the active stage-one plan; the most important
items:

- Coverage ≥ 90% on `src/rlwatch/` (`dashboard.py` excluded), enforced in CI.
- All five tiers green on Ubuntu + macOS × Python 3.10/3.11/3.12.
- TRL integration test green on Ubuntu py3.11 with the `[trl]` extra.
- Perf budgets pass with the 2× CI slack.
- No open issues labeled `bug:p0`.
- `python -c "import rlwatch; rlwatch.attach(framework='manual')"` exits 0
  on every CI cell. **This is cardinal rule #1.**
- No new network calls outside `alerts.py` (CI greps for this).
- Schema migration from any prior version still passes its dedicated test.

Don't ship if any of those are red.
