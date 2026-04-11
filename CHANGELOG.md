# Changelog

All notable changes to rlwatch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-04-11 — stage one of Phase 1

### Added — test harness (ROADMAP 1.1)
- **Five-tier test harness** matching the contract in `CLAUDE.md`:
  - Tier 1 (`tests/unit/`) — exhaustive state-machine coverage for every
    detector, plus dedicated suites for `AlertManager`, storage migrations,
    config layering, and the `attach()` surface.
  - Tier 2 (`tests/property/`) — Hypothesis invariants for warmup, cooldown,
    rate limits, constant input, `None` safety, and the high-value
    "monotonically healthy run produces no critical alerts" property.
  - Tier 3 (`tests/simulations/`) — six deterministic metric-trace generators
    replayed through the full pipeline with declarative expected alert sets.
    The regression moat: every future bug fix adds a fixture here.
  - Tier 4 (`tests/integration/`) — real SQLite on `tmp_path`, real YAML,
    real CLI via `CliRunner`, mocked Slack webhook (`unittest.mock` patch
    on `slack_sdk.webhook.WebhookClient`), mocked SMTP via
    `unittest.mock.patch('smtplib.SMTP')`, end-to-end manual attach test,
    and a TRL integration test gated on the `[trl]` extra.
  - Tier 5 (`tests/perf/`) — `pytest-benchmark` with hard latency assertions:
    `check_step` < 1 ms, `check_step` with rewards array < 2 ms,
    `log_step` full pipeline < 3 ms, `log_metrics` > 1000 rows/sec.
- **`tests/conftest.py`** extended with `tmp_log_dir`, `make_config`, and
  `benchmark_ci_slack` fixtures, plus Hypothesis profile registration
  (`dev` locally, `ci` when `CI=true`).
- **Coverage gate** at 90% on `src/rlwatch/` (excluding `dashboard.py`),
  enforced via `pytest --cov-fail-under=90`.
- **CI workflow restructured** into five jobs: `unit` (3py × 2os, Tiers
  1–3 + smoke + forbidden-pattern grep), `integration` (Linux py3.11),
  `trl` (Linux py3.11 with `[trl]` extra), `perf` (Linux py3.11 with
  2× CI slack), `coverage` (aggregated 90% gate). Caches pip + Hypothesis
  database between runs.
- **`TESTING.md`** — new contributor-facing testing guide.

### Added — library
- **`LossNaNDetector`** — fires critical the instant `loss` is non-finite.
  No rolling state, no warning tier — by the time you see this, the only
  useful action is to stop the run.
- **`GradientNormSpikeDetector`** — z-score model against a frozen baseline
  for the gradient norm. Defaults to `baseline_mode='frozen'` because grad
  norms drift slowly on healthy runs and a rolling baseline silently follows
  the trend.
- **`baseline_mode: "rolling" | "frozen"`** option on `KLExplosionDetector`
  and `AdvantageVarianceDetector`. Existing detectors keep `"rolling"` as
  the default — no behavior change for current users — but the option is
  there for runs where slow drift is the actual failure.
- **`attach(trainer=...)`** kwarg + **`monitor.attach_to_trainer(trainer)`**
  helper. Replaces the prior `gc.get_objects()` walk with explicit Trainer
  registration.
- **`grad_norm`** added to `RLWatch.log_step`, `MetricStore.log_metrics`,
  the SQLite `metrics` table (via the v1→v2 migration), the TRL callback's
  metric mapping, and the CLI `diagnose` summary.
- **`RLWATCH_FRAMEWORK`** environment variable in the config `env_map`.
- **`schema_version`** table now actually carries a real migration: v1 DBs
  open and pick up the new `grad_norm` column via `ALTER TABLE`. The
  `_migrate()` hook is no longer just a stub — it has a tested branch and
  documents how to add the next one.
- **`examples/simulate_grpo_run.py`** accepts a `seed` kwarg so the
  canonical Tier 3 fixture is reproducible.

### Fixed
- **`AlertManager` cooldown preemption.** A critical alert that follows a
  warning from the same detector inside the cooldown window was being
  silently suppressed. Cooldown is now tracked per `(detector, severity)`
  pair, so escalation is never muted by an earlier lesser alert.
- **`load_config(**overrides)` deep-merge.** Passing
  `entropy_collapse={"threshold": 0.5}` no longer replaces the whole
  sub-dataclass with a bare dict. Nested dicts are merged onto the
  existing dataclass; unknown keys raise `ValueError` instead of failing
  silently downstream.
- **`_attach_trl()` no longer walks `gc.get_objects()`.** That walk was
  slow on large processes, fragile (depended on `attach()` being called at
  exactly the right time), and untestable. The new path is explicit: pass
  `trainer=` to `attach()`, or call `monitor.attach_to_trainer(trainer)`
  after constructing your Trainer.

### Notes
- The simplified Hartigan dip test in `RewardHackingDetector` is still the
  home-rolled approximation — replacing it with the real `diptest` package
  is deferred to v0.3 (it adds a C extension dep).
- `dashboard.py` is excluded from the 90% coverage gate. Streamlit isn't
  in-process testable without a headless browser; the dashboard is covered
  by manual smoke tests.

## [0.1.0] — Phase 0 foundation

## [0.1.0] — Phase 0 foundation

Initial internal release. Four detectors (entropy collapse, KL explosion,
reward hacking, advantage variance), two-line `attach()` API with TRL
auto-detection, SQLite metric store in WAL mode, console/Slack/email alert
delivery with cooldown + rate limiting, Streamlit dashboard, Click CLI
(`init` / `runs` / `diagnose` / `dashboard`), YAML config with env-var
overrides, and a pytest suite covering detectors, core, and config.
