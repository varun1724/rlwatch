# Changelog

All notable changes to rlwatch are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-04-11 — stage two of Phase 1, first PyPI release

### Added
- **Discord webhook alert channel** — `DiscordConfig` dataclass, `_DiscordSender`
  using stdlib `urllib.request`, `RLWATCH_DISCORD_WEBHOOK_URL` env var,
  optional role @-mentions on critical alerts only. 11 integration tests.
- **Generic HTTP webhook alert channel** — `WebhookConfig` (url, method,
  headers, template_json, timeout), `_WebhookSender` using `string.Template`
  `${field}` substitution into a JSON payload. Default template ships;
  custom templates supported. `_json_escape` helper for safe substitution
  of arbitrary string content (quotes, backslashes, newlines, unicode).
  Substituted body validated with `json.loads` before sending. 19 integration
  tests including the JSON-escape edge cases.
- **mkdocs-material documentation site** at `https://varun1724.github.io/rlwatch/`.
  Pages: index, getting-started, six per-detector deep dives, configuration,
  CLI, five per-channel alert pages, end-to-end TRL tutorial, FAQ, contributing,
  v0.3.0 launch blog post. README stays the source of truth for the pitch
  and detector table; docs cross-link, never duplicate.
- **End-to-end TRL GRPO tutorial** — GPT-2 + 20-prompt synthetic dataset +
  TRL `GRPOTrainer` with deliberately misconfigured LR (`1e-3`) that
  induces a real entropy collapse around step ~150. Runs on CPU in
  ~5 minutes, deterministic seeds. Lives as both `examples/trl_grpo_tutorial.py`
  and `docs/tutorials/trl-grpo-end-to-end.md`. Monthly CI cron in
  `.github/workflows/tutorial.yml` asserts the alert still fires on future
  TRL releases.
- **`[tutorial]` extra** in `pyproject.toml` pinning `trl`, `transformers`,
  `torch`, `datasets` to known-working versions for the tutorial.
- **`py.typed` marker** — rlwatch is now a typed package; downstream type
  checkers respect our type hints.
- **`release.yml` workflow** — OIDC trusted-publisher PyPI release workflow
  triggered on `v*.*.*` tags. Three jobs: build sdist+wheel, run full
  non-TRL test suite + 90% coverage gate, publish to PyPI via OIDC and
  create the GitHub release. Manual approval gate via the `pypi` GitHub
  Environment.
- **`docs.yml` workflow** — deploys mkdocs-material to GitHub Pages on
  every push to `main` that touches docs, mkdocs.yml, README, or `src/`.
  `mkdocs build --strict` catches broken internal links.

### Changed
- **`streamlit`, `plotly`, and `pandas` moved out of core dependencies into
  the `[dashboard]` extra.** This shrinks the default `pip install rlwatch`
  install footprint by ~7x (from ~150MB to ~20MB). The `rlwatch dashboard`
  CLI now prints a friendly install hint and exits non-zero if the extra
  isn't installed. Existing dashboard users need to switch to
  `pip install "rlwatch[dashboard]"`. **Breaking install-time change.**
- **README polish** — install line now points at PyPI; PyPI/Python/CI/license/docs
  badges added at the top; Discord and webhook subsections added under
  "Setting up alerts"; "Supported frameworks" section updated to reflect
  the `attach(trainer=...)` recommended path; new Documentation section
  pointing at the docs site.
- **`pyproject.toml` metadata** — version `0.3.0`, real homepage URL
  (`github.com/varun1724/rlwatch`), real changelog/docs URL entries,
  `Typing :: Typed` and Linux/macOS classifiers.
- **`load_config` `RLWATCH_DISCORD_WEBHOOK_URL`, `RLWATCH_WEBHOOK_URL`,
  `RLWATCH_WEBHOOK_TEMPLATE`** env vars added to the env_map. Discord and
  webhook channels auto-enable when their URL is set.
- **TRL example** — `examples/trl_integration.py` (a stub with commented-out
  code) replaced by `examples/trl_grpo_tutorial.py` (a real, runnable,
  CPU-friendly tutorial).

### Notes
- v0.2.0 ships as a historical git tag only — no PyPI release. The PyPI
  debut is v0.3.0, which contains v0.2.0's content plus the adoption-focus
  work in this section.
- Coverage stays at 91.6% on `src/rlwatch/` (`dashboard.py` excluded).
- Total tests: 182 + 2 skipped (TRL gated to dedicated CI job).

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
