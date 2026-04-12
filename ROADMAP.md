# rlwatch Roadmap

**Vision:** rlwatch starts as a free, local, drop-in Python library for catching GRPO/PPO training failures, and grows into a paid hosted service that ML teams trust as the standard monitoring layer for RL post-training.

This roadmap is a living document. We update it as we learn from users. Every item has a **status**, an **owner** (unassigned by default), and a **why** line so we can challenge whether it still makes sense.

---

## Guiding principles

1. **Local-first, always.** The open-source library must remain fully functional with zero network calls. The hosted product adds value on top — it never gates the core.
2. **Two-line attach or it doesn't count.** Every new framework integration must preserve the `import rlwatch; rlwatch.attach()` experience.
3. **No false-positive spam.** A noisy monitor gets muted and then uninstalled. Every detector needs a configurable warmup, cooldown, and severity ladder.
4. **Every alert ships with a recommended action.** "KL exploded" is not useful. "KL exploded, reduce LR 5× or revert checkpoint" is.
5. **Tests before features.** See `CLAUDE.md` — every new detector, integration, or alert channel ships with unit + integration + simulation tests.

---

## Phase 0 — Foundation (shipped)

Status: ✅ Done

- Four statistical detectors (entropy collapse, KL explosion, reward hacking, advantage variance)
- Two-line attach API with TRL auto-detection
- SQLite metric store (WAL mode)
- Console + Slack + email alert delivery with cooldown & rate limiting
- Streamlit dashboard
- Click-based CLI (`init`, `runs`, `diagnose`, `dashboard`)
- YAML config + env var overrides
- Basic pytest coverage for detectors, core, config

---

## Phase 1 — Production-ready OSS (v0.2 → v0.5)

**Goal:** make rlwatch something an ML engineer would put into a real training run without hesitation. This is the phase that earns GitHub stars and PyPI downloads.

### 1.1 Test harness hardening — HIGH PRIORITY
Status: 🟢 Shipped in v0.2.0 (stage one) · Why: this is the single biggest risk. A monitoring library that itself has bugs is worse than no monitoring.

- [x] Coverage target ≥ 90% on `src/rlwatch/` (enforced via `pytest --cov --cov-fail-under=90`; `dashboard.py` excluded — Streamlit not in-process testable)
- [x] **Unit tests** — exhaustive state-machine coverage for every detector + AlertManager, storage migration, config layering, attach surface (`tests/unit/`)
- [x] **Property-based tests** with Hypothesis — warmup, cooldown, rate limit, constant input, None safety, monotone-healthy invariants (`tests/property/`)
- [x] **Simulation/golden tests** — six deterministic generators replayed through the full pipeline with declarative expected alert sets (`tests/simulations/`)
- [x] **Integration tests** — real SQLite on `tmp_path`, real YAML, real CLI via `CliRunner` (`tests/integration/`)
- [x] **Alert delivery tests** — Slack via `unittest.mock.patch` on `slack_sdk.webhook.WebhookClient` (responses doesn't intercept urllib), SMTP via `unittest.mock.patch` on `smtplib.SMTP`
- [x] **TRL integration test** — gated by `pytest.importorskip`, runs in a dedicated CI job under the `[trl]` extra (`tests/integration/test_trl_integration.py`)
- [x] **Regression fixture policy** — TESTING.md documents the workflow; every future fix adds a new fixture in `tests/simulations/`
- [x] GitHub Actions CI: matrix on Python 3.10/3.11/3.12 × Ubuntu/macOS; separate jobs for unit, integration, trl, perf, and aggregated coverage
- [x] `pytest-benchmark` for the hot path — `check_step` < 1 ms, `log_step` full pipeline < 3 ms, `log_metrics` > 1000 rows/sec

### 1.2 Detector robustness
- [x] Replace the simplified Hartigan dip test with the real implementation — shipped in v0.4.0 as `[monitoring]` optional extra with fallback
- [x] Add a **gradient norm spike** detector — shipped in v0.2.0 with `baseline_mode='frozen'` default
- [x] Add a **loss NaN / Inf** detector — shipped in v0.2.0
- [x] Add a **reward mean drift** detector — shipped in v0.4.0 (monotone drift for N steps + magnitude threshold)
- [ ] Let detectors declare dependencies on other detectors — *deferred indefinitely, architectural change*
- [x] **`baseline_mode: "rolling" | "frozen"`** option on KL and advantage detectors — shipped in v0.2.0 with `rolling` default for backwards compat

### 1.3 Framework integrations
- [x] **veRL** deep integration — shipped in v0.4.0 via custom tracking backend (`RLWatchVerLTracker`)
- [ ] **OpenRLHF** deep integration — same
- [ ] **Ray RLlib** support (stretch — different audience but big ecosystem)
- [ ] Auto-detection test matrix: import the framework, call `attach()`, assert the right integration path fires

### 1.4 Alert channels
- [ ] PagerDuty (critical alerts to on-call rotation) — *deferred to v0.4, enterprise channel needs real customer to design against*
- [x] Discord webhook (a lot of open-source ML teams live there) — shipped in v0.3.0
- [x] Generic webhook with templated JSON body — shipped in v0.3.0 with `string.Template` `${field}` substitution

### 1.5 Dashboard polish
- [ ] Run comparison view (overlay metrics from 2+ runs)
- [ ] Alert timeline across all runs
- [ ] Metric query export (CSV / Parquet)
- [ ] Auto-refresh toggle

### 1.6 Docs & examples
- [x] Full docs site (`mkdocs-material`) — shipped in v0.3.0 at https://varun1724.github.io/rlwatch/
- [x] End-to-end tutorial: train **GPT-2** with TRL + GRPO on a synthetic dataset with rlwatch attached — shipped in v0.3.0. *Originally planned as Qwen-0.5B + GSM8K but switched to GPT-2 + synthetic for CPU feasibility — Qwen on CPU is unworkable and Colab-only is an adoption-killer.*
- [ ] Playbook: "Your entropy is collapsing, here's what to try" — *partially covered by per-detector docs pages; full playbook deferred to v0.4*
- [ ] Case studies / postmortems (even synthetic ones to start) — *deferred until we have real users*

**Exit criteria for Phase 1:** 100+ GitHub stars, 500+ monthly PyPI downloads, at least 3 external users who have filed issues or PRs. Test suite > 90% coverage, all CI green on every PR.

---

## Phase 2 — Hosted service MVP (v1.0)

**Goal:** turn rlwatch into something a company will pay for. This is where the business starts.

### 2.1 rlwatch cloud — the service
- [ ] `rlwatch.attach(api_key="...")` ships metrics to a hosted backend in addition to local SQLite
- [ ] Multi-run web dashboard (no more `localhost:8501`)
- [ ] Team accounts — multiple researchers share a workspace
- [ ] Org-level alert routing (who gets paged for which run)
- [ ] Historical query over months of runs
- [ ] Public run sharing (read-only links for bug reports)

### 2.2 Backend architecture (first sketch — subject to change)
- Ingest: FastAPI + async workers, batched metric writes
- Storage: Postgres for metadata, ClickHouse or Timescale for metric time-series
- Auth: Clerk or Auth0
- Hosting: Fly.io or Railway to start, AWS when we outgrow them
- All infra defined as code (Terraform or Pulumi) from day one
- **Everything tested.** Integration tests hit real Postgres + real metric DB in a docker-compose'd CI job. Load tests assert we can ingest 10k metric points/sec per run.

### 2.3 Pricing & packaging
- Free tier: 1 user, 10 runs, 30-day retention
- Team tier: $49/user/month, unlimited runs, 1-year retention, PagerDuty integration, SSO
- Enterprise: custom, with on-prem deployment and SLA

### 2.4 Trust & compliance
- SOC 2 Type I within 6 months of launch (required by enterprise buyers)
- Data residency options (EU region) within 12 months
- Clear data deletion flow
- Privacy policy, security.txt, responsible disclosure

**Exit criteria for Phase 2:** 10 paying teams, $5k MRR, churn < 5%/month.

---

## Phase 3 — Beyond monitoring (v1.5+)

**Goal:** become the default RL post-training debugging suite, not just an alert tool.

### 3.1 Automated diagnosis
- [ ] Pattern classification: "this run looks like the classic reward-hacking failure we've seen 17 times before"
- [ ] Suggested hyperparameter deltas based on historical fixes
- [ ] Run comparison: "this run and that run diverged at step 280 — here's what was different"

### 3.2 Checkpoint intelligence
- [ ] Auto-tag the last known-good checkpoint before an instability
- [ ] One-click "resume from last good checkpoint with suggested fixes"

### 3.3 Reward model auditing (adjacent product)
- [ ] Hook into reward model inference to detect reward distribution shifts
- [ ] Flag inputs where the RM and a held-out verifier disagree
- [ ] This is a separate product surface but shares auth/billing/UI with the core

### 3.4 Experiment lineage graph
- [ ] Track parent/child relationships between runs (which run inherited hyperparams from which)
- [ ] Visualize the full experiment tree for a project

---

## Known risks we're actively tracking

| Risk | Why it matters | How we're hedging |
|---|---|---|
| **W&B ships a GRPO health dashboard** | They already have 1,400 enterprise customers and the metric stream | Move faster on the *automated alerting* and *pattern classification* layers — these are not W&B's core competency |
| **TRL v1.0 already logs entropy/KL/clip/reward for free** | Lowers the data-collection moat | Differentiate on *detection quality* and *remediation recommendations*, not on metric collection |
| **ML researchers can write their own Grafana alerts** | Buyer population is the most technically capable | Sell on *time saved* and *shared team knowledge*, not on technical capability |
| **Post-GRPO market is ~15 months old** | Pain isn't yet acute enough | Keep OSS bar extremely low (2-line install) to maximize seed adoption while pain accumulates |
| **Neptune acquired by OpenAI** | Lost a natural acquirer | Build for independence; don't rely on acquisition as the exit |

---

## How we update this roadmap

- Anything in Phase 1 can be reordered freely based on user feedback.
- Anything in Phase 2/3 requires a written justification before being moved earlier.
- Every completed item moves to `CHANGELOG.md` with a link to the PR that shipped it.
- Unchecked items older than 6 months get a status review — ship, defer explicitly, or delete.

Open a GitHub issue with the `roadmap` label to propose changes.
