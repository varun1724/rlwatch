# Contributing

rlwatch is a monitoring library — if it has bugs, it costs someone a GPU budget. The test harness is the most load-bearing part of the repo. Two documents cover the contract every PR has to meet:

- **[CLAUDE.md](https://github.com/varun1724/rlwatch/blob/main/CLAUDE.md)** — the authoritative spec. Cardinal rules, the five-tier test harness contract, code style, repository conventions. Read this first.
- **[TESTING.md](https://github.com/varun1724/rlwatch/blob/main/TESTING.md)** — the practical "how to run, write, and debug tests" guide. Tier-by-tier breakdown, pytest invocation cookbook, "how to write a new detector test" checklist, "how to add a regression fixture" workflow, flaky-test policy.

## Quickstart for a typical change

```bash
git clone https://github.com/varun1724/rlwatch
cd rlwatch
pip install -e ".[dev]"

# Make your change
# ...

pytest -v                                        # all five tiers green
pytest --cov=rlwatch --cov-fail-under=90        # coverage gate

git commit -m "..."
git push
```

CI runs the same checks plus a few more (the cardinal-rule-#1 smoke test, the forbidden-pattern grep, the TRL integration test under the `[trl]` extra). All five tier jobs must be green for a PR to merge.

## What every PR must include

- **Unit tests** for any new branch in the library code
- **Integration tests** if the change touches CLI, alert delivery, storage schema, or framework integration
- **A simulation fixture** if the change fixes or introduces a failure-mode detection (this is the regression moat — see TESTING.md)
- **A benchmark** if the change touches the hot path
- **A `CHANGELOG.md` entry** under the appropriate `[Unreleased]` category

## What will get bounced

- Mocking `DetectorSuite` in a test for `RLWatch.log_step`
- Asserting on exact alert *messages* (assert on `alert.detector` and `alert.severity` instead — messages will change)
- Tests that touch the real filesystem outside `tmp_path`
- Tests that hit the real network
- Tests that depend on wall-clock time
- Flaky tests "fixed" with retries
- Network calls outside `src/rlwatch/alerts.py` (CI greps for this)

See CLAUDE.md's "Anti-patterns to refuse" section for the full list with rationale.
