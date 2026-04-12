"""Smoke test for examples/trl_grpo_tutorial.py.

This is the regression guard for the headline tutorial. The script runs
GPT-2 + TRL GRPO with a deliberately too-high LR and is expected to fire
at least one ``entropy_collapse`` alert. If a future TRL release silently
changes the GRPO loop in a way that breaks the tutorial, this test (run
monthly via .github/workflows/tutorial.yml plus on every change to the
tutorial script) catches it.

Gated by ``pytest.importorskip`` so it auto-skips when the [trl,tutorial]
extras aren't installed (which is the default for the unit + integration
CI jobs — only the dedicated tutorial cron installs them).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Use a dedicated "tutorial" marker — NOT "trl". The quick trl CI job
# (test.yml) runs `pytest -m trl` against the lightweight callback-
# registration test. The heavyweight tutorial test belongs to its own
# dedicated workflow (tutorial.yml) and should not be collected by the trl
# job, which installs [dev,trl] but not [tutorial] (so the pinned versions
# the tutorial depends on may not match).
pytestmark = [pytest.mark.integration, pytest.mark.tutorial]

# Skip the whole module if the tutorial deps aren't present.
pytest.importorskip("trl")
pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("datasets")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TUTORIAL = REPO_ROOT / "examples" / "trl_grpo_tutorial.py"


def test_tutorial_runs_and_fires_entropy_collapse():
    """Run the tutorial as a subprocess and assert the alert fired.

    Subprocess so we get a clean Python process per test (TRL leaves global
    state behind that would otherwise contaminate other tests). 10-minute
    timeout so a stuck CPU machine surfaces as a clear failure rather than
    hanging the CI job.
    """
    assert TUTORIAL.exists(), f"tutorial script missing at {TUTORIAL}"

    result = subprocess.run(
        [sys.executable, str(TUTORIAL)],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes
        cwd=str(REPO_ROOT),
    )

    # Combine stdout + stderr because rlwatch's rich console writes alert
    # panels to stderr while the script's print() lines go to stdout.
    combined = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, (
        f"tutorial exited with code {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "entropy_collapse" in combined.lower(), (
        f"tutorial completed but no entropy_collapse alert was fired.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
