"""Shared pytest fixtures and bootstrap for rlwatch.

Responsibilities:
1. Put ``src/`` on ``sys.path`` so the suite runs from a fresh checkout
   without ``pip install -e .`` (an editable install is still recommended,
   see CLAUDE.md).
2. Provide common fixtures used across all tiers — primarily a per-test
   ``tmp_log_dir`` so no test ever touches a real on-disk metric store.
3. Register Hypothesis profiles. The ``ci`` profile is loaded automatically
   when ``CI`` is set in the environment; locally we use the deterministic
   ``dev`` profile so failures are reproducible without surprising shrinks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck, settings
except ImportError:  # hypothesis only required for Tier 2
    settings = None  # type: ignore[assignment]

if settings is not None:
    settings.register_profile(
        "dev",
        max_examples=100,
        deadline=500,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile(
        "ci",
        max_examples=200,
        deadline=1000,
        derandomize=True,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.load_profile("ci" if os.environ.get("CI") else "dev")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> str:
    """An isolated rlwatch log directory for one test.

    Always prefer this over ``tempfile.mkdtemp()`` — pytest cleans it up.
    """
    d = tmp_path / "rlwatch_logs"
    d.mkdir()
    return str(d)


@pytest.fixture
def make_config(tmp_log_dir):
    """Factory that yields ``RLWatchConfig`` instances pre-pointed at a tmp dir.

    Tests that need to tweak detector thresholds can call
    ``cfg = make_config(); cfg.entropy_collapse.warmup_steps = 2``.
    """
    from rlwatch.config import RLWatchConfig

    def _factory(**top_level) -> RLWatchConfig:
        cfg = RLWatchConfig(**top_level)
        cfg.storage.log_dir = tmp_log_dir
        return cfg

    return _factory


@pytest.fixture
def benchmark_ci_slack() -> float:
    """Slack factor applied to perf assertions on CI hardware.

    Default 1.0 locally, 2.0 in CI (set ``BENCHMARK_CI_SLACK=2.0`` in the
    workflow). Tier 5 tests multiply their threshold by this factor.
    """
    return float(os.environ.get("BENCHMARK_CI_SLACK", "1.0"))
