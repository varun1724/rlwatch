"""Tier 5: ``DetectorSuite.check_step`` latency budget.

CLAUDE.md: < 1ms per check_step on a 2020-era laptop CPU. We multiply the
threshold by ``BENCHMARK_CI_SLACK`` (default 1.0 locally, 2.0 in CI) to
absorb GitHub runner variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.detectors import DetectorSuite

pytestmark = pytest.mark.perf


def _suite() -> DetectorSuite:
    cfg = RLWatchConfig()
    cfg.entropy_collapse.warmup_steps = 0
    cfg.kl_explosion.warmup_steps = 0
    cfg.reward_hacking.warmup_steps = 0
    cfg.advantage_variance.warmup_steps = 0
    cfg.gradient_norm_spike.warmup_steps = 0
    return DetectorSuite(cfg)


def test_check_step_basic_under_1ms(benchmark, benchmark_ci_slack):
    suite = _suite()
    # Prime the detectors so they're past the cold-start branches.
    for step in range(100):
        suite.check_step(
            step,
            entropy=2.5,
            kl_divergence=0.01,
            reward_std=0.5,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    def _run():
        suite.check_step(
            500,
            entropy=2.5,
            kl_divergence=0.01,
            reward_std=0.5,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    benchmark.pedantic(_run, rounds=200, warmup_rounds=10)
    assert benchmark.stats["mean"] < 0.001 * benchmark_ci_slack, (
        f"check_step mean {benchmark.stats['mean']*1000:.3f}ms exceeds "
        f"{1.0 * benchmark_ci_slack:.2f}ms budget"
    )


def test_check_step_with_rewards_array_under_2ms(benchmark, benchmark_ci_slack):
    """The reward-hacking dip-test path is the slowest branch."""
    suite = _suite()
    rewards = np.random.default_rng(0).normal(0, 1, size=256)

    for step in range(100):
        suite.check_step(
            step,
            entropy=2.5,
            kl_divergence=0.01,
            rewards=rewards,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    def _run():
        suite.check_step(
            500,
            entropy=2.5,
            kl_divergence=0.01,
            rewards=rewards,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    benchmark.pedantic(_run, rounds=100, warmup_rounds=10)
    assert benchmark.stats["mean"] < 0.002 * benchmark_ci_slack, (
        f"check_step+rewards mean {benchmark.stats['mean']*1000:.3f}ms exceeds "
        f"{2.0 * benchmark_ci_slack:.2f}ms budget"
    )
