"""Tier 5: full ``log_step`` pipeline (store write + 6 detectors)."""

from __future__ import annotations

import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch

pytestmark = pytest.mark.perf


def test_log_step_under_3ms(benchmark, benchmark_ci_slack, tmp_log_dir):
    cfg = RLWatchConfig()
    cfg.storage.log_dir = tmp_log_dir
    cfg.run_id = "perf_e2e"
    cfg.entropy_collapse.warmup_steps = 0
    cfg.kl_explosion.warmup_steps = 0
    cfg.reward_hacking.warmup_steps = 0
    cfg.advantage_variance.warmup_steps = 0
    cfg.gradient_norm_spike.warmup_steps = 0
    monitor = RLWatch(cfg)
    monitor.start()

    # Prime the detectors past their cold-start branches.
    for step in range(100):
        monitor.log_step(
            step,
            entropy=2.5,
            kl_divergence=0.01,
            reward_std=0.5,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    counter = {"i": 1000}

    def _run():
        s = counter["i"]
        counter["i"] += 1
        monitor.log_step(
            s,
            entropy=2.5,
            kl_divergence=0.01,
            reward_std=0.5,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )

    try:
        benchmark.pedantic(_run, rounds=100, warmup_rounds=10)
        assert benchmark.stats["mean"] < 0.003 * benchmark_ci_slack, (
            f"log_step mean {benchmark.stats['mean']*1000:.3f}ms exceeds "
            f"{3.0 * benchmark_ci_slack:.2f}ms budget"
        )
    finally:
        monitor.stop()
