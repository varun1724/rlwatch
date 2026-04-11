"""Tier 5: SQLite ``log_metrics`` throughput.

CLAUDE.md: > 1000 steps/sec sustained, even with commit-per-row durability.
"""

from __future__ import annotations

import time

import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.storage import MetricStore

pytestmark = pytest.mark.perf


def test_log_metrics_1000_per_sec(tmp_log_dir, benchmark_ci_slack):
    cfg = RLWatchConfig()
    cfg.storage.log_dir = tmp_log_dir
    cfg.run_id = "perf_run"
    store = MetricStore(cfg)
    store.register_run(cfg)

    n = 1000
    start = time.perf_counter()
    for step in range(n):
        store.log_metrics(
            step,
            entropy=2.5,
            kl_divergence=0.01,
            reward_mean=0.0,
            reward_std=0.5,
            advantage_std=1.0,
            loss=0.3,
            grad_norm=1.0,
        )
    elapsed = time.perf_counter() - start
    store.close()

    rate = n / elapsed
    budget = 1000 / benchmark_ci_slack
    assert rate >= budget, (
        f"throughput {rate:.0f} rows/sec below {budget:.0f} rows/sec budget"
    )
