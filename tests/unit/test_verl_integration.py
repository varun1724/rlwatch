"""Unit tests for the veRL tracking backend integration.

These tests mock veRL's ``Tracking`` class and the metric data format so we
can validate the metric mapping and the tracker lifecycle without a real
veRL installation. The integration test in ``tests/integration/`` is gated
by ``importorskip("verl")`` for the real thing.
"""

from __future__ import annotations

from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch
from rlwatch.integrations.verl_tracking import (
    RLWatchVerLTracker,
    _map_metrics,
)


class TestMetricMapping:
    def test_maps_verl_names_to_rlwatch_names(self):
        data = {
            "actor/entropy": 2.5,
            "actor/kl_divergence": 0.05,
            "rewards/mean": 0.8,
            "rewards/std": 0.3,
            "training/policy_loss": 0.42,
            "training/grad_norm": 1.2,
            "advantage_std": 0.9,
        }
        mapped = _map_metrics(data)
        assert mapped["entropy"] == 2.5
        assert mapped["kl_divergence"] == 0.05
        assert mapped["reward_mean"] == 0.8
        assert mapped["reward_std"] == 0.3
        assert mapped["loss"] == 0.42
        assert mapped["grad_norm"] == 1.2
        assert mapped["advantage_std"] == 0.9

    def test_fallback_names_work(self):
        data = {
            "entropy": 2.1,
            "kl": 0.03,
            "reward": 0.6,
            "loss": 0.5,
            "grad_norm": 1.0,
        }
        mapped = _map_metrics(data)
        assert mapped["entropy"] == 2.1
        assert mapped["kl_divergence"] == 0.03
        assert mapped["reward_mean"] == 0.6
        assert mapped["loss"] == 0.5
        assert mapped["grad_norm"] == 1.0

    def test_priority_order_first_hit_wins(self):
        data = {
            "actor/entropy": 2.5,
            "entropy": 1.0,  # should be ignored — first hit wins
        }
        mapped = _map_metrics(data)
        assert mapped["entropy"] == 2.5

    def test_unrecognized_metrics_ignored(self):
        data = {
            "some/custom/metric": 42.0,
            "actor/entropy": 2.5,
        }
        mapped = _map_metrics(data)
        assert "some/custom/metric" not in mapped
        assert mapped["entropy"] == 2.5

    def test_empty_dict_returns_empty(self):
        assert _map_metrics({}) == {}

    def test_non_numeric_values_skipped(self):
        data = {
            "actor/entropy": "not_a_number",
            "rewards/mean": 0.8,
        }
        mapped = _map_metrics(data)
        assert "entropy" not in mapped
        assert mapped["reward_mean"] == 0.8


class TestRLWatchVerLTracker:
    def test_log_forwards_to_monitor(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "verl_test"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            tracker = RLWatchVerLTracker(monitor)
            tracker.log(
                data={
                    "actor/entropy": 2.5,
                    "rewards/mean": 0.8,
                    "training/policy_loss": 0.42,
                },
                step=10,
            )

            metrics = monitor.store.get_metrics()
            assert len(metrics) == 1
            assert metrics[0]["entropy"] == 2.5
            assert metrics[0]["reward_mean"] == 0.8
            assert metrics[0]["loss"] == 0.42
        finally:
            monitor.stop()

    def test_log_with_empty_data_does_not_crash(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "verl_empty"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            tracker = RLWatchVerLTracker(monitor)
            # Empty data dict — no metrics to map, should be a silent no-op.
            tracker.log(data={}, step=0)
            assert len(monitor.store.get_metrics()) == 0
        finally:
            monitor.stop()

    def test_log_with_unrecognized_data_does_not_crash(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "verl_unknown"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            tracker = RLWatchVerLTracker(monitor)
            tracker.log(data={"something/irrelevant": 99.0}, step=0)
            assert len(monitor.store.get_metrics()) == 0
        finally:
            monitor.stop()

    def test_finish_and_close_are_noops(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "verl_lifecycle"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            tracker = RLWatchVerLTracker(monitor)
            tracker.finish()  # must not raise
            tracker.close()  # must not raise
        finally:
            monitor.stop()

    def test_multiple_log_calls_accumulate(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "verl_multi"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            tracker = RLWatchVerLTracker(monitor)
            for step in range(5):
                tracker.log(
                    data={"actor/entropy": 2.5 - 0.1 * step},
                    step=step,
                )
            metrics = monitor.store.get_metrics()
            assert len(metrics) == 5
            # Entropy should decrease across steps.
            assert metrics[0]["entropy"] == 2.5
            assert metrics[4]["entropy"] == 2.1
        finally:
            monitor.stop()
