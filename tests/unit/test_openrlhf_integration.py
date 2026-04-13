"""Unit tests for the OpenRLHF logger integration.

Same pattern as test_verl_integration.py: mock OpenRLHF's logger interface,
validate metric mapping and the logger lifecycle without a real OpenRLHF
installation.
"""

from __future__ import annotations

from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch
from rlwatch.integrations.openrlhf_logger import (
    RLWatchOpenRLHFLogger,
    _map_metrics,
)


class TestMetricMapping:
    def test_maps_openrlhf_names_to_rlwatch_names(self):
        data = {
            "entropy_loss": -2.5,  # negative (loss term)
            "kl": 0.05,
            "rollout/reward_mean": 0.8,
            "rollout/reward_std": 0.3,
            "policy_loss": 0.42,
            "actor_grad_norm": 1.2,
            "advantage_std": 0.9,
        }
        mapped = _map_metrics(data)
        assert mapped["entropy"] == 2.5  # sign-flipped from -2.5
        assert mapped["kl_divergence"] == 0.05
        assert mapped["reward_mean"] == 0.8
        assert mapped["reward_std"] == 0.3
        assert mapped["loss"] == 0.42
        assert mapped["grad_norm"] == 1.2
        assert mapped["advantage_std"] == 0.9

    def test_positive_entropy_loss_not_flipped(self):
        data = {"entropy_loss": 2.5}  # already positive
        mapped = _map_metrics(data)
        assert mapped["entropy"] == 2.5

    def test_fallback_names_work(self):
        data = {
            "ppo_kl": 0.03,
            "reward_mean": 0.6,
            "loss": 0.5,
            "grad_norm": 1.0,
        }
        mapped = _map_metrics(data)
        assert mapped["kl_divergence"] == 0.03
        assert mapped["reward_mean"] == 0.6
        assert mapped["loss"] == 0.5
        assert mapped["grad_norm"] == 1.0

    def test_priority_order_first_hit_wins(self):
        data = {
            "kl": 0.05,
            "ppo_kl": 0.99,  # should be ignored
        }
        mapped = _map_metrics(data)
        assert mapped["kl_divergence"] == 0.05

    def test_unrecognized_metrics_ignored(self):
        data = {
            "timing/generation": 5.0,
            "critic_loss": 0.3,
            "kl": 0.05,
        }
        mapped = _map_metrics(data)
        assert "timing/generation" not in mapped
        assert "critic_loss" not in mapped
        assert mapped["kl_divergence"] == 0.05

    def test_empty_dict_returns_empty(self):
        assert _map_metrics({}) == {}

    def test_non_numeric_values_skipped(self):
        data = {"kl": "not_a_number", "policy_loss": 0.5}
        mapped = _map_metrics(data)
        assert "kl_divergence" not in mapped
        assert mapped["loss"] == 0.5


class TestRLWatchOpenRLHFLogger:
    def test_log_train_forwards_to_monitor(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "openrlhf_test"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            lgr = RLWatchOpenRLHFLogger(monitor)
            lgr.log_train(
                global_step=10,
                logs_dict={
                    "entropy_loss": -2.5,
                    "kl": 0.05,
                    "rollout/reward_mean": 0.8,
                    "policy_loss": 0.42,
                },
            )

            metrics = monitor.store.get_metrics()
            assert len(metrics) == 1
            assert metrics[0]["entropy"] == 2.5  # sign-flipped
            assert metrics[0]["kl_divergence"] == 0.05
            assert metrics[0]["reward_mean"] == 0.8
            assert metrics[0]["loss"] == 0.42
        finally:
            monitor.stop()

    def test_log_train_empty_data_noop(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "openrlhf_empty"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            lgr = RLWatchOpenRLHFLogger(monitor)
            lgr.log_train(global_step=0, logs_dict={})
            assert len(monitor.store.get_metrics()) == 0
        finally:
            monitor.stop()

    def test_log_eval_is_noop(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "openrlhf_eval"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            lgr = RLWatchOpenRLHFLogger(monitor)
            lgr.log_eval(global_step=0, logs_dict={"eval_pass1": 0.8})
            assert len(monitor.store.get_metrics()) == 0
        finally:
            monitor.stop()

    def test_close_and_finish_are_noops(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "openrlhf_lifecycle"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            lgr = RLWatchOpenRLHFLogger(monitor)
            lgr.close()
            lgr.finish()
        finally:
            monitor.stop()

    def test_multiple_log_train_calls(self, tmp_log_dir):
        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        cfg.run_id = "openrlhf_multi"
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            lgr = RLWatchOpenRLHFLogger(monitor)
            for step in range(5):
                lgr.log_train(
                    global_step=step,
                    logs_dict={"policy_loss": 0.5 - 0.05 * step},
                )
            metrics = monitor.store.get_metrics()
            assert len(metrics) == 5
            assert metrics[0]["loss"] == 0.5
            assert metrics[4]["loss"] == 0.3
        finally:
            monitor.stop()
