"""Unit tests for the rlwatch.core attach surface."""

from __future__ import annotations

from unittest.mock import MagicMock

import rlwatch
from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch, _build_trl_callback, attach


class TestAttachManual:
    def test_attach_manual_returns_monitor(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="manual", run_id="t1")
        try:
            assert isinstance(monitor, RLWatch)
            assert monitor.run_id == "t1"
        finally:
            monitor.stop()

    def test_global_monitor_set(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="manual", run_id="t2")
        try:
            assert rlwatch.get_monitor() is monitor
        finally:
            monitor.stop()


class TestAttachToTrainer:
    def test_attach_to_trainer_registers_callback(self, tmp_log_dir, monkeypatch):
        """The helper must call ``trainer.add_callback`` exactly once.

        We can exercise this without ``transformers`` installed by faking the
        TrainerCallback base class via a MagicMock-backed module shim.
        """
        # Build a fake transformers module so _build_trl_callback succeeds.
        import sys
        import types

        fake = types.ModuleType("transformers")

        class _FakeCallback:
            pass

        fake.TrainerCallback = _FakeCallback
        monkeypatch.setitem(sys.modules, "transformers", fake)

        cfg = RLWatchConfig()
        cfg.storage.log_dir = tmp_log_dir
        monitor = RLWatch(cfg)
        monitor.start()

        try:
            trainer = MagicMock()
            monitor.attach_to_trainer(trainer)
            trainer.add_callback.assert_called_once()
            # The cached callback class is reused on subsequent calls.
            cls_first = monitor._trl_callback_class
            monitor.attach_to_trainer(trainer)
            assert monitor._trl_callback_class is cls_first
        finally:
            monitor.stop()


class TestStopIdempotent:
    def test_double_stop_does_not_raise(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="manual", run_id="t3")
        monitor.stop()
        monitor.stop()  # second call must be a no-op


class TestGlobalLogStep:
    def test_raises_when_not_attached(self, monkeypatch):
        # Reset the global so the test is hermetic.
        import rlwatch.core as core_module
        monkeypatch.setattr(core_module, "_global_monitor", None)
        import pytest
        with pytest.raises(RuntimeError, match="not attached"):
            rlwatch.log_step(0, entropy=2.5)

    def test_proxies_when_attached(self, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="manual", run_id="t_global")
        try:
            rlwatch.log_step(0, entropy=2.5)
            metrics = monitor.store.get_metrics()
            assert len(metrics) == 1
            assert metrics[0]["entropy"] == 2.5
        finally:
            monitor.stop()


class TestFrameworkDetection:
    def test_no_frameworks_returns_manual(self, monkeypatch):
        """When no RL framework is imported, detection falls back to manual."""
        import sys
        from rlwatch.core import _detect_framework

        # Hide any frameworks that may already be loaded.
        for name in ("trl", "verl", "openrlhf"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        assert _detect_framework() == "manual"


class TestFrameworkFallback:
    def test_attach_verl_without_install_falls_back(self, tmp_log_dir, monkeypatch):
        import sys
        monkeypatch.delitem(sys.modules, "verl", raising=False)
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="verl", run_id="t_verl")
        try:
            assert monitor.config.framework == "manual"
        finally:
            monitor.stop()

    def test_attach_openrlhf_without_install_falls_back(
        self, tmp_log_dir, monkeypatch
    ):
        import sys
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="openrlhf", run_id="t_oh")
        try:
            assert monitor.config.framework == "manual"
        finally:
            monitor.stop()

    def test_attach_trl_without_install_falls_back(
        self, tmp_log_dir, monkeypatch
    ):
        import sys
        # Make sure transformers is not importable.
        monkeypatch.setitem(sys.modules, "transformers", None)
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monitor = attach(framework="trl", run_id="t_trl_fallback")
        try:
            assert monitor.config.framework == "manual"
        finally:
            monitor.stop()
