"""Auto-detection test matrix for framework integrations.

Tests that ``_detect_framework()`` correctly identifies TRL, veRL, and
OpenRLHF when their modules are in ``sys.modules``, and falls back to
``"manual"`` when none are present. Also tests that ``attach()`` with
``framework="auto"`` routes to the correct ``_attach_*`` function.
"""

from __future__ import annotations

import sys
import types

import pytest

from rlwatch.core import _detect_framework, attach


def _make_fake_module(name: str) -> types.ModuleType:
    """Create a minimal fake module for sys.modules injection."""
    return types.ModuleType(name)


class TestDetectFramework:
    """Unit tests for ``_detect_framework()``."""

    def test_detects_trl(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "trl", _make_fake_module("trl"))
        # Remove others to avoid ambiguity.
        monkeypatch.delitem(sys.modules, "verl", raising=False)
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        assert _detect_framework() == "trl"

    def test_detects_verl(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.setitem(sys.modules, "verl", _make_fake_module("verl"))
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        assert _detect_framework() == "verl"

    def test_detects_openrlhf(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.delitem(sys.modules, "verl", raising=False)
        monkeypatch.setitem(
            sys.modules, "openrlhf", _make_fake_module("openrlhf")
        )
        assert _detect_framework() == "openrlhf"

    def test_falls_back_to_manual(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.delitem(sys.modules, "verl", raising=False)
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        assert _detect_framework() == "manual"

    def test_trl_has_priority_over_verl(self, monkeypatch):
        """When both TRL and veRL are imported, TRL wins (checked first)."""
        monkeypatch.setitem(sys.modules, "trl", _make_fake_module("trl"))
        monkeypatch.setitem(sys.modules, "verl", _make_fake_module("verl"))
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        assert _detect_framework() == "trl"

    def test_verl_has_priority_over_openrlhf(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.setitem(sys.modules, "verl", _make_fake_module("verl"))
        monkeypatch.setitem(
            sys.modules, "openrlhf", _make_fake_module("openrlhf")
        )
        assert _detect_framework() == "verl"


class TestAttachAutoDetection:
    """Integration tests for ``attach(framework="auto")`` routing."""

    def test_auto_falls_back_to_manual_when_no_framework(
        self, tmp_log_dir, monkeypatch
    ):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.delitem(sys.modules, "verl", raising=False)
        monkeypatch.delitem(sys.modules, "openrlhf", raising=False)
        monitor = attach(framework="auto", run_id="auto_manual")
        try:
            assert monitor.config.framework == "manual"
        finally:
            monitor.stop()

    def test_explicit_framework_overrides_auto(
        self, tmp_log_dir, monkeypatch
    ):
        """``framework="manual"`` should not auto-detect even if TRL is loaded."""
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monkeypatch.setitem(sys.modules, "trl", _make_fake_module("trl"))
        monitor = attach(framework="manual", run_id="explicit_manual")
        try:
            assert monitor.config.framework == "manual"
        finally:
            monitor.stop()

    def test_auto_detects_verl_and_attaches(
        self, tmp_log_dir, monkeypatch
    ):
        """When veRL is in sys.modules and framework='auto', the monitor
        should detect veRL and stash the tracker."""
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monkeypatch.delitem(sys.modules, "trl", raising=False)

        # Create a minimal fake verl module tree so _attach_verl doesn't crash.
        fake_verl = _make_fake_module("verl")
        monkeypatch.setitem(sys.modules, "verl", fake_verl)

        monitor = attach(framework="auto", run_id="auto_verl")
        try:
            assert monitor.config.framework == "verl"
            # The tracker should be stashed on the monitor.
            assert hasattr(monitor, "_verl_tracker")
        finally:
            monitor.stop()

    def test_auto_detects_openrlhf_and_attaches(
        self, tmp_log_dir, monkeypatch
    ):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        monkeypatch.delitem(sys.modules, "trl", raising=False)
        monkeypatch.delitem(sys.modules, "verl", raising=False)

        fake_openrlhf = _make_fake_module("openrlhf")
        monkeypatch.setitem(sys.modules, "openrlhf", fake_openrlhf)

        monitor = attach(framework="auto", run_id="auto_openrlhf")
        try:
            assert monitor.config.framework == "openrlhf"
            assert hasattr(monitor, "_openrlhf_logger")
        finally:
            monitor.stop()
