"""Tests for the [dashboard] optional extra refactor.

Two invariants:
1. The core ``import rlwatch`` and ``rlwatch.attach()`` path must work with
   zero dashboard dependencies. ``streamlit``, ``plotly``, and ``pandas``
   live behind the [dashboard] extra in v0.3+, and any accidental import in
   the core path would re-introduce the dependency.
2. ``rlwatch dashboard`` CLI must print a friendly install hint and exit
   non-zero when the extra isn't installed.

Both are simulated via ``sys.modules`` patching — we don't actually
uninstall streamlit at test time.
"""

from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner


@pytest.fixture
def hide_streamlit(monkeypatch):
    """Make ``import streamlit`` raise ImportError for the duration of the test."""
    # Block any future ``import streamlit`` by inserting a sentinel that
    # raises on attribute access. The simplest cross-version trick is to
    # patch sys.modules with a None entry, which makes the import machinery
    # raise ImportError on subsequent imports.
    monkeypatch.setitem(sys.modules, "streamlit", None)
    yield


class TestCoreImportWithoutStreamlit:
    def test_rlwatch_imports_without_streamlit(self, hide_streamlit):
        # If anything in the core import path touches streamlit, this raises.
        # We re-import to bypass any caching from previous tests.
        import importlib

        import rlwatch

        importlib.reload(rlwatch)
        assert hasattr(rlwatch, "attach")
        assert hasattr(rlwatch, "RLWatch")
        assert hasattr(rlwatch, "log_step")

    def test_attach_works_without_streamlit(self, hide_streamlit, tmp_log_dir, monkeypatch):
        monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)
        import rlwatch

        monitor = rlwatch.attach(framework="manual", run_id="no_streamlit_test")
        try:
            monitor.log_step(0, entropy=2.0)
            assert len(monitor.store.get_metrics()) == 1
        finally:
            monitor.stop()


class TestDashboardCliFriendlyError:
    def test_dashboard_subcommand_errors_when_extra_missing(self, hide_streamlit):
        """`rlwatch dashboard` must exit non-zero with a clear hint."""
        from rlwatch.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code != 0
        assert "rlwatch[dashboard]" in result.output
