"""Integration tests for the rlwatch CLI via click.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rlwatch.cli import main
from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch

pytestmark = pytest.mark.integration


def _seed_db(log_dir: str, run_id: str = "cli_run") -> None:
    cfg = RLWatchConfig()
    cfg.storage.log_dir = log_dir
    cfg.run_id = run_id
    cfg.entropy_collapse.warmup_steps = 2
    cfg.entropy_collapse.consecutive_steps = 5
    monitor = RLWatch(cfg)
    monitor.start()
    try:
        for step in range(5):
            monitor.log_step(step, entropy=2.5)
        for step in range(5, 20):
            monitor.log_step(step, entropy=0.1)
    finally:
        monitor.stop()


class TestRunsCommand:
    def test_missing_db_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["runs", "--log-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "No rlwatch database" in result.output

    def test_lists_existing_run(self, tmp_log_dir):
        _seed_db(tmp_log_dir, run_id="cli_run_1")
        runner = CliRunner()
        result = runner.invoke(main, ["runs", "--log-dir", tmp_log_dir])
        assert result.exit_code == 0
        assert "cli_run_1" in result.output


class TestDiagnoseCommand:
    def test_diagnose_rich_output(self, tmp_log_dir):
        _seed_db(tmp_log_dir, run_id="cli_run_2")
        runner = CliRunner()
        result = runner.invoke(
            main, ["diagnose", "--log-dir", tmp_log_dir, "--run-id", "cli_run_2"]
        )
        assert result.exit_code == 0
        # Health should be CRITICAL since we forced an entropy collapse.
        assert "CRITICAL" in result.output

    def test_diagnose_json_output(self, tmp_log_dir):
        _seed_db(tmp_log_dir, run_id="cli_run_3")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "diagnose",
                "--log-dir",
                tmp_log_dir,
                "--run-id",
                "cli_run_3",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["run_id"] == "cli_run_3"
        assert data["health"] in {"healthy", "warning", "critical"}
        assert "entropy_summary" in data

    def test_diagnose_nonexistent_run_errors(self, tmp_log_dir):
        _seed_db(tmp_log_dir)
        runner = CliRunner()
        result = runner.invoke(
            main, ["diagnose", "--log-dir", tmp_log_dir, "--run-id", "no_such_run"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output


class TestInitCommand:
    def test_init_writes_yaml(self, tmp_path):
        runner = CliRunner()
        # CliRunner.isolated_filesystem keeps test side-effects out of CWD.
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path("rlwatch.yaml").exists()
