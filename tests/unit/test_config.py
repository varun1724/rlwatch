"""Tests for rlwatch configuration."""

import os
import tempfile
from pathlib import Path

import yaml
import pytest

from rlwatch.config import RLWatchConfig, load_config


class TestLoadConfig:
    def test_default_config(self):
        config = load_config(config_path="/nonexistent/path.yaml")
        assert config.entropy_collapse.enabled is True
        assert config.entropy_collapse.threshold == 1.0
        assert config.kl_explosion.sigma_multiplier == 3.0
        assert config.alerts.slack.enabled is False

    def test_yaml_override(self, tmp_path):
        config_data = {
            "entropy_collapse": {"threshold": 0.5, "consecutive_steps": 30},
            "alerts": {"slack": {"webhook_url": "https://hooks.slack.com/test"}},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path=str(config_file))
        assert config.entropy_collapse.threshold == 0.5
        assert config.entropy_collapse.consecutive_steps == 30
        assert config.alerts.slack.enabled is True  # Auto-enabled when URL is set

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("RLWATCH_SLACK_WEBHOOK_URL", "https://hooks.slack.com/env")
        config = load_config(config_path="/nonexistent/path.yaml")
        assert config.alerts.slack.webhook_url == "https://hooks.slack.com/env"
        assert config.alerts.slack.enabled is True

    def test_run_id_override(self):
        config = load_config(config_path="/nonexistent/path.yaml", run_id="my_run")
        assert config.run_id == "my_run"

    def test_to_dict(self):
        config = RLWatchConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "entropy_collapse" in d
        assert d["entropy_collapse"]["threshold"] == 1.0


class TestRLWatchConfig:
    def test_default_values(self):
        config = RLWatchConfig()
        assert config.framework == "auto"
        assert config.storage.log_dir == "./rlwatch_logs"
        assert config.dashboard.port == 8501
