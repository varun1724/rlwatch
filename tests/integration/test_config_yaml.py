"""Integration tests for YAML config loading + env var cascade."""

from __future__ import annotations

import yaml
import pytest

from rlwatch.config import load_config

pytestmark = pytest.mark.integration


class TestYAMLCascade:
    def test_default_when_yaml_missing(self):
        cfg = load_config(config_path="/definitely/not/here.yaml")
        assert cfg.entropy_collapse.threshold == 1.0
        assert cfg.framework == "auto"

    def test_yaml_overrides_default(self, tmp_path):
        f = tmp_path / "rlwatch.yaml"
        f.write_text(yaml.dump({
            "entropy_collapse": {"threshold": 0.7},
            "framework": "manual",
        }))
        cfg = load_config(config_path=str(f))
        assert cfg.entropy_collapse.threshold == 0.7
        assert cfg.framework == "manual"

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        f = tmp_path / "rlwatch.yaml"
        f.write_text(yaml.dump({"framework": "trl"}))
        monkeypatch.setenv("RLWATCH_FRAMEWORK", "manual")
        cfg = load_config(config_path=str(f))
        assert cfg.framework == "manual"

    def test_kwarg_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLWATCH_RUN_ID", "from_env")
        cfg = load_config(config_path="/none.yaml", run_id="from_kwarg")
        assert cfg.run_id == "from_kwarg"

    def test_full_cascade(self, tmp_path, monkeypatch):
        f = tmp_path / "rlwatch.yaml"
        f.write_text(yaml.dump({
            "entropy_collapse": {"threshold": 0.5, "consecutive_steps": 25},
        }))
        # env var changes the SMTP host (a yaml-and-env field).
        monkeypatch.setenv("RLWATCH_SMTP_HOST", "smtp.env.test")
        cfg = load_config(
            config_path=str(f),
            entropy_collapse={"threshold": 0.9},  # kwarg deep-merge
        )
        assert cfg.entropy_collapse.threshold == 0.9       # kwarg won
        assert cfg.entropy_collapse.consecutive_steps == 25  # yaml preserved
        assert cfg.alerts.email.smtp_host == "smtp.env.test"
