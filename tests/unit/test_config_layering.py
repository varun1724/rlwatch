"""Unit tests for the config layering: defaults < yaml < env < kwargs."""

from __future__ import annotations

import yaml

from rlwatch.config import RLWatchConfig, load_config


class TestEnvOverrides:
    def test_framework_env_var(self, monkeypatch):
        monkeypatch.setenv("RLWATCH_FRAMEWORK", "verl")
        cfg = load_config(config_path="/nonexistent/path.yaml")
        assert cfg.framework == "verl"

    def test_smtp_port_type_coerced(self, monkeypatch):
        monkeypatch.setenv("RLWATCH_SMTP_PORT", "2525")
        cfg = load_config(config_path="/nonexistent/path.yaml")
        assert cfg.alerts.email.smtp_port == 2525
        assert isinstance(cfg.alerts.email.smtp_port, int)

    def test_to_addrs_split(self, monkeypatch):
        monkeypatch.setenv("RLWATCH_EMAIL_TO", "a@x.com, b@x.com,c@x.com")
        cfg = load_config(config_path="/nonexistent/path.yaml")
        assert cfg.alerts.email.to_addrs == ["a@x.com", "b@x.com", "c@x.com"]
        assert cfg.alerts.email.enabled is True


class TestNestedKwargOverrides:
    def test_nested_dict_deep_merges(self):
        """Passing a nested dict must update fields, not replace the dataclass."""
        cfg = load_config(
            config_path="/nonexistent/path.yaml",
            entropy_collapse={"threshold": 0.42, "consecutive_steps": 7},
        )
        # The sub-dataclass survives — replacing it with a dict would lose the
        # other defaults (warmup_steps, enabled, etc.).
        assert hasattr(cfg.entropy_collapse, "threshold")
        assert cfg.entropy_collapse.threshold == 0.42
        assert cfg.entropy_collapse.consecutive_steps == 7
        # Untouched fields keep their defaults.
        assert cfg.entropy_collapse.enabled is True
        assert cfg.entropy_collapse.warmup_steps == 20

    def test_unknown_nested_key_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown key"):
            load_config(
                config_path="/nonexistent/path.yaml",
                entropy_collapse={"nonexistent_field": 1},
            )

    def test_top_level_kwarg_still_works(self):
        cfg = load_config(config_path="/nonexistent/path.yaml", run_id="my_run")
        assert cfg.run_id == "my_run"


class TestYAMLLoading:
    def test_yaml_loss_nan_section(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(yaml.dump({"loss_nan": {"enabled": False}}))
        cfg = load_config(config_path=str(f))
        assert cfg.loss_nan.enabled is False

    def test_yaml_grad_norm_section(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(
            yaml.dump(
                {
                    "gradient_norm_spike": {
                        "sigma_multiplier": 5.0,
                        "baseline_mode": "rolling",
                    }
                }
            )
        )
        cfg = load_config(config_path=str(f))
        assert cfg.gradient_norm_spike.sigma_multiplier == 5.0
        assert cfg.gradient_norm_spike.baseline_mode == "rolling"

    def test_yaml_partial_override_preserves_defaults(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text(yaml.dump({"entropy_collapse": {"threshold": 0.7}}))
        cfg = load_config(config_path=str(f))
        assert cfg.entropy_collapse.threshold == 0.7
        assert cfg.entropy_collapse.warmup_steps == 20  # default kept


class TestDefaults:
    def test_loss_nan_default(self):
        cfg = RLWatchConfig()
        assert cfg.loss_nan.enabled is True
        assert cfg.loss_nan.warmup_steps == 0

    def test_gradient_norm_default_baseline_mode_frozen(self):
        cfg = RLWatchConfig()
        assert cfg.gradient_norm_spike.baseline_mode == "frozen"

    def test_kl_default_baseline_mode_rolling(self):
        cfg = RLWatchConfig()
        assert cfg.kl_explosion.baseline_mode == "rolling"
