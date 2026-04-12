"""Configuration management for rlwatch via YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EntropyCollapseConfig:
    """Configuration for entropy collapse detector."""
    enabled: bool = True
    threshold: float = 1.0
    consecutive_steps: int = 50
    warmup_steps: int = 20


@dataclass
class KLExplosionConfig:
    """Configuration for KL divergence explosion detector."""
    enabled: bool = True
    sigma_multiplier: float = 3.0
    clip_region: float = 0.2
    rolling_window: int = 100
    warmup_steps: int = 20
    # "rolling": baseline updates with the deque every step (legacy default,
    # catches sharp spikes but misses slow drift).
    # "frozen": baseline is frozen from the first ``rolling_window`` post-warmup
    # samples, mirroring ``RewardHackingDetector``. Catches drift, may produce
    # more late-run alerts.
    baseline_mode: str = "rolling"


@dataclass
class RewardHackingConfig:
    """Configuration for reward hacking proxy detector."""
    enabled: bool = True
    variance_multiplier: float = 3.0
    dip_test_significance: float = 0.05
    baseline_window: int = 100
    warmup_steps: int = 50


@dataclass
class AdvantageVarianceConfig:
    """Configuration for advantage variance spike detector."""
    enabled: bool = True
    std_multiplier: float = 3.0
    rolling_window: int = 100
    warmup_steps: int = 20
    # See KLExplosionConfig.baseline_mode for semantics.
    baseline_mode: str = "rolling"


@dataclass
class LossNaNConfig:
    """Configuration for the loss NaN/Inf detector.

    There is no rolling state — a single non-finite loss fires a critical
    alert immediately. ``warmup_steps`` is honored for consistency with the
    other detectors but defaults to 0.
    """
    enabled: bool = True
    warmup_steps: int = 0


@dataclass
class GradientNormSpikeConfig:
    """Configuration for the gradient norm spike detector.

    Z-score model identical to KL explosion. Defaults to ``baseline_mode =
    "frozen"`` because gradient norms drift slowly and the rolling baseline
    will silently follow them up.
    """
    enabled: bool = True
    sigma_multiplier: float = 3.0
    rolling_window: int = 100
    warmup_steps: int = 20
    baseline_mode: str = "frozen"


@dataclass
class RewardMeanDriftConfig:
    """Configuration for the reward mean drift detector.

    Fires when reward_mean moves monotonically in one direction for
    ``consecutive_steps`` steps, AND the total magnitude of the drift
    exceeds ``min_drift_magnitude``. Catches slow reward hacking where
    the variance doesn't spike but the mean is drifting suspiciously.
    """
    enabled: bool = True
    consecutive_steps: int = 50
    warmup_steps: int = 50
    # Minimum total drift magnitude over the window to alert. Avoids
    # firing on reward_mean that's technically monotone but nearly flat.
    min_drift_magnitude: float = 0.1


@dataclass
class SlackConfig:
    """Slack alert configuration."""
    enabled: bool = False
    webhook_url: str = ""
    channel: str = ""
    mention_users: list[str] = field(default_factory=list)


@dataclass
class EmailConfig:
    """Email alert configuration."""
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = field(default_factory=list)


@dataclass
class DiscordConfig:
    """Discord webhook alert configuration.

    Discord webhooks are HTTP POST endpoints that accept a JSON payload with
    embeds (rich blocks similar to Slack blocks). We mirror the Slack severity
    pattern: red embed + rotating-light emoji for critical, orange + warning
    for warning. Stdlib HTTP only — no extra dependencies.
    """
    enabled: bool = False
    webhook_url: str = ""
    username: str = "rlwatch"
    avatar_url: str = ""  # optional Discord avatar override
    # Role IDs to @-mention on critical alerts. Use Discord developer mode to
    # copy a role ID. Empty list = no mentions.
    mention_role_ids: list[str] = field(default_factory=list)


@dataclass
class WebhookConfig:
    """Generic HTTP webhook alert configuration.

    Universal escape hatch for any system rlwatch doesn't have a dedicated
    sender for. Posts a JSON body to ``url`` (POST or PUT, configurable).
    The body is built from a ``string.Template`` with ``${field}``
    substitutions; an empty ``template_json`` uses a sensible default.
    Substitutable fields are documented in ``docs/alerts/webhook.md``.
    """
    enabled: bool = False
    url: str = ""
    method: str = "POST"  # POST or PUT
    headers: dict[str, str] = field(default_factory=dict)
    template_json: str = ""  # empty → use _DEFAULT_WEBHOOK_TEMPLATE
    timeout_seconds: int = 10


@dataclass
class AlertConfig:
    """Alert delivery configuration."""
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    cooldown_steps: int = 100
    max_alerts_per_run: int = 50


@dataclass
class StorageConfig:
    """Metric storage configuration."""
    log_dir: str = "./rlwatch_logs"
    db_name: str = "metrics.db"


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration."""
    port: int = 8501
    host: str = "0.0.0.0"


@dataclass
class RLWatchConfig:
    """Top-level rlwatch configuration."""
    run_id: str = ""
    framework: str = "auto"  # auto, trl, verl, openrlhf, manual
    entropy_collapse: EntropyCollapseConfig = field(default_factory=EntropyCollapseConfig)
    kl_explosion: KLExplosionConfig = field(default_factory=KLExplosionConfig)
    reward_hacking: RewardHackingConfig = field(default_factory=RewardHackingConfig)
    advantage_variance: AdvantageVarianceConfig = field(default_factory=AdvantageVarianceConfig)
    loss_nan: LossNaNConfig = field(default_factory=LossNaNConfig)
    gradient_norm_spike: GradientNormSpikeConfig = field(default_factory=GradientNormSpikeConfig)
    reward_mean_drift: RewardMeanDriftConfig = field(default_factory=RewardMeanDriftConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def to_dict(self) -> dict:
        """Serialize config to dict for YAML output."""
        import dataclasses
        return dataclasses.asdict(self)


def _merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict) -> RLWatchConfig:
    """Convert a flat/nested dict to RLWatchConfig."""
    cfg = RLWatchConfig()

    if "run_id" in data:
        cfg.run_id = data["run_id"]
    if "framework" in data:
        cfg.framework = data["framework"]

    if "entropy_collapse" in data:
        for k, v in data["entropy_collapse"].items():
            setattr(cfg.entropy_collapse, k, v)

    if "kl_explosion" in data:
        for k, v in data["kl_explosion"].items():
            setattr(cfg.kl_explosion, k, v)

    if "reward_hacking" in data:
        for k, v in data["reward_hacking"].items():
            setattr(cfg.reward_hacking, k, v)

    if "advantage_variance" in data:
        for k, v in data["advantage_variance"].items():
            setattr(cfg.advantage_variance, k, v)

    if "loss_nan" in data:
        for k, v in data["loss_nan"].items():
            setattr(cfg.loss_nan, k, v)

    if "gradient_norm_spike" in data:
        for k, v in data["gradient_norm_spike"].items():
            setattr(cfg.gradient_norm_spike, k, v)

    if "reward_mean_drift" in data:
        for k, v in data["reward_mean_drift"].items():
            setattr(cfg.reward_mean_drift, k, v)

    if "alerts" in data:
        alerts = data["alerts"]
        if "slack" in alerts:
            for k, v in alerts["slack"].items():
                setattr(cfg.alerts.slack, k, v)
        if "email" in alerts:
            for k, v in alerts["email"].items():
                setattr(cfg.alerts.email, k, v)
        if "discord" in alerts:
            for k, v in alerts["discord"].items():
                setattr(cfg.alerts.discord, k, v)
        if "webhook" in alerts:
            for k, v in alerts["webhook"].items():
                setattr(cfg.alerts.webhook, k, v)
        if "cooldown_steps" in alerts:
            cfg.alerts.cooldown_steps = alerts["cooldown_steps"]
        if "max_alerts_per_run" in alerts:
            cfg.alerts.max_alerts_per_run = alerts["max_alerts_per_run"]

    if "storage" in data:
        for k, v in data["storage"].items():
            setattr(cfg.storage, k, v)

    if "dashboard" in data:
        for k, v in data["dashboard"].items():
            setattr(cfg.dashboard, k, v)

    return cfg


def load_config(
    config_path: Optional[str | Path] = None,
    **overrides,
) -> RLWatchConfig:
    """Load configuration from YAML file with optional overrides.

    Resolution order:
    1. Default values
    2. YAML config file (if provided or found at ./rlwatch.yaml)
    3. Environment variables (RLWATCH_SLACK_WEBHOOK_URL, etc.)
    4. Keyword overrides
    """
    data: dict = {}

    # Try to load YAML config
    if config_path is None:
        # Check default locations
        for candidate in ["rlwatch.yaml", "rlwatch.yml", ".rlwatch.yaml"]:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    data = loaded

    # Apply env var overrides
    env_map = {
        "RLWATCH_SLACK_WEBHOOK_URL": ("alerts", "slack", "webhook_url"),
        "RLWATCH_SLACK_CHANNEL": ("alerts", "slack", "channel"),
        "RLWATCH_SMTP_HOST": ("alerts", "email", "smtp_host"),
        "RLWATCH_SMTP_PORT": ("alerts", "email", "smtp_port"),
        "RLWATCH_SMTP_USER": ("alerts", "email", "smtp_user"),
        "RLWATCH_SMTP_PASSWORD": ("alerts", "email", "smtp_password"),
        "RLWATCH_EMAIL_FROM": ("alerts", "email", "from_addr"),
        "RLWATCH_EMAIL_TO": ("alerts", "email", "to_addrs"),
        "RLWATCH_RUN_ID": ("run_id",),
        "RLWATCH_LOG_DIR": ("storage", "log_dir"),
        "RLWATCH_FRAMEWORK": ("framework",),
        "RLWATCH_DISCORD_WEBHOOK_URL": ("alerts", "discord", "webhook_url"),
        "RLWATCH_WEBHOOK_URL": ("alerts", "webhook", "url"),
        "RLWATCH_WEBHOOK_TEMPLATE": ("alerts", "webhook", "template_json"),
    }

    for env_key, path_parts in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            # Build nested dict from path parts
            d = data
            for part in path_parts[:-1]:
                d = d.setdefault(part, {})
            key = path_parts[-1]
            # Type coercion
            if key == "smtp_port":
                val = int(val)
            elif key == "to_addrs":
                val = [a.strip() for a in val.split(",")]
            d[key] = val

    # Enable Slack if webhook URL is configured
    slack_url = data.get("alerts", {}).get("slack", {}).get("webhook_url", "")
    if slack_url:
        data.setdefault("alerts", {}).setdefault("slack", {})["enabled"] = True

    # Enable email if to_addrs is configured
    to_addrs = data.get("alerts", {}).get("email", {}).get("to_addrs", [])
    if to_addrs:
        data.setdefault("alerts", {}).setdefault("email", {})["enabled"] = True

    # Enable Discord if webhook URL is configured
    discord_url = data.get("alerts", {}).get("discord", {}).get("webhook_url", "")
    if discord_url:
        data.setdefault("alerts", {}).setdefault("discord", {})["enabled"] = True

    # Enable generic webhook if URL is configured
    webhook_url = data.get("alerts", {}).get("webhook", {}).get("url", "")
    if webhook_url:
        data.setdefault("alerts", {}).setdefault("webhook", {})["enabled"] = True

    cfg = _dict_to_config(data)

    # Apply direct overrides. If a kwarg targets a sub-dataclass and the value
    # is a dict, deep-merge the dict's keys onto the existing dataclass instead
    # of replacing the dataclass with a bare dict (which would silently break
    # downstream attribute access).
    import dataclasses as _dc
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            continue
        existing = getattr(cfg, k)
        if isinstance(v, dict) and _dc.is_dataclass(existing):
            for inner_k, inner_v in v.items():
                if hasattr(existing, inner_k):
                    setattr(existing, inner_k, inner_v)
                else:
                    raise ValueError(
                        f"Unknown key '{inner_k}' for config section '{k}'"
                    )
        else:
            setattr(cfg, k, v)

    return cfg
