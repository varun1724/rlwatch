"""Core rlwatch API — attach(), log_step(), and the RLWatch monitor.

Usage:
    import rlwatch
    rlwatch.attach()  # Auto-detect framework and start monitoring

Or with manual metric logging:
    monitor = rlwatch.attach(framework="manual")
    for step in range(num_steps):
        ...
        monitor.log_step(step, entropy=ent, kl_divergence=kl, ...)
"""

from __future__ import annotations

import atexit
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from rlwatch.alerts import AlertManager
from rlwatch.config import RLWatchConfig, load_config
from rlwatch.detectors import DetectorSuite
from rlwatch.storage import MetricStore

logger = logging.getLogger("rlwatch")

# Global singleton monitor
_global_monitor: Optional[RLWatch] = None


class RLWatch:
    """Main monitoring class that orchestrates detection, storage, and alerting."""

    def __init__(self, config: RLWatchConfig):
        self.config = config

        # Generate run ID if not set
        if not config.run_id:
            config.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self.run_id = config.run_id
        self._detectors = DetectorSuite(config)
        self._store = MetricStore(config)
        self._alerts = AlertManager(config.alerts, run_id=config.run_id)
        self._step_count = 0
        self._started = False

        # Register the run
        self._store.register_run(config)

    def start(self):
        """Mark monitoring as active."""
        self._started = True
        logger.info(
            "rlwatch started — run_id=%s, log_dir=%s",
            self.run_id,
            self.config.storage.log_dir,
        )

    def log_step(
        self,
        step: int,
        *,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None,
        rewards: Optional[np.ndarray] = None,
        advantage_std: Optional[float] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        grad_norm: Optional[float] = None,
        **extra_metrics,
    ):
        """Log metrics for a training step and run all detectors.

        This is the primary API for feeding metrics into rlwatch. Called either
        by framework callbacks (TRL, veRL, OpenRLHF) or directly by the user.
        """
        self._step_count += 1

        # Compute reward stats from raw rewards if not provided
        if rewards is not None and reward_mean is None:
            reward_mean = float(np.mean(rewards))
        if rewards is not None and reward_std is None:
            reward_std = float(np.std(rewards))

        # Persist metrics
        self._store.log_metrics(
            step,
            entropy=entropy,
            kl_divergence=kl_divergence,
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_min=float(np.min(rewards)) if rewards is not None else None,
            reward_max=float(np.max(rewards)) if rewards is not None else None,
            advantage_std=advantage_std,
            loss=loss,
            learning_rate=learning_rate,
            clip_fraction=clip_fraction,
            grad_norm=grad_norm,
            **extra_metrics,
        )

        # Run detectors
        alerts = self._detectors.check_step(
            step,
            entropy=entropy,
            kl_divergence=kl_divergence,
            reward_mean=reward_mean,
            reward_std=reward_std,
            rewards=rewards,
            advantage_std=advantage_std,
            loss=loss,
            grad_norm=grad_norm,
        )

        # Deliver alerts (only persist those that pass cooldown/rate limits)
        sent_alerts = []
        for alert in alerts:
            was_sent = self._alerts.send(alert)
            if was_sent:
                self._store.log_alert(
                    step=alert.step,
                    detector=alert.detector,
                    severity=alert.severity,
                    message=alert.message,
                    metric_values=alert.metric_values,
                    recommendation=alert.recommendation,
                )
                sent_alerts.append(alert)

        return sent_alerts

    @property
    def store(self) -> MetricStore:
        return self._store

    @property
    def alert_manager(self) -> AlertManager:
        return self._alerts

    def attach_to_trainer(self, trainer) -> None:
        """Register the rlwatch TRL callback on an active HuggingFace Trainer.

        Use this when ``attach()`` was called before the ``Trainer`` was
        constructed (the most common case for two-line-attach users), or when
        you want to make the wiring explicit.

        Example:
            monitor = rlwatch.attach()
            trainer = SFTTrainer(...)
            monitor.attach_to_trainer(trainer)
        """
        callback_cls = getattr(self, "_trl_callback_class", None)
        if callback_cls is None:
            # Build the callback lazily so users who never touch TRL never pay
            # the import cost.
            callback_cls = _build_trl_callback(self)
            self._trl_callback_class = callback_cls
        trainer.add_callback(callback_cls())
        logger.info("rlwatch attached to TRL Trainer via callback")

    def stop(self):
        """Stop monitoring and clean up."""
        if self._started:
            logger.info(
                "rlwatch stopped — run_id=%s, steps=%d, alerts=%d",
                self.run_id,
                self._step_count,
                self._alerts.total_alerts_sent,
            )
            self._started = False
            self._store.close()


def attach(
    config_path: Optional[str] = None,
    framework: str = "auto",
    run_id: str = "",
    trainer=None,
    **kwargs,
) -> RLWatch:
    """Attach rlwatch to the current training process.

    Two-line usage:
        import rlwatch
        rlwatch.attach()

    Args:
        config_path: Path to rlwatch.yaml config file.
        framework: Framework to integrate with: "auto", "trl", "verl", "openrlhf", or "manual".
        run_id: Optional run identifier. Auto-generated if not provided.
        trainer: Optional HuggingFace TRL ``Trainer`` instance. When provided
            the rlwatch callback is registered on it directly — this is the
            recommended path for TRL users and avoids the fragile process-wide
            object scan we used to do.
        **kwargs: Override any config values.

    Returns:
        RLWatch monitor instance.
    """
    global _global_monitor

    config = load_config(config_path, **kwargs)
    if run_id:
        config.run_id = run_id
    if framework != "auto":
        config.framework = framework

    # Auto-detect framework
    if config.framework == "auto":
        config.framework = _detect_framework()

    monitor = RLWatch(config)
    monitor.start()

    # Install framework-specific hooks
    if config.framework == "trl":
        _attach_trl(monitor, trainer=trainer)
    elif config.framework == "verl":
        _attach_verl(monitor)
    elif config.framework == "openrlhf":
        _attach_openrlhf(monitor)
    # "manual" means the user will call log_step() themselves

    _global_monitor = monitor

    # Register cleanup on exit
    atexit.register(monitor.stop)

    return monitor


def get_monitor() -> Optional[RLWatch]:
    """Get the global rlwatch monitor instance."""
    return _global_monitor


def log_step(step: int, **kwargs):
    """Convenience function to log a step on the global monitor."""
    if _global_monitor is None:
        raise RuntimeError(
            "rlwatch not attached. Call rlwatch.attach() first."
        )
    return _global_monitor.log_step(step, **kwargs)


def _detect_framework() -> str:
    """Auto-detect which RL framework is in use."""
    import sys

    # Check for TRL
    if "trl" in sys.modules:
        try:
            import trl
            return "trl"
        except ImportError:
            pass

    # Check for veRL
    if "verl" in sys.modules:
        return "verl"

    # Check for OpenRLHF
    if "openrlhf" in sys.modules:
        return "openrlhf"

    return "manual"


def _build_trl_callback(monitor: "RLWatch"):
    """Build the rlwatch TRL ``TrainerCallback`` class bound to a monitor.

    Imported lazily so users who never touch TRL never pay the import cost.
    Raises ``ImportError`` if transformers isn't installed.
    """
    from transformers import TrainerCallback

    class RLWatchTRLCallback(TrainerCallback):
        """TRL/Transformers trainer callback for rlwatch."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return

            step = state.global_step
            metrics: dict = {}

            # Map TRL metric names to rlwatch names. Listed in priority order
            # — first hit wins.
            metric_map = {
                "entropy": ["entropy", "policy_entropy", "approx_entropy"],
                "kl_divergence": ["kl", "kl_divergence", "kl_penalty", "approx_kl"],
                "reward_mean": ["reward", "reward_mean", "rewards/mean", "env/reward_mean"],
                "reward_std": ["reward_std", "rewards/std", "env/reward_std"],
                "advantage_std": ["advantage_std", "advantages_std"],
                "loss": ["loss", "train_loss", "policy_loss"],
                "learning_rate": ["learning_rate", "lr"],
                "clip_fraction": ["clip_fraction", "clipfrac", "policy/clip_fraction"],
                "grad_norm": ["grad_norm", "gradient_norm", "grads/norm"],
            }

            for rlwatch_name, candidates in metric_map.items():
                for candidate in candidates:
                    if candidate in logs:
                        metrics[rlwatch_name] = logs[candidate]
                        break

            if metrics:
                monitor.log_step(step, **metrics)

    return RLWatchTRLCallback


def _attach_trl(monitor: RLWatch, trainer=None):
    """Attach to HuggingFace TRL via TrainerCallback.

    If ``trainer`` is provided, register the callback directly. Otherwise stash
    a callback class on the monitor and tell the user how to wire it up. We
    deliberately do not walk the live object graph looking for a Trainer —
    that approach was slow, fragile, and depended on attach() being called
    *after* Trainer construction in just the right place. See BUILD_DECISIONS.md
    for the full rationale.
    """
    try:
        callback_cls = _build_trl_callback(monitor)
    except ImportError:
        logger.warning("TRL/transformers not found. Falling back to manual mode.")
        monitor.config.framework = "manual"
        return

    monitor._trl_callback_class = callback_cls

    if trainer is not None:
        trainer.add_callback(callback_cls())
        logger.info("rlwatch attached to TRL Trainer via callback")
        return

    logger.info(
        "rlwatch TRL callback ready. After constructing your Trainer, call:\n"
        "  monitor.attach_to_trainer(trainer)\n"
        "or pass trainer=trainer to rlwatch.attach()."
    )


def _attach_verl(monitor: RLWatch):
    """Attach to veRL framework."""
    try:
        # veRL uses a different callback mechanism
        # We monkey-patch the common metric logging points
        import verl

        logger.info(
            "rlwatch ready for veRL. Use monitor.log_step() in your training loop "
            "or pass metrics from veRL's callback system."
        )
    except ImportError:
        logger.warning("veRL not found. Falling back to manual mode.")
        monitor.config.framework = "manual"


def _attach_openrlhf(monitor: RLWatch):
    """Attach to OpenRLHF framework."""
    try:
        import openrlhf

        logger.info(
            "rlwatch ready for OpenRLHF. Use monitor.log_step() in your training loop "
            "or integrate with OpenRLHF's callback system."
        )
    except ImportError:
        logger.warning("OpenRLHF not found. Falling back to manual mode.")
        monitor.config.framework = "manual"
