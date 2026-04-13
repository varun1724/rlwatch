"""OpenRLHF logger integration for rlwatch.

OpenRLHF does not expose a callback/hook API. Instead it uses direct logger
classes (``WandbLogger``, ``TensorboardLogger``) that are called from the
training loop's ``save_logs_and_checkpoints()`` method. We implement a
logger-compatible class that intercepts ``log_train(global_step, logs_dict)``
and forwards mapped metrics to ``RLWatch.log_step()``.

Integration path (in ``core.py::_attach_openrlhf``):
1. Import this module.
2. Instantiate ``RLWatchOpenRLHFLogger(monitor)`` bound to the active monitor.
3. Inject it into the trainer alongside the user's existing loggers.

Metric mapping (OpenRLHF → rlwatch):
    entropy_loss        → entropy (sign-inverted if negative)
    kl / ppo_kl         → kl_divergence
    reward_mean / rollout/reward_mean → reward_mean
    reward_std / rollout/reward_std   → reward_std
    policy_loss         → loss
    actor_grad_norm     → grad_norm
    advantage_std       → advantage_std
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlwatch.core import RLWatch

logger = logging.getLogger("rlwatch.integrations.openrlhf")

# OpenRLHF metric name → rlwatch kwarg name. First hit wins per field.
_METRIC_MAP: dict[str, list[str]] = {
    "entropy": ["entropy_loss", "entropy", "policy_entropy"],
    "kl_divergence": ["kl", "ppo_kl", "kl_divergence", "vllm_kl"],
    "reward_mean": [
        "rollout/reward_mean", "reward_mean", "reward",
    ],
    "reward_std": [
        "rollout/reward_std", "reward_std",
    ],
    "loss": ["policy_loss", "loss"],
    "grad_norm": ["actor_grad_norm", "grad_norm", "gradient_norm"],
    "advantage_std": ["advantage_std", "advantages_std"],
}


def _map_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Map an OpenRLHF metrics dict to rlwatch kwarg names.

    Returns only the fields that had a match — unrecognized metrics are
    silently dropped.
    """
    mapped: dict[str, float] = {}
    for rlwatch_name, candidates in _METRIC_MAP.items():
        for candidate in candidates:
            if candidate in data:
                try:
                    val = float(data[candidate])
                except (TypeError, ValueError):
                    continue
                # OpenRLHF logs entropy_loss as a negative value (it's a
                # loss term, not raw entropy). If the matched key is
                # "entropy_loss" and the value is negative, flip the sign
                # so rlwatch sees positive entropy.
                if candidate == "entropy_loss" and val < 0:
                    val = -val
                mapped[rlwatch_name] = val
                break
    return mapped


class RLWatchOpenRLHFLogger:
    """An OpenRLHF-compatible logger that forwards metrics to rlwatch.

    Duck-types the interface that OpenRLHF's ``BasePPOTrainer`` expects
    from its loggers: ``log_train(global_step, logs_dict)`` and
    ``log_eval(global_step, logs_dict)``.
    """

    def __init__(self, monitor: "RLWatch"):
        self._monitor = monitor

    def log_train(self, global_step: int, logs_dict: dict[str, Any]) -> None:
        """Called by OpenRLHF on every training logging step.

        Maps OpenRLHF metric names to rlwatch names and forwards to
        ``monitor.log_step()``.
        """
        metrics = _map_metrics(logs_dict)
        if metrics:
            self._monitor.log_step(global_step, **metrics)

    def log_eval(self, global_step: int, logs_dict: dict[str, Any]) -> None:
        """Called by OpenRLHF on evaluation steps. Currently a no-op.

        Evaluation metrics (pass@k, response length) are not relevant to
        rlwatch's training instability detectors. We accept the call so
        the logger doesn't crash if OpenRLHF invokes it.
        """

    def close(self) -> None:
        """Called when training ends."""

    def finish(self) -> None:
        """Alias for close in some OpenRLHF versions."""
