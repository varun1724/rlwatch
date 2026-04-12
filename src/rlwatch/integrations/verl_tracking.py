"""veRL tracking backend for rlwatch.

veRL does not expose a callback/hook API like TRL's ``TrainerCallback``.
Instead it uses a ``Tracking`` logger class that wraps multiple backends
(WandB, MLflow, TensorBoard, etc.). We implement a custom tracking backend
that intercepts metrics at the ``log(data, step)`` call and forwards them
to ``RLWatch.log_step()``.

Integration path (in ``core.py::_attach_verl``):
1. Import this module.
2. Instantiate ``RLWatchVerLTracker(monitor)`` bound to the active monitor.
3. Register it with veRL's ``Tracking`` class so it receives every
   ``log(data, step)`` call alongside the user's other configured backends.

Metric mapping (veRL → rlwatch):
    actor/entropy          → entropy
    actor/kl_divergence    → kl_divergence
    training/kl_penalty    → kl_divergence  (fallback)
    rewards/mean           → reward_mean
    rewards/std            → reward_std
    training/policy_loss   → loss
    loss                   → loss  (fallback)
    training/grad_norm     → grad_norm
    grad_norm              → grad_norm  (fallback)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlwatch.core import RLWatch

logger = logging.getLogger("rlwatch.integrations.verl")

# veRL metric name → rlwatch kwarg name. Listed in priority order per
# rlwatch field — first hit wins (same pattern as the TRL callback).
_METRIC_MAP: dict[str, list[str]] = {
    "entropy": ["actor/entropy", "entropy", "policy_entropy"],
    "kl_divergence": [
        "actor/kl_divergence",
        "training/kl_penalty",
        "kl",
        "kl_divergence",
    ],
    "reward_mean": ["rewards/mean", "reward_mean", "reward"],
    "reward_std": ["rewards/std", "reward_std"],
    "loss": ["training/policy_loss", "policy_loss", "loss"],
    "grad_norm": ["training/grad_norm", "grad_norm", "gradient_norm"],
    "advantage_std": ["advantage_std", "advantages_std"],
}


def _map_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Map a veRL metrics dict to rlwatch kwarg names.

    Returns only the fields that had a match — unrecognized metrics are
    silently dropped (they're not rlwatch's concern).
    """
    mapped: dict[str, float] = {}
    for rlwatch_name, candidates in _METRIC_MAP.items():
        for candidate in candidates:
            if candidate in data:
                try:
                    mapped[rlwatch_name] = float(data[candidate])
                except (TypeError, ValueError):
                    continue
                break
    return mapped


class RLWatchVerLTracker:
    """A veRL-compatible tracking backend that forwards metrics to rlwatch.

    This class duck-types the interface that veRL's ``Tracking`` class
    expects from its backends: a ``log(data, step, **kwargs)`` method.
    """

    def __init__(self, monitor: "RLWatch"):
        self._monitor = monitor

    def log(self, data: dict[str, Any], step: int, **kwargs) -> None:
        """Called by veRL's Tracking on every training step.

        Maps veRL metric names to rlwatch names and forwards to
        ``monitor.log_step()``.
        """
        metrics = _map_metrics(data)
        if metrics:
            self._monitor.log_step(step, **metrics)

    # veRL's Tracking may also call these lifecycle methods. Implement
    # as no-ops so our tracker doesn't crash if veRL evolves its backend
    # interface.
    def finish(self) -> None:
        """Called when training ends."""

    def close(self) -> None:
        """Alias for finish in some veRL versions."""
