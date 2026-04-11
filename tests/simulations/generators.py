"""Deterministic metric-trace generators for Tier 3 (simulation/golden) tests.

Each generator returns a list of ``dict`` rows that can be fed straight into
``RLWatch.log_step``. They're pure functions of their kwargs and a seed, so
the resulting trace is the regression fixture — checked-in JSON would be the
same data, just less convenient to update when the simulation logic itself
needs to evolve.

Convention for every generator:
- accept ``n_steps`` (int) and ``seed`` (int) at minimum
- yield a list of dicts
- each dict has at least a ``step`` key plus whatever metrics that scenario
  produces

Adding a new fixture? Write a generator here, register it in
``test_replay.py::FIXTURES``, and add the expected alert tuples there too.
"""

from __future__ import annotations

import math
import random


def _seeded(seed: int):
    """Return a seeded ``random.Random`` so generators don't pollute global state."""
    return random.Random(seed)


def healthy_run(n_steps: int = 200, seed: int = 0) -> list[dict]:
    """A boring, well-behaved training run. Should produce zero alerts.

    Intentionally low-noise: with tight std on metrics whose detectors use a
    rolling z-score, even small "natural" excursions become 3σ outliers
    because the baseline std is also small. Keeping the noise structured so
    the std is meaningful avoids spurious warnings without faking the data.
    """
    rng = _seeded(seed)
    rows = []
    for step in range(n_steps):
        # Use sinusoidal jitter rather than half-normal noise so the rolling
        # std is non-degenerate and natural variation doesn't look like a 3σ
        # event. Real training metrics behave this way too — they oscillate.
        wobble = (step % 7) - 3.0  # -3..3 sawtooth
        rows.append({
            "step": step,
            "entropy": 2.5 + 0.03 * wobble,
            "kl_divergence": 0.02 + 0.003 * wobble,
            "reward_mean": -1.5 + step / n_steps + 0.05 * wobble,
            "reward_std": 0.5 + 0.02 * wobble,
            "advantage_std": 1.0 + 0.05 * wobble,
            "loss": max(0.05, 0.5 - 0.001 * step + 0.01 * wobble),
            "grad_norm": 1.0 + 0.1 * wobble,
        })
    return rows


def entropy_collapse(
    n_steps: int = 400, collapse_start: int = 200, seed: int = 0
) -> list[dict]:
    """Healthy first half, entropy collapse second half."""
    rng = _seeded(seed)
    rows = []
    for step in range(n_steps):
        if step < collapse_start:
            entropy = 2.8 - (step / n_steps) * 0.5 + rng.gauss(0, 0.02)
        else:
            decay = math.exp(-(step - collapse_start) / 25)
            entropy = max(0.05, 2.8 * decay * 0.3 + rng.gauss(0, 0.01))
        rows.append({
            "step": step,
            "entropy": entropy,
            "kl_divergence": 0.01,
            "reward_std": 0.5,
            "advantage_std": 1.0,
            "loss": max(0.05, 0.4 - 0.0005 * step),
            "grad_norm": 1.0,
        })
    return rows


def kl_spike(n_steps: int = 200, spike_at: int = 150, seed: int = 0) -> list[dict]:
    """Healthy KL with one massive single-step spike."""
    rng = _seeded(seed)
    rows = []
    for step in range(n_steps):
        if step == spike_at:
            kl = 5.0
        else:
            kl = 0.01 + abs(rng.gauss(0, 0.0005))
        rows.append({
            "step": step,
            "entropy": 2.5,
            "kl_divergence": kl,
            "reward_std": 0.5,
            "advantage_std": 1.0,
            "loss": 0.3,
            "grad_norm": 1.0,
        })
    return rows


def reward_variance_explosion(
    n_steps: int = 200, explosion_at: int = 120, seed: int = 0
) -> list[dict]:
    """Reward std jumps 10x at a fixed step."""
    rng = _seeded(seed)
    rows = []
    for step in range(n_steps):
        if step < explosion_at:
            reward_std = 0.5 + abs(rng.gauss(0, 0.02))
        else:
            reward_std = 5.0 + abs(rng.gauss(0, 0.5))
        rows.append({
            "step": step,
            "entropy": 2.5,
            "kl_divergence": 0.01,
            "reward_std": reward_std,
            "advantage_std": 1.0,
            "loss": 0.3,
            "grad_norm": 1.0,
        })
    return rows


def loss_nan_at(n_steps: int = 200, nan_at: int = 100, seed: int = 0) -> list[dict]:
    """Healthy loss until ``nan_at``, then NaN."""
    rows = []
    for step in range(n_steps):
        loss = float("nan") if step >= nan_at else max(0.05, 0.5 - 0.001 * step)
        rows.append({
            "step": step,
            "entropy": 2.5,
            "kl_divergence": 0.01,
            "reward_std": 0.5,
            "advantage_std": 1.0,
            "loss": loss,
            "grad_norm": 1.0,
        })
    return rows


def gradient_norm_spike(
    n_steps: int = 200, spike_at: int = 150, seed: int = 0
) -> list[dict]:
    """Healthy grad_norm with a 10x spike at a fixed step.

    Stays high after the spike so the rolling-window detector still has a
    z-score gap to fire on; otherwise the very next step would be back to
    baseline and the alert window is exactly 1 step wide.
    """
    rng = _seeded(seed)
    rows = []
    for step in range(n_steps):
        if step >= spike_at:
            grad_norm = 10.0 + abs(rng.gauss(0, 0.5))
        else:
            grad_norm = 1.0 + abs(rng.gauss(0, 0.05))
        rows.append({
            "step": step,
            "entropy": 2.5,
            "kl_divergence": 0.01,
            "reward_std": 0.5,
            "advantage_std": 1.0,
            "loss": 0.3,
            "grad_norm": grad_norm,
        })
    return rows
