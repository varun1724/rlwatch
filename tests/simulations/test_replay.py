"""Tier 3: simulation/golden replay tests.

For each canned scenario, replay the trace through the full pipeline
(``DetectorSuite`` + ``MetricStore`` + ``AlertManager``) and assert the
fired-alert set matches the expected (detector, severity) pairs.

Adding a new fixture: write a generator in ``generators.py``, append a tuple
to ``FIXTURES`` below, and the parametrize loop picks it up automatically.
"""

from __future__ import annotations

import pytest

from rlwatch.config import RLWatchConfig
from rlwatch.core import RLWatch

from . import generators

# (name, generator_fn, generator_kwargs, expected_detectors_with_severity)
#
# expected_detectors_with_severity is a set of (detector_name, severity)
# tuples — we don't pin exact step numbers because detector heuristics may
# legitimately drift by a few steps without breaking correctness. We DO pin
# the set of detectors that fire and at what severity.
FIXTURES = [
    (
        "healthy_run",
        generators.healthy_run,
        {"n_steps": 200, "seed": 0},
        set(),  # zero alerts expected
    ),
    (
        "entropy_collapse",
        generators.entropy_collapse,
        {"n_steps": 400, "collapse_start": 200, "seed": 0},
        {("entropy_collapse", "warning"), ("entropy_collapse", "critical")},
    ),
    (
        "kl_spike",
        generators.kl_spike,
        {"n_steps": 200, "spike_at": 150, "seed": 0},
        # The warning fires before the critical because the spike is large
        # enough that the same step crosses both thresholds and they're tracked
        # independently by AlertManager.
        {("kl_explosion", "warning"), ("kl_explosion", "critical")},
    ),
    (
        "reward_variance_explosion",
        generators.reward_variance_explosion,
        {"n_steps": 200, "explosion_at": 120, "seed": 0},
        {("reward_hacking", "critical")},
    ),
    (
        "loss_nan",
        generators.loss_nan_at,
        {"n_steps": 200, "nan_at": 100, "seed": 0},
        {("loss_nan", "critical")},
    ),
    (
        "gradient_norm_spike",
        generators.gradient_norm_spike,
        {"n_steps": 200, "spike_at": 150, "seed": 0},
        # Warning fires from the rolling fallback path before the frozen
        # baseline locks; critical fires once the baseline is set and the
        # spike crosses the 1.5x sigma threshold.
        {("gradient_norm_spike", "warning"), ("gradient_norm_spike", "critical")},
    ),
]


def _make_monitor(tmp_log_dir: str, run_id: str) -> RLWatch:
    cfg = RLWatchConfig()
    cfg.storage.log_dir = tmp_log_dir
    cfg.run_id = run_id
    # Speed up warmups so the short fixtures actually exercise the detectors.
    cfg.entropy_collapse.warmup_steps = 5
    cfg.entropy_collapse.consecutive_steps = 20
    cfg.kl_explosion.warmup_steps = 5
    cfg.kl_explosion.rolling_window = 50
    cfg.reward_hacking.warmup_steps = 5
    cfg.reward_hacking.baseline_window = 30
    cfg.advantage_variance.warmup_steps = 5
    cfg.gradient_norm_spike.warmup_steps = 5
    cfg.gradient_norm_spike.rolling_window = 30
    cfg.gradient_norm_spike.baseline_mode = "frozen"
    # Wide cooldown so we see at least one of each tier.
    cfg.alerts.cooldown_steps = 10
    cfg.alerts.max_alerts_per_run = 1000
    monitor = RLWatch(cfg)
    monitor.start()
    return monitor


@pytest.mark.parametrize(
    "name,generator_fn,kwargs,expected", FIXTURES, ids=[f[0] for f in FIXTURES]
)
def test_replay_matches_expected_alerts(
    name, generator_fn, kwargs, expected, tmp_log_dir
):
    monitor = _make_monitor(tmp_log_dir, run_id=f"replay_{name}")
    try:
        rows = generator_fn(**kwargs)
        for row in rows:
            step = row.pop("step")
            monitor.log_step(step, **row)

        # Check the persisted alert set against expectations.
        alerts = monitor.store.get_alerts()
        observed = {(a["detector"], a["severity"]) for a in alerts}
        assert observed == expected, (
            f"fixture '{name}' alert set mismatch.\n"
            f"  expected: {sorted(expected)}\n"
            f"  observed: {sorted(observed)}"
        )

        # Metric rowcount sanity check — we logged exactly n rows.
        metrics = monitor.store.get_metrics()
        assert len(metrics) == len(rows)
    finally:
        monitor.stop()


def test_fixture_set_is_nonempty():
    """Guard against accidental fixture deletion."""
    assert len(FIXTURES) >= 6
