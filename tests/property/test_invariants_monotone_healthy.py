"""Property: a monotonically healthy run never produces critical alerts.

This is the highest-value invariant in Tier 2 — it asserts that a run that's
"obviously going well" by every common-sense metric does not get nagged.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlwatch.config import RLWatchConfig
from rlwatch.detectors import DetectorSuite

pytestmark = pytest.mark.property


@given(
    n_steps=st.integers(min_value=50, max_value=200),
    initial_entropy=st.floats(min_value=2.0, max_value=4.0),
    initial_loss=st.floats(min_value=0.5, max_value=2.0),
)
def test_no_critical_alerts_on_healthy_monotone_run(
    n_steps, initial_entropy, initial_loss
):
    cfg = RLWatchConfig()
    # Tighten warmups so the test exercises the post-warmup path quickly.
    cfg.entropy_collapse.warmup_steps = 5
    cfg.kl_explosion.warmup_steps = 5
    cfg.reward_hacking.warmup_steps = 5
    cfg.advantage_variance.warmup_steps = 5
    cfg.gradient_norm_spike.warmup_steps = 5

    suite = DetectorSuite(cfg)

    critical_alerts = []
    for step in range(n_steps):
        # Healthy trends:
        # - entropy stays well above threshold (1.0)
        # - kl_divergence stays small and stable
        # - reward_std stays stable
        # - advantage_std stays stable
        # - loss decreases smoothly
        # - grad_norm stays low and stable
        entropy = max(1.5, initial_entropy - 0.001 * step)
        kl = 0.01
        reward_std = 0.5
        advantage_std = 1.0
        loss = max(0.05, initial_loss - 0.002 * step)
        grad_norm = 1.0

        alerts = suite.check_step(
            step,
            entropy=entropy,
            kl_divergence=kl,
            reward_std=reward_std,
            advantage_std=advantage_std,
            loss=loss,
            grad_norm=grad_norm,
        )
        critical_alerts.extend(a for a in alerts if a.severity == "critical")

    assert critical_alerts == [], (
        f"Healthy monotone run produced critical alerts: "
        f"{[(a.detector, a.step) for a in critical_alerts]}"
    )
