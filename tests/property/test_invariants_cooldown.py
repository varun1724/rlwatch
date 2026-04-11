"""Property: AlertManager honors cooldown for repeated same-severity alerts.

Companion invariant: a critical IS allowed to preempt a warning still in
cooldown — that's the bug we fixed.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.property

COOLDOWN = 50


def _alert(detector, severity, step):
    return Alert(
        detector=detector, severity=severity, step=step,
        message="m", metric_values={}, recommendation="r",
    )


@given(steps=st.lists(st.integers(min_value=0, max_value=10_000),
                       min_size=1, max_size=50))
def test_same_severity_repeats_respect_cooldown(steps):
    mgr = AlertManager(
        AlertConfig(cooldown_steps=COOLDOWN, max_alerts_per_run=10_000),
        run_id="r",
    )
    sent_steps: list[int] = []
    for step in sorted(set(steps)):
        if mgr.send(_alert("entropy_collapse", "warning", step)):
            sent_steps.append(step)
    # Every consecutive pair of accepted alerts must be ≥ cooldown apart.
    for a, b in zip(sent_steps, sent_steps[1:]):
        assert b - a >= COOLDOWN


@given(warning_step=st.integers(min_value=0, max_value=10_000),
        critical_offset=st.integers(min_value=0, max_value=COOLDOWN - 1))
def test_critical_preempts_warning_within_cooldown(warning_step, critical_offset):
    """A critical inside the warning's cooldown window must still be accepted."""
    mgr = AlertManager(
        AlertConfig(cooldown_steps=COOLDOWN, max_alerts_per_run=10_000),
        run_id="r",
    )
    assert mgr.send(_alert("kl_explosion", "warning", warning_step)) is True
    crit_step = warning_step + critical_offset
    assert mgr.send(_alert("kl_explosion", "critical", crit_step)) is True
