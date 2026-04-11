"""Property: total alerts sent never exceeds max_alerts_per_run."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlwatch.alerts import AlertManager
from rlwatch.config import AlertConfig
from rlwatch.detectors import Alert

pytestmark = pytest.mark.property


def _alert(detector, severity, step):
    return Alert(
        detector=detector, severity=severity, step=step,
        message="m", metric_values={}, recommendation="r",
    )


@given(
    max_alerts=st.integers(min_value=1, max_value=20),
    candidate_count=st.integers(min_value=0, max_value=200),
)
def test_max_alerts_never_exceeded(max_alerts, candidate_count):
    mgr = AlertManager(
        AlertConfig(cooldown_steps=0, max_alerts_per_run=max_alerts),
        run_id="r",
    )
    # Generate (detector, severity, step) tuples that always pass cooldown
    # by spacing them very widely.
    for i in range(candidate_count):
        mgr.send(_alert(f"det_{i % 3}", "warning", step=i * 10_000))
    assert mgr.total_alerts_sent <= max_alerts
