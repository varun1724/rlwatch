"""Property: no detector ever fires inside its warmup window."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlwatch.config import (
    AdvantageVarianceConfig,
    EntropyCollapseConfig,
    GradientNormSpikeConfig,
    KLExplosionConfig,
    LossNaNConfig,
    RewardHackingConfig,
)
from rlwatch.detectors import (
    AdvantageVarianceDetector,
    EntropyCollapseDetector,
    GradientNormSpikeDetector,
    KLExplosionDetector,
    LossNaNDetector,
    RewardHackingDetector,
)

pytestmark = pytest.mark.property

WARMUP = 30


@given(values=st.lists(st.floats(min_value=-1e3, max_value=1e3,
                                  allow_nan=False, allow_infinity=False),
                        min_size=0, max_size=WARMUP))
def test_entropy_no_alert_during_warmup(values):
    det = EntropyCollapseDetector(
        EntropyCollapseConfig(warmup_steps=WARMUP, threshold=1.0, consecutive_steps=5)
    )
    for step, v in enumerate(values):
        assert det.check(step, entropy=v) is None


@given(values=st.lists(st.floats(min_value=-1e3, max_value=1e3,
                                  allow_nan=False, allow_infinity=False),
                        min_size=0, max_size=WARMUP))
def test_kl_no_alert_during_warmup(values):
    det = KLExplosionDetector(KLExplosionConfig(warmup_steps=WARMUP))
    for step, v in enumerate(values):
        assert det.check(step, kl_divergence=v) is None


@given(values=st.lists(st.floats(min_value=0, max_value=1e3,
                                  allow_nan=False, allow_infinity=False),
                        min_size=0, max_size=WARMUP))
def test_reward_no_alert_during_warmup(values):
    det = RewardHackingDetector(RewardHackingConfig(warmup_steps=WARMUP))
    for step, v in enumerate(values):
        assert det.check(step, reward_std=v) is None


@given(values=st.lists(st.floats(min_value=0, max_value=1e3,
                                  allow_nan=False, allow_infinity=False),
                        min_size=0, max_size=WARMUP))
def test_advantage_no_alert_during_warmup(values):
    det = AdvantageVarianceDetector(
        AdvantageVarianceConfig(warmup_steps=WARMUP)
    )
    for step, v in enumerate(values):
        assert det.check(step, advantage_std=v) is None


@given(values=st.lists(st.floats(allow_nan=True, allow_infinity=True),
                        min_size=0, max_size=WARMUP))
def test_loss_nan_no_alert_during_warmup(values):
    det = LossNaNDetector(LossNaNConfig(warmup_steps=WARMUP))
    for step, v in enumerate(values):
        assert det.check(step, loss=v) is None


@given(values=st.lists(st.floats(min_value=0, max_value=1e3,
                                  allow_nan=False, allow_infinity=False),
                        min_size=0, max_size=WARMUP))
def test_grad_norm_no_alert_during_warmup(values):
    det = GradientNormSpikeDetector(
        GradientNormSpikeConfig(warmup_steps=WARMUP)
    )
    for step, v in enumerate(values):
        assert det.check(step, grad_norm=v) is None
