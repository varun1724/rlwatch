"""Property: arbitrary interleavings of None and valid values never raise."""

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


def _maybe_none(strategy):
    return st.one_of(st.none(), strategy)


finite = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)


@given(values=st.lists(_maybe_none(finite), min_size=0, max_size=80))
def test_entropy_none_safe(values):
    det = EntropyCollapseDetector(EntropyCollapseConfig(warmup_steps=2))
    for step, v in enumerate(values):
        det.check(step, entropy=v)  # must not raise


@given(values=st.lists(_maybe_none(positive), min_size=0, max_size=80))
def test_kl_none_safe(values):
    det = KLExplosionDetector(KLExplosionConfig(warmup_steps=2))
    for step, v in enumerate(values):
        det.check(step, kl_divergence=v)


@given(values=st.lists(_maybe_none(positive), min_size=0, max_size=80))
def test_reward_none_safe(values):
    det = RewardHackingDetector(RewardHackingConfig(warmup_steps=2))
    for step, v in enumerate(values):
        det.check(step, reward_std=v)


@given(values=st.lists(_maybe_none(positive), min_size=0, max_size=80))
def test_advantage_none_safe(values):
    det = AdvantageVarianceDetector(AdvantageVarianceConfig(warmup_steps=2))
    for step, v in enumerate(values):
        det.check(step, advantage_std=v)


@given(values=st.lists(_maybe_none(finite), min_size=0, max_size=80))
def test_loss_nan_none_safe(values):
    det = LossNaNDetector(LossNaNConfig())
    for step, v in enumerate(values):
        det.check(step, loss=v)


@given(values=st.lists(_maybe_none(positive), min_size=0, max_size=80))
def test_grad_norm_none_safe(values):
    det = GradientNormSpikeDetector(GradientNormSpikeConfig(warmup_steps=2))
    for step, v in enumerate(values):
        det.check(step, grad_norm=v)
