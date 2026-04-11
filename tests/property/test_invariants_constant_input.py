"""Property: a healthy constant input never produces critical alerts."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlwatch.config import (
    AdvantageVarianceConfig,
    EntropyCollapseConfig,
    GradientNormSpikeConfig,
    KLExplosionConfig,
    RewardHackingConfig,
)
from rlwatch.detectors import (
    AdvantageVarianceDetector,
    EntropyCollapseDetector,
    GradientNormSpikeDetector,
    KLExplosionDetector,
    RewardHackingDetector,
)

pytestmark = pytest.mark.property

NSTEPS = 200


@given(entropy=st.floats(min_value=1.5, max_value=4.0))
def test_entropy_constant_above_threshold_never_alerts(entropy):
    det = EntropyCollapseDetector(
        EntropyCollapseConfig(warmup_steps=10, threshold=1.0, consecutive_steps=20)
    )
    for step in range(NSTEPS):
        assert det.check(step, entropy=entropy) is None


@given(kl=st.floats(min_value=0.001, max_value=0.05))
def test_kl_constant_never_alerts(kl):
    det = KLExplosionDetector(KLExplosionConfig(warmup_steps=10))
    for step in range(NSTEPS):
        det.check(step, kl_divergence=kl)
    # No assertion on intermediate alerts — the std-zero guard makes z-score 0
    # when constant, so this is just a "did not raise" check.


@given(reward_std=st.floats(min_value=0.1, max_value=2.0))
def test_reward_std_constant_never_alerts(reward_std):
    det = RewardHackingDetector(
        RewardHackingConfig(warmup_steps=10, baseline_window=30)
    )
    for step in range(NSTEPS):
        assert det.check(step, reward_std=reward_std) is None


@given(advantage_std=st.floats(min_value=0.5, max_value=2.0))
def test_advantage_constant_never_alerts(advantage_std):
    det = AdvantageVarianceDetector(
        AdvantageVarianceConfig(warmup_steps=10)
    )
    for step in range(NSTEPS):
        det.check(step, advantage_std=advantage_std)
    # Same rationale as KL — std-zero guard means ratio stays at 1.0.


@given(grad_norm=st.floats(min_value=0.1, max_value=5.0))
def test_grad_norm_constant_never_alerts(grad_norm):
    det = GradientNormSpikeDetector(
        GradientNormSpikeConfig(warmup_steps=10, rolling_window=30)
    )
    for step in range(NSTEPS):
        det.check(step, grad_norm=grad_norm)
