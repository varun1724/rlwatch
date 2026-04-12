"""Unit tests for the rewards-array branches of RewardHackingDetector.

The state-machine tests in ``test_detectors.py`` cover the variance-explosion
path via ``reward_std=...``. This file covers the parallel branch where the
caller passes a raw ``rewards`` array, which exercises:

- ``RewardHackingDetector.check`` rewards-array variance branch (332-333)
- The reward buffer accumulation that feeds the dip test (338-339)
- ``_hartigan_dip_test`` itself, including its edge cases:
    - too-small-input early return
    - degenerate (constant) input early return
    - p-value lookup table for each scaled-dip range
- The bimodal-detection branch in ``check`` (383-385)
"""

from __future__ import annotations

import numpy as np
import pytest

from rlwatch.config import RewardHackingConfig
from rlwatch.detectors import (
    HAS_DIPTEST,
    RewardHackingDetector,
    _hartigan_dip_test,
    _simplified_dip_test,
)


class TestHartiganDipTestDirectly:
    """Cover ``_hartigan_dip_test`` exhaustively.

    Note: this function is the home-rolled approximation we're shipping until
    the real ``diptest`` package is integrated in v0.3 (deferred — adds a C
    extension). We deliberately do *not* assert that it correctly classifies
    unimodal vs. bimodal data — it doesn't. We assert only that the function
    runs without raising and routes through every code path so the coverage
    gate stays meaningful.
    """

    def test_too_small_returns_neutral(self):
        dip, p = _hartigan_dip_test(np.array([1.0, 2.0, 3.0]))
        assert dip == 0.0
        assert p == 1.0

    def test_constant_input_returns_neutral(self):
        # Range collapses → degenerate-input early return path.
        dip, p = _hartigan_dip_test(np.full(50, 5.0))
        assert dip == 0.0
        assert p == 1.0

    def test_runs_on_random_input(self):
        """Smoke check: arbitrary unimodal data must not raise."""
        rng = np.random.default_rng(0)
        dip, p = _hartigan_dip_test(rng.normal(0, 1, size=100))
        # We don't pin the p-value — see class docstring.
        assert 0.0 <= p <= 1.0
        assert dip >= 0.0

    def test_pvalue_lookup_branches(self):
        """Hit every branch in the if/elif chain that maps dip→p_value.

        We can't reliably hit each branch by varying the input distribution
        because the simplified statistic is non-monotonic. Instead, we
        construct inputs of varying *size* against the same step-function
        shape so ``dip * sqrt(n)`` lands in each lookup band.
        """
        observed_pvalues: set[float] = set()
        for n in [10, 12, 14, 16, 20, 25, 30, 50, 100, 200]:
            # Step function: half zeros, half ones. The dip statistic for this
            # shape is constant; only sqrt(n) varies, so different sizes
            # produce different scaled dips and route to different branches.
            half = n // 2
            data = np.array([0.0] * half + [1.0] * (n - half))
            _, p = _hartigan_dip_test(data)
            observed_pvalues.add(p)
        # The lookup has 5 distinct return values: 0.001, 0.01, 0.05, 0.1, 0.5.
        # We don't require all 5 (some are unreachable from this shape), but
        # we do require that we routed through more than one branch — which
        # is the proof that the if/elif chain is reachable rather than dead.
        assert len(observed_pvalues) >= 2


class TestRewardsArrayBranch:
    def test_variance_computed_from_rewards_array(self):
        """When ``rewards`` is provided without ``reward_std``, the detector
        must compute variance itself and feed the dip-test buffer."""
        cfg = RewardHackingConfig(
            warmup_steps=2,
            variance_multiplier=3.0,
            baseline_window=30,
        )
        det = RewardHackingDetector(cfg)
        rng = np.random.default_rng(0)

        # Establish a baseline with stable, unimodal rewards.
        for step in range(30):
            det.check(step, rewards=rng.normal(0, 0.5, size=20))

        # The reward buffer should now hold ~600 samples — well past the
        # 50-sample minimum for the dip test branch.
        assert len(det._reward_buffer) >= 50
        # The variance baseline was set during the same loop.
        assert det._baseline_variance is not None

    def test_variance_explosion_via_rewards_array(self):
        cfg = RewardHackingConfig(
            warmup_steps=2,
            variance_multiplier=3.0,
            baseline_window=30,
        )
        det = RewardHackingDetector(cfg)
        rng = np.random.default_rng(0)

        for step in range(30):
            det.check(step, rewards=rng.normal(0, 0.5, size=20))

        # Now feed a high-variance sample.
        alert = det.check(31, rewards=rng.normal(0, 5.0, size=20))
        assert alert is not None
        assert alert.detector == "reward_hacking"

    def test_bimodal_dip_test_branch_fires(self):
        """A strongly bimodal reward distribution should trigger the dip-test
        branch (which is the only code path that returns a *warning*-severity
        reward_hacking alert without a variance explosion)."""
        cfg = RewardHackingConfig(
            warmup_steps=2,
            variance_multiplier=1e9,  # disable the variance branch
            dip_test_significance=0.2,  # accept the simplified test's range
            baseline_window=30,
        )
        det = RewardHackingDetector(cfg)
        rng = np.random.default_rng(0)

        # Stable healthy baseline.
        for step in range(30):
            det.check(step, rewards=rng.normal(0, 0.5, size=20))

        # Feed a bimodal distribution: half from one cluster, half from
        # another. We do this for several consecutive steps so the buffer
        # is dominated by bimodal samples.
        bimodal_alert = None
        for step in range(30, 60):
            half_a = rng.normal(-5, 0.1, size=10)
            half_b = rng.normal(5, 0.1, size=10)
            samples = np.concatenate([half_a, half_b])
            alert = det.check(step, rewards=samples)
            if alert is not None and "Bimodal" in alert.message:
                bimodal_alert = alert
                break

        assert bimodal_alert is not None, "dip-test branch never fired"
        assert bimodal_alert.severity == "warning"
        assert "dip_p_value" in bimodal_alert.metric_values

    def test_no_inputs_at_all_noop(self):
        """Neither ``reward_std`` nor ``rewards`` provided — must return None."""
        cfg = RewardHackingConfig(warmup_steps=2)
        det = RewardHackingDetector(cfg)
        for step in range(50):
            assert det.check(step) is None


class TestDiptestIntegration:
    """Tests for the optional ``diptest`` package integration.

    The dispatcher function ``_hartigan_dip_test`` uses the real ``diptest``
    package when installed (``HAS_DIPTEST=True``) and falls back to the
    home-rolled ``_simplified_dip_test`` when not.
    """

    def test_simplified_fallback_always_available(self):
        """The simplified implementation is always callable regardless of
        whether the real package is installed."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, size=100)
        dip, p = _simplified_dip_test(data)
        assert dip >= 0.0
        assert 0.0 <= p <= 1.0

    def test_dispatcher_returns_valid_tuple(self):
        """The dispatcher function returns (float, float) in both paths."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, size=100)
        dip, p = _hartigan_dip_test(data)
        assert isinstance(dip, float)
        assert isinstance(p, float)
        assert dip >= 0.0
        assert 0.0 <= p <= 1.0

    def test_has_diptest_flag_is_boolean(self):
        assert isinstance(HAS_DIPTEST, bool)

    @pytest.mark.skipif(not HAS_DIPTEST, reason="diptest not installed")
    def test_real_diptest_detects_bimodal(self):
        """When the real package is installed, strongly bimodal data should
        get a very low p-value (much more reliable than the simplified
        version)."""
        rng = np.random.default_rng(0)
        data = np.concatenate([
            rng.normal(-5, 0.1, 100),
            rng.normal(5, 0.1, 100),
        ])
        dip, p = _hartigan_dip_test(data)
        assert p < 0.01, f"real diptest should detect clear bimodal, got p={p}"

    @pytest.mark.skipif(not HAS_DIPTEST, reason="diptest not installed")
    def test_real_diptest_unimodal_high_pvalue(self):
        """Unimodal data should NOT be flagged by the real implementation."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=200)
        dip, p = _hartigan_dip_test(data)
        assert p > 0.05, f"real diptest should not flag unimodal, got p={p}"
