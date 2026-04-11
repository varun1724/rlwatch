"""Statistical detectors for GRPO/PPO training instabilities.

Four detectors:
1. Entropy collapse - policy entropy drops below threshold over consecutive steps
2. KL divergence explosion - KL from reference exceeds configurable sigma
3. Reward hacking proxy - reward variance explosion or bimodal distribution
4. Advantage variance spike - advantage std exceeds rolling historical baseline
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from rlwatch.config import (
    AdvantageVarianceConfig,
    EntropyCollapseConfig,
    GradientNormSpikeConfig,
    KLExplosionConfig,
    LossNaNConfig,
    RewardHackingConfig,
)


@dataclass
class Alert:
    """Represents a detected instability alert."""
    detector: str
    severity: str  # "warning" or "critical"
    step: int
    message: str
    metric_values: dict
    recommendation: str


class EntropyCollapseDetector:
    """Detects when policy entropy drops below threshold over consecutive steps.

    Entropy collapse indicates the policy is becoming deterministic too quickly,
    which often means it has converged to a degenerate solution.
    """

    def __init__(self, config: EntropyCollapseConfig):
        self.config = config
        self._consecutive_below = 0
        self._history: deque[float] = deque(maxlen=500)
        self._step_count = 0
        # Track the "initial" (pre-collapse) entropy from early healthy steps.
        # Separate from ``_history`` so it doesn't drift as the deque fills.
        self._initial_entropy_samples: list[float] = []
        # One-shot flags: prevent warning/critical from firing every step once
        # the detector is past the threshold. Reset whenever entropy recovers.
        self._warning_fired = False
        self._critical_fired = False

    def check(self, step: int, entropy: Optional[float]) -> Optional[Alert]:
        """Check for entropy collapse at the current step."""
        if not self.config.enabled or entropy is None:
            return None

        self._step_count += 1
        self._history.append(entropy)

        # Build up a stable "initial entropy" baseline during early healthy
        # steps so the alert message reflects the pre-collapse value.
        if (
            len(self._initial_entropy_samples) < 20
            and entropy >= self.config.threshold
        ):
            self._initial_entropy_samples.append(entropy)

        if self._step_count < self.config.warmup_steps:
            return None

        if entropy < self.config.threshold:
            self._consecutive_below += 1
        else:
            # Entropy recovered — reset counter and re-arm both alert tiers.
            self._consecutive_below = 0
            self._warning_fired = False
            self._critical_fired = False

        warning_threshold = max(1, self.config.consecutive_steps // 2)
        critical_threshold = self.config.consecutive_steps

        if self._consecutive_below >= critical_threshold and not self._critical_fired:
            self._critical_fired = True
            initial_entropy = (
                float(np.mean(self._initial_entropy_samples))
                if self._initial_entropy_samples
                else float(np.mean(list(self._history)[:min(20, len(self._history))]))
            )
            return Alert(
                detector="entropy_collapse",
                severity="critical",
                step=step,
                message=(
                    f"Entropy collapse detected — policy entropy dropped from "
                    f"{initial_entropy:.2f} to {entropy:.4f} over {self._consecutive_below} "
                    f"consecutive steps (threshold: {self.config.threshold})."
                ),
                metric_values={
                    "current_entropy": entropy,
                    "initial_entropy": float(initial_entropy),
                    "consecutive_steps_below": self._consecutive_below,
                    "threshold": self.config.threshold,
                },
                recommendation=(
                    "Reduce learning rate by 5x or increase KL penalty coefficient. "
                    "Consider increasing entropy bonus if available."
                ),
            )
        elif (
            self._consecutive_below >= warning_threshold
            and not self._warning_fired
        ):
            self._warning_fired = True
            initial_entropy = (
                float(np.mean(self._initial_entropy_samples))
                if self._initial_entropy_samples
                else float(np.mean(list(self._history)[:min(20, len(self._history))]))
            )
            return Alert(
                detector="entropy_collapse",
                severity="warning",
                step=step,
                message=(
                    f"Entropy declining — policy entropy at {entropy:.4f}, "
                    f"below threshold for {self._consecutive_below} steps "
                    f"(alert at {self.config.consecutive_steps})."
                ),
                metric_values={
                    "current_entropy": entropy,
                    "initial_entropy": float(initial_entropy),
                    "consecutive_steps_below": self._consecutive_below,
                    "threshold": self.config.threshold,
                },
                recommendation=(
                    "Monitor closely. If entropy continues declining, "
                    "consider reducing learning rate."
                ),
            )
        return None


class KLExplosionDetector:
    """Detects when KL divergence from reference policy explodes.

    KL explosion indicates the policy is diverging too far from the reference,
    which can lead to reward hacking or mode collapse.
    """

    def __init__(self, config: KLExplosionConfig):
        self.config = config
        self._history: deque[float] = deque(maxlen=config.rolling_window)
        self._step_count = 0
        self._frozen_mean: Optional[float] = None
        self._frozen_std: Optional[float] = None

    def check(self, step: int, kl_divergence: Optional[float]) -> Optional[Alert]:
        """Check for KL divergence explosion at the current step."""
        if not self.config.enabled or kl_divergence is None:
            return None

        self._step_count += 1
        self._history.append(kl_divergence)

        if self._step_count < self.config.warmup_steps:
            return None

        if len(self._history) < 10:
            return None

        # Pick the baseline source based on mode. Frozen baseline locks in the
        # first time the rolling window fills, mirroring RewardHackingDetector.
        if self.config.baseline_mode == "frozen":
            if self._frozen_mean is None and len(self._history) >= self.config.rolling_window:
                arr = np.array(self._history)
                self._frozen_mean = float(np.mean(arr))
                self._frozen_std = float(np.std(arr))
            if self._frozen_mean is None:
                # Window not full yet — fall back to rolling so we still alert
                # on sharp early spikes.
                values = np.array(self._history)
                rolling_mean = float(np.mean(values[:-1]))
                rolling_std = float(np.std(values[:-1]))
            else:
                rolling_mean = self._frozen_mean
                rolling_std = self._frozen_std
        else:
            values = np.array(self._history)
            rolling_mean = float(np.mean(values[:-1]))
            rolling_std = float(np.std(values[:-1]))

        if rolling_std < 1e-8:
            rolling_std = 1e-8

        z_score = (kl_divergence - rolling_mean) / rolling_std
        threshold = self.config.sigma_multiplier

        if z_score > threshold * 1.5:
            return Alert(
                detector="kl_explosion",
                severity="critical",
                step=step,
                message=(
                    f"KL divergence explosion — KL={kl_divergence:.4f}, "
                    f"{z_score:.1f}σ above rolling mean ({rolling_mean:.4f}). "
                    f"Exceeds {threshold * 1.5:.1f}σ critical threshold."
                ),
                metric_values={
                    "kl_divergence": kl_divergence,
                    "rolling_mean": float(rolling_mean),
                    "rolling_std": float(rolling_std),
                    "z_score": float(z_score),
                    "clip_region": self.config.clip_region,
                },
                recommendation=(
                    "Immediately reduce learning rate or increase KL penalty. "
                    "Consider reverting to a previous checkpoint."
                ),
            )
        elif z_score > threshold:
            return Alert(
                detector="kl_explosion",
                severity="warning",
                step=step,
                message=(
                    f"KL divergence elevated — KL={kl_divergence:.4f}, "
                    f"{z_score:.1f}σ above rolling mean ({rolling_mean:.4f})."
                ),
                metric_values={
                    "kl_divergence": kl_divergence,
                    "rolling_mean": float(rolling_mean),
                    "rolling_std": float(rolling_std),
                    "z_score": float(z_score),
                    "clip_region": self.config.clip_region,
                },
                recommendation=(
                    "Monitor KL divergence closely. Consider increasing KL penalty "
                    "coefficient if this trend continues."
                ),
            )
        return None


def _hartigan_dip_test(data: np.ndarray) -> tuple[float, float]:
    """Compute Hartigan's dip test statistic for unimodality.

    Returns (dip_statistic, p_value).
    Uses a simplified implementation suitable for online monitoring.
    """
    if len(data) < 10:
        return 0.0, 1.0

    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Compute the empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Compute the greatest convex minorant (GCM) and least concave majorant (LCM)
    # Simplified: compute max deviation between ECDF and uniform distribution
    uniform = np.linspace(sorted_data[0], sorted_data[-1], n)
    if sorted_data[-1] - sorted_data[0] < 1e-10:
        return 0.0, 1.0

    # Normalize to [0, 1]
    normalized = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
    uniform_cdf = np.linspace(0, 1, n)

    # Dip = max deviation between ECDF and best-fitting unimodal distribution
    dip = np.max(np.abs(normalized - uniform_cdf)) / 2

    # Approximate p-value using the asymptotic distribution
    # For n > 10, the dip statistic * sqrt(n) has a known distribution
    dip_scaled = dip * math.sqrt(n)

    # Conservative p-value approximation
    if dip_scaled > 1.0:
        p_value = 0.001
    elif dip_scaled > 0.7:
        p_value = 0.01
    elif dip_scaled > 0.5:
        p_value = 0.05
    elif dip_scaled > 0.3:
        p_value = 0.1
    else:
        p_value = 0.5

    return float(dip), float(p_value)


class RewardHackingDetector:
    """Detects reward hacking via variance explosion or bimodal reward distribution.

    Reward hacking manifests as either:
    1. Per-sample reward variance exploding (>3x baseline)
    2. Bimodal reward distribution (detected via Hartigan's dip test)
    """

    def __init__(self, config: RewardHackingConfig):
        self.config = config
        self._variance_history: deque[float] = deque(maxlen=config.baseline_window)
        self._reward_buffer: deque[float] = deque(maxlen=200)
        self._step_count = 0
        self._baseline_variance: Optional[float] = None

    def check(
        self,
        step: int,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None,
        rewards: Optional[np.ndarray] = None,
    ) -> Optional[Alert]:
        """Check for reward hacking at the current step."""
        if not self.config.enabled:
            return None

        self._step_count += 1

        # Track variance
        if reward_std is not None:
            variance = reward_std ** 2
            self._variance_history.append(variance)
        elif rewards is not None:
            variance = float(np.var(rewards))
            self._variance_history.append(variance)
        else:
            return None

        # Buffer individual rewards for dip test
        if rewards is not None:
            self._reward_buffer.extend(rewards.tolist())

        if self._step_count < self.config.warmup_steps:
            return None

        # Establish baseline from first N steps after warmup
        if self._baseline_variance is None and len(self._variance_history) >= 20:
            self._baseline_variance = float(np.mean(list(self._variance_history)[:20]))
            if self._baseline_variance < 1e-8:
                self._baseline_variance = 1e-8

        if self._baseline_variance is None:
            return None

        current_variance = variance
        variance_ratio = current_variance / self._baseline_variance

        # Check 1: Variance explosion
        if variance_ratio > self.config.variance_multiplier:
            alert = Alert(
                detector="reward_hacking",
                severity="critical" if variance_ratio > self.config.variance_multiplier * 2 else "warning",
                step=step,
                message=(
                    f"Reward hacking proxy triggered — reward variance exploded "
                    f"{variance_ratio:.1f}x above baseline "
                    f"(current: {current_variance:.4f}, baseline: {self._baseline_variance:.4f})."
                ),
                metric_values={
                    "current_variance": current_variance,
                    "baseline_variance": self._baseline_variance,
                    "variance_ratio": variance_ratio,
                    "reward_mean": float(reward_mean) if reward_mean is not None else None,
                    "reward_std": float(reward_std) if reward_std is not None else None,
                },
                recommendation=(
                    "Investigate reward model outputs for exploitation patterns. "
                    "Consider adding reward model regularization or capping rewards."
                ),
            )
            return alert

        # Check 2: Bimodal distribution via dip test
        if len(self._reward_buffer) >= 50:
            dip, p_value = _hartigan_dip_test(np.array(self._reward_buffer))
            if p_value < self.config.dip_test_significance:
                return Alert(
                    detector="reward_hacking",
                    severity="warning",
                    step=step,
                    message=(
                        f"Bimodal reward distribution detected — Hartigan dip test "
                        f"p-value={p_value:.4f} (significance: {self.config.dip_test_significance}). "
                        f"Policy may be exploiting reward model artifacts."
                    ),
                    metric_values={
                        "dip_statistic": dip,
                        "dip_p_value": p_value,
                        "reward_mean": float(reward_mean) if reward_mean is not None else None,
                        "n_samples": len(self._reward_buffer),
                    },
                    recommendation=(
                        "Inspect reward distribution for clusters. "
                        "Consider adding diversity regularization or reward model ensembling."
                    ),
                )

        return None


class AdvantageVarianceDetector:
    """Detects advantage variance spikes above rolling historical baseline.

    Large spikes in advantage standard deviation indicate unstable value
    function estimates or problematic reward scaling.
    """

    def __init__(self, config: AdvantageVarianceConfig):
        self.config = config
        self._history: deque[float] = deque(maxlen=config.rolling_window)
        self._step_count = 0
        self._frozen_mean: Optional[float] = None
        self._frozen_std: Optional[float] = None

    def check(self, step: int, advantage_std: Optional[float]) -> Optional[Alert]:
        """Check for advantage variance spike at the current step."""
        if not self.config.enabled or advantage_std is None:
            return None

        self._step_count += 1
        self._history.append(advantage_std)

        if self._step_count < self.config.warmup_steps:
            return None

        if len(self._history) < 10:
            return None

        if self.config.baseline_mode == "frozen":
            if self._frozen_mean is None and len(self._history) >= self.config.rolling_window:
                arr = np.array(self._history)
                self._frozen_mean = float(np.mean(arr))
                self._frozen_std = float(np.std(arr))
            if self._frozen_mean is None:
                values = np.array(self._history)
                baseline_mean = float(np.mean(values[:-1]))
                baseline_std = float(np.std(values[:-1]))
            else:
                baseline_mean = self._frozen_mean
                baseline_std = self._frozen_std
        else:
            values = np.array(self._history)
            baseline_mean = float(np.mean(values[:-1]))
            baseline_std = float(np.std(values[:-1]))

        if baseline_mean < 1e-8:
            baseline_mean = 1e-8

        ratio = advantage_std / baseline_mean

        if ratio > self.config.std_multiplier * 2:
            return Alert(
                detector="advantage_variance",
                severity="critical",
                step=step,
                message=(
                    f"Advantage variance spike — advantage std={advantage_std:.4f}, "
                    f"{ratio:.1f}x above rolling baseline ({baseline_mean:.4f}). "
                    f"Exceeds {self.config.std_multiplier * 2:.0f}x critical threshold."
                ),
                metric_values={
                    "advantage_std": advantage_std,
                    "baseline_mean": float(baseline_mean),
                    "baseline_std": float(baseline_std),
                    "ratio": ratio,
                },
                recommendation=(
                    "Value function estimates are unstable. Reduce learning rate, "
                    "increase batch size, or add advantage normalization."
                ),
            )
        elif ratio > self.config.std_multiplier:
            return Alert(
                detector="advantage_variance",
                severity="warning",
                step=step,
                message=(
                    f"Advantage variance elevated — advantage std={advantage_std:.4f}, "
                    f"{ratio:.1f}x above rolling baseline ({baseline_mean:.4f})."
                ),
                metric_values={
                    "advantage_std": advantage_std,
                    "baseline_mean": float(baseline_mean),
                    "baseline_std": float(baseline_std),
                    "ratio": ratio,
                },
                recommendation=(
                    "Monitor advantage statistics. Consider normalizing advantages "
                    "or adjusting GAE lambda."
                ),
            )
        return None


class LossNaNDetector:
    """Fires a critical alert the instant the training loss is non-finite.

    Loss going to NaN or ±Inf means the optimizer has blown up. There's no
    rolling state and no warning tier — by the time you see this, the run is
    already corrupted and the only useful action is to stop.
    """

    def __init__(self, config: LossNaNConfig):
        self.config = config
        self._step_count = 0

    def check(self, step: int, loss: Optional[float]) -> Optional[Alert]:
        if not self.config.enabled or loss is None:
            return None

        self._step_count += 1
        if self._step_count < self.config.warmup_steps:
            return None

        if math.isfinite(loss):
            return None

        kind = "NaN" if math.isnan(loss) else ("+Inf" if loss > 0 else "-Inf")
        return Alert(
            detector="loss_nan",
            severity="critical",
            step=step,
            message=(
                f"Loss is non-finite ({kind}). The optimizer has blown up; "
                f"any further updates will corrupt the policy."
            ),
            metric_values={"loss": float(loss) if math.isfinite(loss) else None, "kind": kind},
            recommendation=(
                "Stop the run immediately. Revert to the last good checkpoint. "
                "Reduce learning rate, clip gradients, and check for divide-by-zero "
                "or log-of-zero in the loss computation."
            ),
        )


class GradientNormSpikeDetector:
    """Detects gradient-norm spikes via z-score against a frozen baseline.

    Defaults to ``baseline_mode='frozen'`` because gradient norms drift slowly
    on healthy runs and a rolling baseline tends to follow them up, masking the
    actual spike. Mirrors KLExplosionDetector's z-score approach.
    """

    def __init__(self, config: GradientNormSpikeConfig):
        self.config = config
        self._history: deque[float] = deque(maxlen=config.rolling_window)
        self._step_count = 0
        self._frozen_mean: Optional[float] = None
        self._frozen_std: Optional[float] = None

    def check(self, step: int, grad_norm: Optional[float]) -> Optional[Alert]:
        if not self.config.enabled or grad_norm is None:
            return None

        self._step_count += 1
        self._history.append(grad_norm)

        if self._step_count < self.config.warmup_steps:
            return None
        if len(self._history) < 10:
            return None

        if self.config.baseline_mode == "frozen":
            if self._frozen_mean is None and len(self._history) >= self.config.rolling_window:
                arr = np.array(self._history)
                self._frozen_mean = float(np.mean(arr))
                self._frozen_std = float(np.std(arr))
            if self._frozen_mean is None:
                values = np.array(self._history)
                baseline_mean = float(np.mean(values[:-1]))
                baseline_std = float(np.std(values[:-1]))
            else:
                baseline_mean = self._frozen_mean
                baseline_std = self._frozen_std
        else:
            values = np.array(self._history)
            baseline_mean = float(np.mean(values[:-1]))
            baseline_std = float(np.std(values[:-1]))

        if baseline_std < 1e-8:
            baseline_std = 1e-8

        z_score = (grad_norm - baseline_mean) / baseline_std
        threshold = self.config.sigma_multiplier

        if z_score > threshold * 1.5:
            return Alert(
                detector="gradient_norm_spike",
                severity="critical",
                step=step,
                message=(
                    f"Gradient norm spike — grad_norm={grad_norm:.4f}, "
                    f"{z_score:.1f}σ above baseline ({baseline_mean:.4f}). "
                    f"Often a precursor to loss NaN."
                ),
                metric_values={
                    "grad_norm": grad_norm,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "z_score": float(z_score),
                },
                recommendation=(
                    "Clip gradients (max_grad_norm), reduce learning rate, "
                    "or check for exploding activations in the loss path."
                ),
            )
        elif z_score > threshold:
            return Alert(
                detector="gradient_norm_spike",
                severity="warning",
                step=step,
                message=(
                    f"Gradient norm elevated — grad_norm={grad_norm:.4f}, "
                    f"{z_score:.1f}σ above baseline ({baseline_mean:.4f})."
                ),
                metric_values={
                    "grad_norm": grad_norm,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "z_score": float(z_score),
                },
                recommendation=(
                    "Monitor gradient norms. Consider tightening gradient clipping."
                ),
            )
        return None


class DetectorSuite:
    """Runs all configured detectors on each training step."""

    def __init__(self, config):
        self.entropy_detector = EntropyCollapseDetector(config.entropy_collapse)
        self.kl_detector = KLExplosionDetector(config.kl_explosion)
        self.reward_detector = RewardHackingDetector(config.reward_hacking)
        self.advantage_detector = AdvantageVarianceDetector(config.advantage_variance)
        self.loss_nan_detector = LossNaNDetector(config.loss_nan)
        self.grad_norm_detector = GradientNormSpikeDetector(config.gradient_norm_spike)

    def check_step(
        self,
        step: int,
        *,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None,
        rewards: Optional[np.ndarray] = None,
        advantage_std: Optional[float] = None,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> list[Alert]:
        """Run all detectors and return any triggered alerts."""
        alerts = []

        result = self.entropy_detector.check(step, entropy)
        if result:
            alerts.append(result)

        result = self.kl_detector.check(step, kl_divergence)
        if result:
            alerts.append(result)

        result = self.reward_detector.check(
            step,
            reward_mean=reward_mean,
            reward_std=reward_std,
            rewards=rewards,
        )
        if result:
            alerts.append(result)

        result = self.advantage_detector.check(step, advantage_std)
        if result:
            alerts.append(result)

        result = self.loss_nan_detector.check(step, loss)
        if result:
            alerts.append(result)

        result = self.grad_norm_detector.check(step, grad_norm)
        if result:
            alerts.append(result)

        return alerts
