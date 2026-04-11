"""Shared Hypothesis strategies for rlwatch property-based tests.

We deliberately keep strategies *small and constrained*:
- Float ranges are bounded so we don't waste examples on cases the user will
  never see (entropy of 1e308, etc.).
- Sequences are bounded so each property test runs in well under a second.
- ``allow_nan`` and ``allow_infinity`` are off by default — the LossNaN
  detector has its own dedicated tests for those.
"""

from __future__ import annotations

from hypothesis import strategies as st

# A "valid" finite float in a range that real GRPO/PPO metrics live in.
finite_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Healthy ranges for each metric — used for "no false positive" invariants.
healthy_entropy = st.floats(min_value=1.5, max_value=4.0)
healthy_kl = st.floats(min_value=0.001, max_value=0.05)
healthy_reward_std = st.floats(min_value=0.1, max_value=2.0)
healthy_advantage_std = st.floats(min_value=0.5, max_value=2.0)
healthy_grad_norm = st.floats(min_value=0.1, max_value=5.0)

# Bounded sequences to keep test runtime tight.
metric_sequence = st.lists(finite_float, min_size=0, max_size=300)
healthy_entropy_sequence = st.lists(healthy_entropy, min_size=20, max_size=200)
healthy_kl_sequence = st.lists(healthy_kl, min_size=20, max_size=200)
healthy_reward_std_sequence = st.lists(
    healthy_reward_std, min_size=20, max_size=200
)
healthy_advantage_sequence = st.lists(
    healthy_advantage_std, min_size=20, max_size=200
)


def maybe_none(strategy):
    """Wrap a strategy so it occasionally yields ``None``.

    Used by None-safety invariants.
    """
    return st.one_of(st.none(), strategy)
