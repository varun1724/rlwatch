"""Simulate a GRPO training run with rlwatch monitoring.

This script demonstrates the core user journey: a researcher starts a GRPO run,
rlwatch detects an entropy collapse at step ~340, and sends an alert.

Usage:
    python examples/simulate_grpo_run.py

This generates realistic-looking training metrics including an entropy collapse
scenario, so you can see rlwatch's detection and alerting in action.
"""

import math
import random
import time

import numpy as np

import rlwatch


def simulate_grpo_run(total_steps=500, collapse_start=280, seed: int | None = None):
    """Simulate a GRPO training run with entropy collapse.

    Models a realistic scenario where:
    - Training starts normally with healthy entropy ~2.8
    - At step ~280, entropy begins collapsing due to a learning rate issue
    - By step ~340, entropy has collapsed below threshold (1.0)
    - rlwatch detects this and fires an alert

    Args:
        total_steps: Number of training steps to simulate.
        collapse_start: Step at which the synthetic entropy collapse begins.
        seed: Optional integer seed for ``random`` and ``numpy.random``. Set
            this when generating reproducible Tier 3 fixtures; leave ``None``
            for casual interactive runs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Attach rlwatch with a descriptive run ID
    monitor = rlwatch.attach(
        framework="manual",
        run_id="grpo_v3_exp12",
    )

    print(f"Starting simulated GRPO run: {monitor.run_id}")
    print(f"Log directory: {monitor.config.storage.log_dir}")
    print(f"Simulating entropy collapse starting at step {collapse_start}...")
    print()

    # Initial parameters
    initial_entropy = 2.8
    initial_kl = 0.01
    initial_reward_mean = -1.5
    initial_lr = 1e-5

    for step in range(total_steps):
        # Normal noise
        noise = random.gauss(0, 0.05)

        # --- Entropy ---
        if step < collapse_start:
            # Normal training: entropy slowly decreases
            entropy = initial_entropy - (step / total_steps) * 0.5 + noise * 0.1
        else:
            # Collapse: entropy drops rapidly
            steps_since_collapse = step - collapse_start
            decay = math.exp(-steps_since_collapse / 30)
            entropy = max(0.05, initial_entropy * decay * 0.3 + noise * 0.02)

        # --- KL Divergence ---
        if step < collapse_start:
            kl = initial_kl + (step / total_steps) * 0.05 + abs(noise) * 0.01
        else:
            # KL spikes during collapse
            steps_since = step - collapse_start
            kl = initial_kl + 0.05 + steps_since * 0.005 + abs(noise) * 0.02

        # --- Rewards ---
        if step < collapse_start:
            # Rewards slowly improve
            reward_mean = initial_reward_mean + (step / total_steps) * 2.0 + noise * 0.1
            reward_std = 0.5 + noise * 0.05
        else:
            # Rewards plateau or become noisy during collapse
            reward_mean = initial_reward_mean + 1.5 + noise * 0.3
            reward_std = 0.5 + (step - collapse_start) * 0.01 + abs(noise) * 0.1

        # Generate per-sample rewards
        rewards = np.random.normal(reward_mean, max(0.1, reward_std), size=64)

        # --- Advantage std ---
        if step < collapse_start:
            advantage_std = 1.0 + noise * 0.1
        else:
            # Advantage std increases during collapse
            advantage_std = 1.0 + (step - collapse_start) * 0.02 + abs(noise) * 0.2

        # --- Loss ---
        loss = 0.5 - (step / total_steps) * 0.3 + noise * 0.02

        # --- Learning Rate ---
        lr = initial_lr * (1 - step / total_steps * 0.5)

        # Log to rlwatch
        alerts = monitor.log_step(
            step,
            entropy=entropy,
            kl_divergence=kl,
            reward_mean=reward_mean,
            reward_std=reward_std,
            rewards=rewards,
            advantage_std=advantage_std,
            loss=loss,
            learning_rate=lr,
            clip_fraction=random.uniform(0.05, 0.15),
        )

        # Print progress every 50 steps
        if step % 50 == 0:
            print(
                f"Step {step:4d} | entropy={entropy:.3f} kl={kl:.4f} "
                f"reward={reward_mean:.3f} adv_std={advantage_std:.3f}"
            )

    # Finish
    monitor.stop()

    print()
    print("=" * 60)
    print("Simulation complete!")
    print(f"Total alerts: {monitor.alert_manager.total_alerts_sent}")
    print(f"Logs saved to: {monitor.config.storage.log_dir}")
    print()
    print("Next steps:")
    print("  1. Run retrospective analysis:")
    print(f"     rlwatch diagnose --log-dir {monitor.config.storage.log_dir}")
    print("  2. Launch the dashboard:")
    print(f"     rlwatch dashboard --log-dir {monitor.config.storage.log_dir}")


if __name__ == "__main__":
    simulate_grpo_run()
