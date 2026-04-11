"""Basic rlwatch usage examples.

Shows the two-line attach API and manual metric logging.
"""

import numpy as np

import rlwatch

# --- Example 1: Two-line attach (manual framework) ---

monitor = rlwatch.attach(framework="manual", run_id="example_basic")

# In your training loop:
for step in range(100):
    # ... your training code here ...

    # Log metrics from your training step
    rlwatch.log_step(
        step,
        entropy=2.5 - step * 0.02,          # Policy entropy
        kl_divergence=0.01 + step * 0.001,   # KL from reference
        reward_mean=float(np.random.normal(0, 1)),
        reward_std=float(np.abs(np.random.normal(0.5, 0.1))),
        advantage_std=float(np.abs(np.random.normal(1.0, 0.1))),
        loss=1.0 - step * 0.005,
    )

monitor.stop()
print("Basic example complete!")
print(f"Logs saved to: {monitor.config.storage.log_dir}")
