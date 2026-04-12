"""End-to-end TRL + GRPO tutorial — rlwatch catches a real entropy collapse.

This is the headline tutorial for v0.3.0. It runs a tiny GPT-2 model
through TRL's GRPOTrainer on a synthetic 20-prompt dataset with a
*deliberately* misconfigured learning rate that induces an entropy collapse
within the first ~150 steps. rlwatch is attached via the standard two-line
API and fires a critical alert when the collapse happens.

Constraints (matched to v0.3.0 plan):
- CPU-friendly. Runs in ~5 minutes on a 2020-era laptop.
- Reproducible. Three seeds set (random, numpy, torch).
- No external API keys.
- One alert minimum: a real ``entropy_collapse`` from the real detector
  on real TRL training, not synthetic metrics.

Install:
    pip install "rlwatch[trl,tutorial]"

Run:
    python examples/trl_grpo_tutorial.py

The tutorial CI cron in ``.github/workflows/tutorial.yml`` runs this script
every month and asserts the alert fires. If a future TRL release silently
breaks the path, that cron catches it.
"""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np

# Quiet TRL/transformers warnings that aren't actionable for the tutorial.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)


def main() -> int:
    # ------------------------------------------------------------------
    # Reproducibility — three seeds, fixed.
    # ------------------------------------------------------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(
            f"Missing tutorial dependencies: {e}\n\n"
            f"Install with:\n"
            f"  pip install \"rlwatch[trl,tutorial]\"",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(SEED)

    import rlwatch

    # ------------------------------------------------------------------
    # Tiny model. GPT-2 (124M params) is the smallest causal LM that
    # ships with transformers and is small enough to GRPO-train on CPU.
    # ------------------------------------------------------------------
    print("[1/4] Loading GPT-2 tokenizer and model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Trivial synthetic dataset. The "task" is: respond with the word
    # "YES". The reward function gives 1.0 if the completion starts with
    # YES (case-insensitive) and 0.0 otherwise. This is an absurdly easy
    # task for a real model — but with a *deliberately* too-high LR,
    # GRPO will collapse entropy long before it learns to do it reliably.
    # ------------------------------------------------------------------
    print("[2/4] Building synthetic dataset (20 prompts)...")
    prompts = [
        "Reply YES if you understand:",
        "Confirm the request:",
        "Acknowledge:",
        "Are you ready?",
        "Do you copy?",
        "Roger that:",
        "Standing by:",
        "Awaiting confirmation:",
        "Ready when you are:",
        "Please respond:",
    ] * 2  # 20 prompts total
    dataset = Dataset.from_dict({"prompt": prompts})

    def reward_starts_with_yes(completions, **kwargs):
        return [
            1.0 if completion.strip().upper().startswith("YES") else 0.0
            for completion in completions
        ]

    # ------------------------------------------------------------------
    # GRPO config with a *deliberately* too-high learning rate.
    # 1e-3 on a 124M model is an order of magnitude above what's safe;
    # combined with the trivial reward function and a small batch, GRPO
    # collapses entropy within the first ~150 steps. Healthy LR for this
    # setup is around 5e-6.
    # ------------------------------------------------------------------
    print("[3/4] Building GRPOTrainer (LR is deliberately too high)...")
    args = GRPOConfig(
        output_dir="./_rlwatch_tutorial_output",
        learning_rate=1e-2,                  # deliberately too high (10x safe)
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,                   # GRPO requires >1
        max_completion_length=8,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        # Use CPU explicitly so the tutorial is reproducible across machines.
        use_cpu=True,
        # Classic GRPO loss — TRL 1.1.0 defaults to "dapo" which clips
        # gradients too aggressively for a high-LR collapse demo.
        loss_type="grpo",
        num_iterations=3,                    # multiple grad steps per batch
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_starts_with_yes,
        args=args,
        train_dataset=dataset,
    )

    # ------------------------------------------------------------------
    # Two-line attach. Pass the trainer in directly so the callback is
    # registered without scanning the object graph (the v0.2 path).
    # ------------------------------------------------------------------
    print("[4/4] Attaching rlwatch and starting training...\n")
    monitor = rlwatch.attach(
        trainer=trainer,
        run_id="trl_grpo_tutorial",
        # Tighten the entropy collapse detector so it fires inside the
        # 200-ish steps the tutorial runs for. Defaults are warmup=20,
        # consecutive=50 — fine for production runs that go thousands of
        # steps, too patient for a 5-minute demo.
        entropy_collapse={"warmup_steps": 5, "consecutive_steps": 15},
    )

    try:
        trainer.train()
    finally:
        # Read alerts BEFORE stop() — stop() closes the SQLite connection.
        alerts = monitor.store.get_alerts()
        monitor.stop()
    print("\n" + "=" * 64)
    print(f"Tutorial complete. {len(alerts)} alert(s) fired.")
    print("=" * 64)
    if any(a["detector"] == "entropy_collapse" for a in alerts):
        print(
            "\n✅ rlwatch caught the entropy collapse caused by the\n"
            "   deliberately-too-high learning rate.\n\n"
            "Next steps:\n"
            "  1. Run `rlwatch diagnose` to see the full report.\n"
            "  2. Re-run with `learning_rate=5e-6` and watch the alert NOT fire.\n"
            "  3. Read the tutorial walkthrough at\n"
            "     https://varun1724.github.io/rlwatch/tutorials/trl-grpo-end-to-end/\n"
        )
        return 0
    else:
        print(
            "\n⚠️  No entropy_collapse alert fired. This is unexpected — the\n"
            "   tutorial is supposed to deterministically reproduce a collapse.\n"
            "   Check that your installed versions of trl/transformers/torch\n"
            "   match the [tutorial] extra in pyproject.toml.\n"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
