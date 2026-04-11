"""Example: Using rlwatch with HuggingFace TRL.

This shows how to integrate rlwatch into a TRL GRPO training script.
rlwatch auto-detects TRL and registers a TrainerCallback.

Prerequisites:
    pip install rlwatch[trl]
"""

# Step 1: Import and attach rlwatch BEFORE creating the trainer
import rlwatch
monitor = rlwatch.attach()  # Auto-detects TRL

# Step 2: Your normal TRL training code
# from trl import GRPOTrainer, GRPOConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model = AutoModelForCausalLM.from_pretrained("your-model")
# tokenizer = AutoTokenizer.from_pretrained("your-model")
#
# training_args = GRPOConfig(
#     output_dir="./output",
#     per_device_train_batch_size=4,
#     num_train_epochs=1,
#     learning_rate=1e-5,
# )
#
# trainer = GRPOTrainer(
#     model=model,
#     args=training_args,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     reward_model=reward_model,
# )
#
# # If auto-detection didn't find the trainer, add the callback manually:
# # trainer.add_callback(monitor._trl_callback_class())
#
# trainer.train()
# # rlwatch monitors entropy, KL, rewards automatically via on_log callback

print("TRL integration example (see comments in source for full usage)")
monitor.stop()
