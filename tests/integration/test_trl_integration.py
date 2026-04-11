"""Integration test for the TRL ``Trainer`` callback path.

Skipped automatically when ``transformers``/``trl`` aren't installed. The
plan gates v0.2.0 on this passing in CI under the ``[trl]`` extras job.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.trl]

transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")


def test_attach_to_trainer_registers_callback(tmp_log_dir, monkeypatch):
    """Ensure ``attach(trainer=...)`` adds our callback without raising.

    We don't actually run training (would need a model + dataset + GPU). The
    invariant is: passing a Trainer to attach() registers the callback class
    via ``trainer.add_callback`` and the callback ends up in
    ``trainer.callback_handler.callbacks``.
    """
    monkeypatch.setenv("RLWATCH_LOG_DIR", tmp_log_dir)

    from transformers import TrainingArguments
    from transformers.trainer_callback import CallbackHandler

    # Build a Trainer-shaped object cheap. The Trainer constructor is heavy;
    # we use TrainingArguments + a stub callback handler so we can call
    # ``add_callback`` without a real model.
    handler = CallbackHandler(callbacks=[], model=None, processing_class=None,
                               optimizer=None, lr_scheduler=None)

    class _StubTrainer:
        def __init__(self):
            self.callback_handler = handler

        def add_callback(self, callback):
            self.callback_handler.add_callback(callback)

    import rlwatch
    trainer = _StubTrainer()
    monitor = rlwatch.attach(framework="trl", trainer=trainer, run_id="trl_test")
    try:
        from rlwatch.core import _build_trl_callback
        callback_cls = _build_trl_callback(monitor)
        assert any(
            isinstance(cb, callback_cls.__bases__[0])  # TrainerCallback
            for cb in handler.callbacks
        )
    finally:
        monitor.stop()
