"""Where flex_attention is a compiled kernel, the log-prob forward runs in the training forward's grad mode."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

trainer_module = pytest.importorskip("miles.backends.torchtitan_utils.trainer")


@pytest.mark.parametrize("flex_attention", [True, False], ids=["cuda", "tpu"])
def test_the_log_prob_forward_takes_the_platforms_grad_mode_and_detaches(monkeypatch, flex_attention):
    grad_enabled, preds = [], []
    weight = torch.nn.Parameter(torch.ones(2))

    def model(inputs):
        grad_enabled.append(torch.is_grad_enabled())
        return inputs * weight

    monkeypatch.setattr(trainer_module.routing_replay, "consumption_guard", lambda parts, n: nullcontext())
    monkeypatch.setattr(trainer_module.platforms, "current_platform", SimpleNamespace(flex_attention=flex_attention))
    trainer = trainer_module.TitanTrainer.__new__(trainer_module.TitanTrainer)
    trainer.model_parts = [model]
    trainer.loss_fn = type(
        "LossFn",
        (),
        {
            "arm": lambda self, batches, compute, is_training: None,
            "__call__": lambda self, pred, label: preds.append(pred),
            "collect": lambda self: ["collected"],
        },
    )()
    trainer.parallelism = SimpleNamespace(pp_enabled=False, activate_spmd=nullcontext)
    trainer._microbatch_inputs = lambda batches: [(torch.ones(2), None, {}, None) for _ in batches]
    trainer._reshard_kept_params = lambda: None
    trainer.has_last_stage = lambda: True

    with torch.no_grad():
        assert trainer.run_forward([object(), object()], compute=None) == ["collected"]

    assert grad_enabled == [flex_attention, flex_attention]
    assert all(not pred.requires_grad for pred in preds)
