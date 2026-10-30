"""The torchtitan log-prob forward runs like the training forward, so both produce the same log-probs."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

trainer_module = pytest.importorskip("miles.backends.torchtitan_utils.trainer")


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.grad_enabled: list[bool] = []

    def forward(self, inputs):
        self.grad_enabled.append(torch.is_grad_enabled())
        return inputs * self.weight


class _LossFn:
    def __init__(self):
        self.preds = []

    def arm(self, batches, closure, *, is_training):
        pass

    def __call__(self, pred, label):
        self.preds.append(pred)

    def collect(self):
        return ["collected"]


def test_the_log_prob_forward_runs_under_grad_and_hands_the_loss_a_detached_prediction(monkeypatch):
    monkeypatch.setattr(trainer_module.routing_replay, "consumption_guard", lambda parts, n: nullcontext())
    trainer = trainer_module.TitanTrainer.__new__(trainer_module.TitanTrainer)
    model, loss_fn = _RecordingModel(), _LossFn()
    trainer.model_parts, trainer.loss_fn = [model], loss_fn
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer._microbatch_inputs = lambda batches: ([{"tokens": torch.ones(2)}] * len(batches), [None] * len(batches))
    trainer.post_dataloading_process = lambda input_dict, label: (input_dict["tokens"], label, {})
    trainer.has_last_stage = lambda: True

    with torch.no_grad():
        assert trainer.run_forward([object(), object()], compute=None) == ["collected"]

    assert model.grad_enabled == [True, True]
    assert all(not pred.requires_grad for pred in loss_fn.preds)
