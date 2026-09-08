"""The shared forward objectives preserve the training gradients."""

import math
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import tinker_losses


@pytest.mark.parametrize(
    "loss_fn, expected_loss, expected_grad",
    [
        ("cross_entropy", -2 * math.log(0.25) + 0.5 * math.log(0.5), [-2.0, 0.5]),
        ("importance_sampling", -3.5, [0.5, -4.0]),
        ("ppo", -2.25, [0.0, 0.0]),
        ("cispo", 0.75 * math.log(0.25) - 3 * math.log(0.5), [0.75, -3.0]),
        ("dro", 0.3 * math.log(2) ** 2, [1 - 0.3 * math.log(2), -2 + 0.3 * math.log(2)]),
    ],
)
def test_training_loss_value_and_gradient(monkeypatch, loss_fn, expected_loss, expected_grad):
    logprobs = torch.tensor([math.log(0.25), math.log(0.5)], requires_grad=True)
    batch = {
        "sample_indices": [4],
        "loss_weights": [[2.0, -0.5]],
        "advantages": [[-1.0, 2.0]],
        "rollout_log_probs": [[math.log(0.5), math.log(0.25)]],
        "loss_fn_config": {"clip_low_threshold": 0.75, "clip_high_threshold": 1.5, "beta": 0.3},
    }
    monkeypatch.setattr(tinker_losses, "_target_logprobs", lambda args, batch, logits: [logprobs])
    tinker_losses.start_per_datum_outputs()
    try:
        loss, metrics = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn](Namespace(), batch, logprobs, None)
    finally:
        per_datum = tinker_losses.drain_per_datum_outputs()

    assert float(loss.detach()) == pytest.approx(expected_loss, abs=1e-6)
    assert float(metrics["loss"]) == pytest.approx(expected_loss, abs=1e-6)
    assert per_datum[0]["sample_index"] == 4
    assert float(per_datum[0]["loss"]) == pytest.approx(expected_loss, abs=1e-6)
    assert not per_datum[0]["logprobs"].requires_grad
    loss.backward()
    assert logprobs.grad.tolist() == pytest.approx(expected_grad, abs=1e-6)
