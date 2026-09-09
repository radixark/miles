"""Per-datum outputs belong to their loss pass, including when backward recomputes the loss."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import loss as loss_module
from miles.backends.training_utils.loss_hub import tinker_losses


@pytest.mark.parametrize("recompute", [False, True], ids=["direct", "recomputed"])
def test_loss_passes_return_independent_detached_outputs(monkeypatch, recompute):
    parallel = SimpleNamespace(cp=SimpleNamespace(size=1), intra_dp=SimpleNamespace(size=1))
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tinker_losses, "_target_logprobs", lambda _args, _batch, logits: [logits])
    args = Namespace(
        calculate_per_token_loss=False,
        qkv_format="thd",
        recompute_loss_function=recompute,
        use_dynamic_global_batch_size=True,
        global_batch_size=1,
        multi_lora=True,
    )
    completed = []
    for sample_index in (7, 11):
        logprobs = torch.tensor([-0.5, -0.25], requires_grad=True)
        batch = {
            "loss_fn": "cross_entropy",
            "loss_weights": [[2.0, 3.0]],
            "loss_masks": [torch.ones(2)],
            "total_lengths": [3],
            "response_lengths": [2],
            "sample_indices": [sample_index],
            "dynamic_global_batch_size": 1,
        }
        loss, _, logging = loss_module.loss_function(args, batch, 1, logprobs)
        loss.backward()
        assert logprobs.grad.tolist() == [-2.0, -3.0]
        assert logging["keys"] == ["loss"]
        assert logging["values"].tolist() == [1.0, 1.75]
        completed.append(logging["per_datum"])

    assert [[output["sample_index"] for output in outputs] for outputs in completed] == [[7], [11]]
    for outputs in completed:
        assert len(outputs) == 1, "loss recomputation must not append another datum output"
        assert outputs[0]["loss"].item() == 1.75
        assert outputs[0]["logprobs"].tolist() == [-0.5, -0.25]
        assert not outputs[0]["loss"].requires_grad
        assert not outputs[0]["logprobs"].requires_grad
