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


@pytest.mark.parametrize("loss_fn", sorted(tinker_losses.TINKER_LOSS_FUNCTIONS))
def test_a_zero_loss_mask_removes_the_datum_from_every_objective(monkeypatch, loss_fn):
    logprobs = torch.tensor([-0.5, -0.25], requires_grad=True)
    monkeypatch.setattr(tinker_losses, "_target_logprobs", lambda _args, _batch, logits: [logits])
    batch = {
        "loss_fn": loss_fn,
        "loss_weights": [[2.0, 3.0]],
        "advantages": [[1.0, 1.0]],
        "rollout_log_probs": [[-1.5, -1.25]],  # differs from logprobs so the DRO divergence term is nonzero
        "loss_masks": [torch.zeros(2)],
        "total_lengths": [3],
        "response_lengths": [2],
        "sample_indices": [0],
    }
    loss, _ = tinker_losses.TINKER_LOSS_FUNCTIONS[loss_fn](Namespace(), batch, logprobs, None)
    loss.backward()
    assert loss.item() == 0.0
    assert logprobs.grad.abs().sum().item() == 0.0, "a DP-padding datum must contribute no gradient"


@pytest.mark.parametrize("recompute", [False, True], ids=["direct", "recomputed"])
@pytest.mark.parametrize(
    "loss_fn, config, token_losses, gradients",
    [
        ("importance_sampling", {}, [-1, 1.5, -2, 3, -10, 15], [-1, 1.5, -2, 3, -10, 15]),
        ("ppo", {}, [-1, 2.4, -2, 3, -2.4, 15], [-1, 0, -2, 3, 0, 15]),
        (
            "ppo",
            {"clip_low_threshold": 0.4, "clip_high_threshold": 6},
            [-1, 1.5, -2, 3, -10, 15],
            [-1, 1.5, -2, 3, -10, 15],
        ),
        ("cispo", {}, [1, -1.5, 2, -3, 8, -12], [-1, 1.5, -2, 3, -8, 12]),
        (
            "cispo",
            {"clip_low_threshold": 0.75, "clip_high_threshold": 2},
            [1.5, -2.25, 2, -3, 4, -6],
            [-1.5, 2.25, -2, 3, -4, 6],
        ),
        ("dro", {}, None, None),
        ("dro", {"beta": 0.5}, None, None),
    ],
    ids=["is", "ppo-default", "ppo-override", "cispo-default", "cispo-override", "dro-default", "dro-override"],
)
def test_nonzero_objectives_and_gradients(monkeypatch, recompute, loss_fn, config, token_losses, gradients):
    parallel = SimpleNamespace(cp=SimpleNamespace(size=1), intra_dp=SimpleNamespace(size=1))
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tinker_losses, "_target_logprobs", lambda _args, _batch, logits: [logits[:3], logits[3:]])
    args = Namespace(
        calculate_per_token_loss=False,
        qkv_format="thd",
        recompute_loss_function=recompute,
        use_dynamic_global_batch_size=True,
        global_batch_size=4,
        multi_lora=True,
    )
    # Both advantage signs cross each clipping boundary; the last two tokens are masked.
    ratios = torch.tensor([0.5, 0.5, 1, 1, 5, 5, 0.5, 5], dtype=torch.float64)
    logprobs = torch.full((8,), -1.0, dtype=torch.float64, requires_grad=True)
    sampling_logprobs = -1 - ratios.log()
    batch = {
        "loss_fn": loss_fn,
        "loss_fn_config": config,
        "advantages": [[2, -3, 2], [-3, 2, -3, 11, -13]],
        "rollout_log_probs": [sampling_logprobs[:3], sampling_logprobs[3:]],
        "loss_masks": [torch.ones(3), torch.tensor([1, 1, 1, 0, 0])],
        "total_lengths": [4, 6],
        "response_lengths": [3, 5],
        "sample_indices": [7, 11],
        "dynamic_global_batch_size": 4,
    }
    if loss_fn == "dro":
        beta = 0.05 if not config else 0.5
        advantages = torch.tensor([2, -3, 2, -3, 2, -3], dtype=torch.float64)
        divergence = ratios[:6].log()
        token_losses = (advantages + beta / 2 * divergence.square()).tolist()
        gradients = (-advantages + beta * divergence).tolist()
    expected_losses = torch.tensor([sum(token_losses[:3]), sum(token_losses[3:])], dtype=torch.float64)
    loss, _, logging = loss_module.loss_function(args, batch, 1, logprobs)
    loss.backward()
    torch.testing.assert_close(loss, expected_losses.sum())
    torch.testing.assert_close(logprobs.grad, torch.tensor(gradients + [0, 0], dtype=torch.float64))
    outputs = logging["per_datum"]
    assert [output["sample_index"] for output in outputs] == [7, 11]
    torch.testing.assert_close(torch.stack([output["loss"] for output in outputs]), expected_losses)
    torch.testing.assert_close(torch.cat([output["logprobs"] for output in outputs]), logprobs.detach())
    assert all(not output["loss"].requires_grad and not output["logprobs"].requires_grad for output in outputs)
