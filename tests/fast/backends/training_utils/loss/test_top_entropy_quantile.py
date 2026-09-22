from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss_hub import losses as losses_module
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_top_entropy_mask

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("top-entropy-quantile") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _run_policy_loss(args, inputs):
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _make(quantile: float):
    make_parallel_state()
    args = make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        observe_training_entropy=False,
        top_entropy_quantile=quantile,
    )
    inputs = make_inputs(
        seed=7,
        batch_size=2,
        prompt_lens=[16, 24],
        response_lens=[12, 20],
        vocab_size=64,
        args=args,
    )
    return args, inputs


def test_quantile_one_leaves_the_policy_loss_untouched(process_group, monkeypatch):
    spy = Mock(wraps=losses_module.compute_top_entropy_mask)
    monkeypatch.setattr(losses_module, "compute_top_entropy_mask", spy)

    args, inputs = _make(quantile=1.0)
    loss, metrics, grad = _run_policy_loss(args, inputs)

    baseline_args = make_args(advantage_estimator="grpo", entropy_coef=0.0, observe_training_entropy=False)
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(baseline_args, inputs)

    assert spy.call_count == 0
    assert torch.equal(loss, baseline_loss)
    assert torch.equal(grad, baseline_grad)
    assert torch.equal(metrics["pg_loss"], baseline_metrics["pg_loss"])


def test_quantile_below_one_masks_low_entropy_tokens(process_group, monkeypatch):
    returned = []

    def recording_mask(*call_args):
        mask = compute_top_entropy_mask(*call_args)
        returned.append(mask)
        return mask

    spy = Mock(side_effect=recording_mask)
    monkeypatch.setattr(losses_module, "compute_top_entropy_mask", spy)

    args, inputs = _make(quantile=0.25)
    loss, metrics, grad = _run_policy_loss(args, inputs)
    full_args, full_inputs = _make(quantile=1.0)
    full_loss, _, full_grad = _run_policy_loss(full_args, full_inputs)

    assert spy.call_count == 1
    (entropy, loss_mask, quantile), _ = spy.call_args
    assert quantile == 0.25
    assert entropy.shape == loss_mask.shape == (sum(inputs["response_lens"]),)
    kept = returned[0]
    # A quarter of 32 response tokens survive the threshold.
    assert kept.sum().item() == 8
    # Entropy is computed for the mask even though no entropy bonus or metric asked for it.
    assert metrics["entropy_loss"].item() > 0
    assert not torch.equal(loss, full_loss)
    assert not torch.equal(grad, full_grad)
