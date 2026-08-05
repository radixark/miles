from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub import losses as losses_module
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.losses import policy_loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data


def _run_policy_loss(args, batch, inputs):
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("skip-actor-logprobs") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("advantage_estimator", ["grpo", "gspo"])
def test_policy_loss_uses_detached_training_log_probs_as_single_step_baseline(
    process_group,
    advantage_estimator,
    monkeypatch,
):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        advantage_estimator=advantage_estimator,
        entropy_coef=0.0,
        kl_coef=0.0,
        skip_actor_logprobs_forward=True,
        true_on_policy_mode=False,
    )
    inputs = make_inputs(
        seed=1234,
        batch_size=3,
        prompt_lens=[20, 64, 40],
        response_lens=[10, 48, 32],
        vocab_size=128,
        args=args,
    )

    gather_spy = Mock(wraps=losses_module.all_gather_with_cp)
    if advantage_estimator == "gspo":
        monkeypatch.setattr(losses_module, "all_gather_with_cp", gather_spy)

    skip_batch = make_batch(inputs, "policy_loss")
    del skip_batch["log_probs"]
    skip_loss, skip_metrics, skip_grad = _run_policy_loss(args, skip_batch, inputs)

    if advantage_estimator == "gspo":
        assert gather_spy.call_count == len(inputs["response_lens"])

    baseline_args = make_args(
        advantage_estimator=advantage_estimator,
        entropy_coef=0.0,
        true_on_policy_mode=False,
        kl_coef=0.0,
    )
    baseline_batch = make_batch(inputs, "policy_loss")
    baseline_batch["log_probs"] = [
        value.detach()
        for value in get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=baseline_args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    ]
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(baseline_args, baseline_batch, inputs)

    assert torch.equal(skip_loss, baseline_loss)
    assert torch.equal(skip_grad, baseline_grad)
    assert torch.count_nonzero(skip_grad).item() > 0
    assert skip_metrics["ppo_kl"].item() == 0.0
    assert skip_metrics["pg_clipfrac"].item() == 0.0
    assert baseline_metrics["ppo_kl"].item() == 0.0


def test_advantages_use_response_shapes_when_actor_log_probs_are_skipped():
    parallel_state = make_parallel_state()
    args = make_args(skip_actor_logprobs_forward=True, true_on_policy_mode=False, kl_coef=0.0)
    inputs = make_inputs(
        seed=1234,
        batch_size=3,
        prompt_lens=[20, 64, 40],
        response_lens=[10, 48, 32],
        vocab_size=128,
        args=args,
    )

    skipped = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs", "values"):
        del skipped[key]
    compute_advantages_and_returns(args, skipped)

    baseline = make_rollout_data(inputs)
    del baseline["values"]
    compute_advantages_and_returns(make_args(true_on_policy_mode=False, kl_coef=0.0), baseline)

    assert parallel_state.is_pp_last_stage
    for actual, expected in zip(skipped["advantages"], baseline["advantages"], strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(skipped["returns"], baseline["returns"], strict=True):
        assert torch.equal(actual, expected)


def test_ppo_advantages_use_critic_values_when_actor_log_probs_are_skipped():
    make_parallel_state()
    args = make_args(
        advantage_estimator="ppo",
        skip_actor_logprobs_forward=True,
        true_on_policy_mode=False,
        kl_coef=0.0,
    )
    inputs = make_inputs(
        seed=1234,
        batch_size=3,
        prompt_lens=[20, 64, 40],
        response_lens=[10, 48, 32],
        vocab_size=128,
        args=args,
    )

    skipped = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs"):
        del skipped[key]
    compute_advantages_and_returns(args, skipped)

    baseline = make_rollout_data(inputs)
    compute_advantages_and_returns(
        make_args(advantage_estimator="ppo", true_on_policy_mode=False, kl_coef=0.0), baseline
    )

    for key in ("advantages", "returns"):
        for actual, expected in zip(skipped[key], baseline[key], strict=True):
            assert torch.equal(actual, expected)


def test_skipped_log_probs_do_not_compute_advantages_on_intermediate_pp_stage():
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = False
    args = make_args(skip_actor_logprobs_forward=True, true_on_policy_mode=False, kl_coef=0.0)
    inputs = make_inputs(
        seed=1234,
        batch_size=1,
        prompt_lens=[20],
        response_lens=[10],
        vocab_size=128,
        args=args,
    )
    rollout_data = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs", "values"):
        del rollout_data[key]

    compute_advantages_and_returns(args, rollout_data)

    assert "advantages" not in rollout_data
    assert "returns" not in rollout_data
    make_parallel_state()
