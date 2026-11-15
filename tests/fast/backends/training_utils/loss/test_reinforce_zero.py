"""REINFORCE Zero: group-mean advantages with unclipped token-level importance sampling.

The loss is `-mean_responses(mean_tokens(A * pi_theta / mu))`, with `mu` the rollout log-probs
(https://github.com/hijkzzz/REINFORCE-Zero, ALGORITHM.md).
"""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns
from miles.utils.arguments import _validate_reinforce_zero_args
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("reinforce-zero") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _setup(**overrides):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        advantage_estimator="reinforce_zero",
        use_rollout_logprobs=True,
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        **overrides,
    )
    inputs = make_inputs(
        seed=7,
        batch_size=3,
        prompt_lens=[12, 30, 20],
        response_lens=[6, 24, 16],
        vocab_size=64,
        args=args,
    )
    return args, inputs


def _reducer(args, batch):
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.backend.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )


def _run(args, inputs, batch):
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    loss, metrics = policy_loss_function(args, batch, logits, _reducer(args, batch))
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def _reference(args, inputs, batch):
    """The paper's loss written out directly: per response, mean over tokens of -A * exp(log pi - log mu)."""
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    log_probs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )["log_probs"]
    per_response = [
        (-(advantage * torch.exp(log_prob - rollout_log_prob)) * mask).sum() / mask.sum()
        for log_prob, rollout_log_prob, advantage, mask in zip(
            log_probs, batch["rollout_log_probs"], batch["advantages"], batch["loss_masks"], strict=True
        )
    ]
    loss = torch.stack(per_response).sum()
    loss.backward()
    return loss.detach(), logits.grad.clone()


def test_loss_and_gradient_match_the_paper_formula(process_group):
    args, inputs = _setup()
    batch = make_batch(inputs, "policy_loss")
    # mask a few tokens out so the per-response token mean is exercised
    batch["loss_masks"][1][:5] = 0.0

    loss, metrics, grad = _run(args, inputs, batch)
    ref_loss, ref_grad = _reference(args, inputs, batch)

    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(grad, ref_grad)
    assert metrics["pg_clipfrac"].item() == 0.0


def test_importance_weight_is_not_clipped(process_group):
    # every rollout log-prob is log(4) below the learner's, so pi_theta / mu = 4 everywhere: far outside
    # PPO's [1 - eps_clip, 1 + eps_clip_high], yet the loss keeps the full weight
    args, inputs = _setup(eps_clip=0.2, eps_clip_high=0.2)
    batch = make_batch(inputs, "policy_loss")
    learner_log_probs = get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )["log_probs"]
    batch["rollout_log_probs"] = [log_prob.detach() - torch.log(torch.tensor(4.0)) for log_prob in learner_log_probs]
    batch["advantages"] = [torch.ones_like(advantage) for advantage in batch["advantages"]]

    loss, _, _ = _run(args, inputs, batch)

    # 3 responses, each with token mean -A * w = -1 * 4
    torch.testing.assert_close(loss, torch.tensor(-12.0))


def test_advantage_is_the_centered_reward_on_every_token(process_group):
    args, inputs = _setup()
    rollout_data = make_rollout_data(inputs)
    rollout_data["rewards"] = [0.5, -1.0, 2.0]  # already group-centered by reward post-processing

    compute_advantages_and_returns(args, rollout_data)

    for advantage, reward, length in zip(
        rollout_data["advantages"], rollout_data["rewards"], inputs["response_lens"], strict=True
    ):
        torch.testing.assert_close(advantage, torch.full((length,), reward))


def _valid_args(**overrides) -> Namespace:
    values = dict(
        loss_type="policy_loss",
        use_rollout_logprobs=True,
        rewards_normalization=True,
        normalize_advantages=False,
        use_tis=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        eps_clip_c=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
    )
    return Namespace(**(values | overrides))


def test_validation_accepts_the_recipe():
    _validate_reinforce_zero_args(_valid_args())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_rollout_logprobs": False}, "--use-rollout-logprobs"),
        ({"rewards_normalization": False}, "group mean"),
        ({"normalize_advantages": True}, "--normalize-advantages"),
        ({"use_tis": True}, "--use-tis"),
        ({"use_opsm": True}, "--use-opsm"),
        ({"get_mismatch_metrics": True}, "--get-mismatch-metrics"),
        ({"eps_clip_c": 3.0}, "--eps-clip-c"),
        ({"custom_pg_loss_reducer_function_path": "my.reducer"}, "averages tokens within each response"),
        ({"calculate_per_token_loss": True}, "--calculate-per-token-loss"),
        ({"loss_type": "sft_loss"}, "--loss-type policy_loss"),
    ],
)
def test_validation_rejects_settings_that_change_the_loss(override, message):
    with pytest.raises(ValueError, match="reinforce_zero") as excinfo:
        _validate_reinforce_zero_args(_valid_args(**override))
    assert message in str(excinfo.value)
