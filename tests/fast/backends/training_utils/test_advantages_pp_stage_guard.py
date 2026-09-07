from argparse import Namespace

import torch

from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _set_parallel_state(is_pp_last_stage: bool) -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            pp=GroupInfo(rank=0 if not is_pp_last_stage else 1, size=2, group=None),
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
            is_pp_last_stage=is_pp_last_stage,
        )
    )


def _ppo_args(use_rollout_logprobs: bool) -> Namespace:
    return Namespace(
        advantage_estimator="ppo",
        use_rollout_logprobs=use_rollout_logprobs,
        skip_actor_forward_only=False,
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
        qkv_format="thd",
        use_opd=False,
        normalize_advantages=False,
    )


def _rollout_data(log_probs_key: str, with_values: bool) -> dict:
    response_length = 4
    data = {
        log_probs_key: [torch.zeros(response_length)],
        "ref_log_probs": [torch.zeros(response_length)],
        "rewards": [1.0],
        "response_lengths": [response_length],
        "loss_masks": [torch.ones(response_length)],
        "total_lengths": [response_length + 2],
    }
    if with_values:
        data["values"] = [torch.zeros(response_length)]
    return data


def test_intermediate_pp_stage_returns_early_with_rollout_log_probs() -> None:
    """Rollout log-probs exist on every pipeline stage; only the last stage owns values.

    The stage decision must come from the parallel state, not from `log_probs`
    and `values` both being absent: under --use-rollout-logprobs an
    intermediate stage has log-probs but no values, and running the PPO
    estimator there dereferences ``values``.
    """
    _set_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data("rollout_log_probs", with_values=False)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert "advantages" not in rollout_data
    assert "returns" not in rollout_data


def test_intermediate_pp_stage_returns_early_without_any_log_probs() -> None:
    _set_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data("log_probs", with_values=False)
    del rollout_data["log_probs"]

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=False), rollout_data)

    assert "advantages" not in rollout_data


def test_last_pp_stage_computes_advantages_with_rollout_log_probs() -> None:
    _set_parallel_state(is_pp_last_stage=True)
    rollout_data = _rollout_data("rollout_log_probs", with_values=True)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert len(rollout_data["advantages"]) == 1
    assert len(rollout_data["returns"]) == 1
    assert rollout_data["advantages"][0].shape == (4,)
