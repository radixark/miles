from argparse import Namespace

import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub.advantages import compute_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _parallel_state(rank: int = 0, world_size: int = 1) -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    cp_group = dist.group.WORLD if world_size > 1 else None
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        cp=GroupInfo(rank=rank, size=world_size, group=cp_group),
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _run_ppo_case(rank: int, total_length: int, response_length: int, expected_local_sizes: list[int]) -> None:
    args = Namespace(advantage_estimator="ppo", kl_coef=0.1, gamma=0.0, lambd=0.0, qkv_format="thd")
    full_kl = torch.arange(1, response_length + 1, dtype=torch.float32)
    full_values = torch.zeros(response_length)

    set_parallel_state(_parallel_state(rank=rank, world_size=2))
    local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length)
    local_values = slice_log_prob_with_cp(full_values, total_length, response_length)
    assert local_kl.numel() == expected_local_sizes[rank]

    advantages, returns = compute_advantages(
        args=args,
        kl=[local_kl],
        rewards=[10.0],
        log_probs=[torch.empty_like(local_kl)],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
        max_seq_lens=None,
        values=[local_values],
    )
    cp_advantages = all_gather_with_cp(advantages[0], total_length, response_length)
    cp_returns = all_gather_with_cp(returns[0], total_length, response_length)

    set_parallel_state(_parallel_state())
    baseline_advantages, baseline_returns = compute_advantages(
        args=args,
        kl=[full_kl.clone()],
        rewards=[10.0],
        log_probs=[torch.empty_like(full_kl)],
        loss_masks=[torch.ones(response_length)],
        total_lengths=[total_length],
        response_lengths=[response_length],
        max_seq_lens=None,
        values=[full_values],
    )

    expected = -0.1 * full_kl
    expected[-1] += 10.0
    torch.testing.assert_close(cp_advantages, expected)
    torch.testing.assert_close(cp_returns, expected)
    torch.testing.assert_close(cp_advantages, baseline_advantages[0])
    torch.testing.assert_close(cp_returns, baseline_returns[0])


def _worker_tail_on_rank_one(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=6, expected_local_sizes=[2, 4])
    finally:
        dist.destroy_process_group()


def _worker_empty_rank_zero(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_ppo_case(rank, total_length=7, response_length=2, expected_local_sizes=[0, 2])
    finally:
        dist.destroy_process_group()


def _worker_bshd_layout_metadata(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        args = Namespace(
            advantage_estimator="ppo",
            use_rollout_logprobs=False,
            kl_coef=0.1,
            kl_loss_type="k1",
            gamma=0.0,
            lambd=0.0,
            qkv_format="bshd",
            use_opd=False,
            normalize_advantages=False,
        )
        total_lengths = [8, 11]
        response_lengths = [5, 6]
        max_seq_lens = [12, 12]
        rewards = [10.0, 20.0]
        full_log_probs = [
            torch.arange(1, response_length + 1, dtype=torch.float32) for response_length in response_lengths
        ]
        full_ref_log_probs = [torch.zeros_like(log_probs) for log_probs in full_log_probs]
        full_values = [torch.zeros_like(log_probs) for log_probs in full_log_probs]

        set_parallel_state(_parallel_state(rank=rank, world_size=world_size))
        local_log_probs = [
            slice_log_prob_with_cp(log_probs, total_length, response_length, "bshd", max_seq_len)
            for log_probs, total_length, response_length, max_seq_len in zip(
                full_log_probs, total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]
        local_ref_log_probs = [torch.zeros_like(log_probs) for log_probs in local_log_probs]
        local_values = [torch.zeros_like(log_probs) for log_probs in local_log_probs]
        expected_local_sizes = [[1, 1], [4, 5]][rank]
        assert [tensor.numel() for tensor in local_log_probs] == expected_local_sizes

        rollout_data = {
            "log_probs": local_log_probs,
            "ref_log_probs": local_ref_log_probs,
            "rewards": rewards,
            "values": local_values,
            "response_lengths": response_lengths,
            "loss_masks": [torch.ones(response_length) for response_length in response_lengths],
            "total_lengths": total_lengths,
            "max_seq_lens": max_seq_lens,
        }
        compute_advantages_and_returns(args, rollout_data)
        cp_advantages = [
            all_gather_with_cp(advantage, total_length, response_length, "bshd", max_seq_len)
            for advantage, total_length, response_length, max_seq_len in zip(
                rollout_data["advantages"], total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]
        cp_returns = [
            all_gather_with_cp(ret, total_length, response_length, "bshd", max_seq_len)
            for ret, total_length, response_length, max_seq_len in zip(
                rollout_data["returns"], total_lengths, response_lengths, max_seq_lens, strict=True
            )
        ]

        set_parallel_state(_parallel_state())
        baseline_data = {
            "log_probs": [tensor.clone() for tensor in full_log_probs],
            "ref_log_probs": [tensor.clone() for tensor in full_ref_log_probs],
            "rewards": rewards,
            "values": [tensor.clone() for tensor in full_values],
            "response_lengths": response_lengths,
            "loss_masks": [torch.ones(response_length) for response_length in response_lengths],
            "total_lengths": total_lengths,
            "max_seq_lens": max_seq_lens,
        }
        compute_advantages_and_returns(args, baseline_data)

        for cp_advantage, cp_return, baseline_advantage, baseline_return in zip(
            cp_advantages,
            cp_returns,
            baseline_data["advantages"],
            baseline_data["returns"],
            strict=True,
        ):
            torch.testing.assert_close(cp_advantage, baseline_advantage)
            torch.testing.assert_close(cp_return, baseline_return)
    finally:
        dist.destroy_process_group()


def test_ppo_terminal_reward_is_added_to_global_response_tail() -> None:
    run_multiprocess(_worker_tail_on_rank_one)


def test_ppo_terminal_reward_handles_empty_rank_zero_shard() -> None:
    run_multiprocess(_worker_empty_rank_zero)


def test_ppo_bshd_cp_uses_padded_layout_metadata() -> None:
    run_multiprocess(_worker_bshd_layout_metadata)
