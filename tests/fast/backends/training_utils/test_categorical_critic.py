from argparse import Namespace

import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp
from miles.backends.training_utils.loss.hub.logit_processors import get_values
from miles.backends.training_utils.loss.hub.losses import value_loss_function
from miles.backends.training_utils.loss.hub.math_utils import get_advantages_and_returns_batch
from miles.backends.training_utils.loss.hub.value_distribution import hl_gauss_loss
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _args() -> Namespace:
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=group,
            tp=group,
            pp=group,
            ep=group,
            etp=group,
            indep_dp=group,
        )
    )
    return Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        allgather_cp=False,
        bootstrap_truncated=True,
        critic_value_bins=3,
        critic_value_min=0.0,
        critic_value_max=3.0,
        critic_value_sigma=0.25,
    )


def test_categorical_values_and_final_state_bootstrap() -> None:
    args = _args()
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]])
    result = get_values(
        logits,
        args=args,
        unconcat_tokens=[torch.arange(5)],
        total_lengths=[5],
        response_lengths=[4],
        truncated=[1],
        include_logits=True,
    )
    torch.testing.assert_close(result["values"][0], torch.tensor([0.5, 1.5, 2.5, 0.5]), atol=2e-4, rtol=0)
    torch.testing.assert_close(result["bootstrap_values"][0], torch.tensor([2.5]), atol=2e-4, rtol=0)
    assert result["value_logits"][0].shape == (4, 3)
    completed = get_values(
        logits,
        args=args,
        unconcat_tokens=[torch.arange(5)],
        total_lengths=[5],
        response_lengths=[4],
        truncated=[0],
    )
    torch.testing.assert_close(completed["bootstrap_values"][0], torch.tensor([0.0]))


def test_hl_gauss_loss_has_finite_tail_and_gradient() -> None:
    args = _args()
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = hl_gauss_loss(logits, torch.tensor([1.5, 100.0]), args).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] < 0
    assert logits.grad[1, 2] < 0


def test_categorical_value_loss_updates_response_positions() -> None:
    args = _args()
    logits = torch.zeros(1, 5, 3, requires_grad=True)
    batch = {
        "values": [torch.zeros(4)],
        "returns": [torch.tensor([0.5, 1.5, 2.5, 0.5])],
        "unconcat_tokens": [torch.arange(5)],
        "total_lengths": [5],
        "response_lengths": [4],
    }
    loss, metrics = value_loss_function(args, batch, logits, torch.mean)
    loss.backward()
    assert torch.isfinite(loss)
    assert "value_loss" in metrics
    torch.testing.assert_close(metrics["value_out_of_support"], torch.tensor(0.0))
    assert logits.grad[0, :4].abs().sum() > 0
    assert logits.grad[0, 4].abs().sum() == 0


def test_scalar_critic_retains_clipped_mse() -> None:
    args = _args()
    args.critic_value_bins = 1
    args.value_clip = 0.2
    batch = {
        "values": [torch.zeros(4)],
        "returns": [torch.ones(4)],
        "unconcat_tokens": [torch.arange(5)],
        "total_lengths": [5],
        "response_lengths": [4],
    }
    loss, metrics = value_loss_function(args, batch, torch.full((1, 5, 1), 0.5), torch.mean)
    torch.testing.assert_close(loss, torch.tensor(0.64))
    torch.testing.assert_close(metrics["value_clipfrac"], torch.tensor(1.0))


def _check_cp_bootstrap(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        single = GroupInfo(rank=0, size=1, group=None)
        cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        set_parallel_state(
            ParallelState(
                intra_dp=single,
                intra_dp_cp=cp,
                cp=cp,
                tp=single,
                pp=single,
                ep=single,
                etp=single,
                indep_dp=single,
            )
        )
        for allgather_cp, positions in [
            (False, ([0, 1, 6, 7] if rank == 0 else [2, 3, 4, 5])),
            (True, list(range(rank * 4, rank * 4 + 4))),
        ]:
            args = Namespace(
                qkv_format="thd",
                true_on_policy_mode=False,
                allgather_cp=allgather_cp,
                bootstrap_truncated=True,
                critic_value_bins=1,
            )
            result = get_values(
                torch.tensor(positions, dtype=torch.float32).view(1, 4, 1),
                args=args,
                unconcat_tokens=[torch.arange(8)],
                total_lengths=[8],
                response_lengths=[4],
                truncated=[1],
            )
            torch.testing.assert_close(result["bootstrap_values"][0], torch.tensor([7.0]))
            _, returns = get_advantages_and_returns_batch(
                total_lengths=[8],
                response_lengths=[4],
                values_list=result["values"],
                rewards_list=[torch.zeros_like(result["values"][0])],
                terminal_rewards=[3.0],
                qkv_format="thd",
                max_seq_lens=None,
                loss_masks=[torch.ones(4)],
                gamma=0.9,
                lambd=1.0,
                bootstrap_values=result["bootstrap_values"],
            )
            full_returns = all_gather_with_cp(returns[0], 8, 4)
            torch.testing.assert_close(full_returns, torch.tensor([6.7797, 7.533, 8.37, 9.3]))
            if allgather_cp:
                args.critic_value_bins = 3
                args.critic_value_min = 0.0
                args.critic_value_max = 3.0
                categorical_logits = (
                    torch.nn.functional.one_hot(torch.tensor(positions) % 3, num_classes=3)
                    .float()
                    .view(1, 4, 3)
                    .mul(10)
                    .requires_grad_()
                )
                categorical = get_values(
                    categorical_logits,
                    args=args,
                    unconcat_tokens=[torch.arange(8)],
                    total_lengths=[8],
                    response_lengths=[4],
                    truncated=[1],
                    include_logits=True,
                )
                full_logits = all_gather_with_cp(categorical["value_logits"][0], 8, 4)
                torch.testing.assert_close(full_logits.argmax(dim=-1), torch.tensor([0, 1, 2, 0]))
                torch.testing.assert_close(categorical["bootstrap_values"][0], torch.tensor([1.5]), atol=2e-4, rtol=0)
                full_logits.sum().backward()
                assert torch.isfinite(categorical_logits.grad).all()
    finally:
        dist.destroy_process_group()


def test_final_state_bootstrap_with_context_parallelism() -> None:
    run_multiprocess(_check_cp_bootstrap)
