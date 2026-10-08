"""CP=2 consistency tests for the allgather-CP loss path.

Each case runs the same data through cp=1 (plain path) and cp=2
(allgather-CP layout: each rank holds one contiguous half of the global
logits) and asserts that per-token log-probs match exactly and the summed
loss matches the cp=1 loss. Chained with the cp=1 snapshot tests, this
anchors the cp=2 path without storing multi-rank snapshots.
"""

from datetime import timedelta
from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args, make_batch, make_inputs

VOCAB_SIZE = 64
MODEL_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

# (name, prompt_lens, response_lens) — totals must divide by 2*cp for the zigzag layout.
CASES = [
    # response 0 spans both contiguous halves, response 1 sits in rank 1's half
    ("split_responses", [16, 2], [8, 6]),
    # rank 0's contiguous half holds no response logits (pre-redistribute empty)
    ("contiguous_empty_rank", [40], [24]),
    # rank 0's zigzag share holds no response tokens (triggers the sft loss placeholder)
    ("zigzag_empty_rank", [12], [16]),
]


def _set_parallel_state(rank: int, world_size: int, tp_group) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD if world_size > 1 else None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=cp,
            cp=cp,
            tp=GroupInfo(rank=0, size=1, group=tp_group),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _run_case(rank: int, world_size: int, port: int, *, prompt_lens: list[int], response_lens: list[int]) -> None:
    init_gloo(rank, world_size, port=port)
    # The fused CE needs a real group per rank; new_group is collective, so build all of them.
    tp_group = [dist.new_group([r]) for r in range(world_size)][rank]

    args = make_args(
        loss_type="sft_loss",
        true_on_policy_mode=False,
        allgather_cp=True,
        log_probs_chunk_size=4,
        rollout_temperature=0.7,
    )
    inputs = make_inputs(42, len(prompt_lens), prompt_lens, response_lens, VOCAB_SIZE, args)
    total_lens = inputs["total_lens"]
    logits_full = inputs["policy_logits"]  # [1, T, V] fp32
    t_local = logits_full.size(1) // world_size

    for dtype in MODEL_DTYPES:
        logits = logits_full.to(dtype)

        args.allgather_cp = False
        _set_parallel_state(rank=0, world_size=1, tp_group=tp_group)
        base_res = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=inputs["unconcat_tokens"],
            total_lengths=total_lens,
            response_lengths=response_lens,
        )
        base_loss, _, _ = loss_function(args, make_batch(inputs, "sft_loss"), 1, logits)

        args.allgather_cp = True
        _set_parallel_state(rank=rank, world_size=world_size, tp_group=tp_group)
        local_logits = logits[:, rank * t_local : (rank + 1) * t_local]
        cp_res = get_log_probs_and_entropy(
            local_logits,
            args=args,
            unconcat_tokens=inputs["unconcat_tokens"],
            total_lengths=total_lens,
            response_lengths=response_lens,
        )
        cp_loss, _, _ = loss_function(args, make_batch(inputs, "sft_loss"), 1, local_logits)

        # Per-token log-probs: CE is row-wise, so cp must be exactly transparent.
        for i, (total_len, response_len) in enumerate(zip(total_lens, response_lens, strict=True)):
            full = all_gather_with_cp(cp_res["log_probs"][i], total_len, response_len)
            torch.testing.assert_close(full, base_res["log_probs"][i], rtol=0, atol=0)

        # The summed loss reassociates one addition per split sample.
        cp_loss_sum = cp_loss.detach().clone()
        dist.all_reduce(cp_loss_sum)
        torch.testing.assert_close(cp_loss_sum, base_loss.detach(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("name", "prompt_lens", "response_lens"), CASES, ids=[c[0] for c in CASES])
def test_allgather_cp2_matches_cp1(name: str, prompt_lens: list[int], response_lens: list[int]) -> None:
    run_multiprocess(partial(_run_case, prompt_lens=prompt_lens, response_lens=response_lens))


def _run_backward_case(
    rank: int,
    world_size: int,
    port: int,
    *,
    response_length: int,
    true_on_policy: bool,
    entropy_coef: float,
) -> None:
    # Bound the failure when an empty rank skips a differentiable CP collective.
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
        total_length = 128
        args = make_args(
            advantage_estimator="gspo",
            true_on_policy_mode=true_on_policy,
            use_rollout_logprobs=True,
            entropy_coef=entropy_coef,
            observe_training_entropy=True,
        )
        inputs = make_inputs(42, 1, [total_length - response_length], [response_length], VOCAB_SIZE, args)
        inputs["advantages"] = [torch.ones(response_length)]
        inputs["loss_masks"][0][::3] = 0

        _set_parallel_state(rank=0, world_size=1, tp_group=tp_group)
        base_logits = inputs["policy_logits"].clone().requires_grad_()
        base_res = get_log_probs_and_entropy(
            base_logits.detach(),
            args=args,
            unconcat_tokens=inputs["unconcat_tokens"],
            total_lengths=[total_length],
            response_lengths=[response_length],
        )
        inputs["rollout_log_probs"] = base_res["log_probs"]
        base_loss, _, _ = loss_function(args, make_batch(inputs, "policy_loss"), 1, base_logits)
        base_loss.backward()
        assert base_logits.grad.norm() > 0

        args.allgather_cp = True
        _set_parallel_state(rank=rank, world_size=world_size, tp_group=tp_group)
        cp_batch = make_batch(inputs, "policy_loss")
        for key in ("advantages", "rollout_log_probs", "log_probs", "ref_log_probs"):
            cp_batch[key] = [slice_log_prob_with_cp(value, total_length, response_length) for value in cp_batch[key]]
        local_slice = slice(rank * total_length // world_size, (rank + 1) * total_length // world_size)
        local_logits = inputs["policy_logits"][:, local_slice].clone().requires_grad_()
        # Model CP collectives must run after the loss collectives, even on prompt-only ranks.
        model_logits = torch.cat(dist.nn.all_gather(local_logits), dim=1)[:, local_slice]
        cp_loss, _, _ = loss_function(args, cp_batch, 1, model_logits)
        cp_loss.backward()

        cp_loss_sum = cp_loss.detach().clone()
        dist.all_reduce(cp_loss_sum)
        torch.testing.assert_close(cp_loss_sum, base_loss.detach())
        torch.testing.assert_close(local_logits.grad, base_logits.grad[:, local_slice])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("response_length", [8, 80], ids=["empty_rank", "split_response"])
@pytest.mark.parametrize("true_on_policy", [False, True], ids=["megatron_logprobs", "true_on_policy"])
@pytest.mark.parametrize("entropy_coef", [0.0, 0.01], ids=["observed_entropy", "entropy_loss"])
def test_allgather_cp2_gspo_backward_matches_cp1(
    response_length: int, true_on_policy: bool, entropy_coef: float
) -> None:
    run_multiprocess(
        partial(
            _run_backward_case,
            response_length=response_length,
            true_on_policy=true_on_policy,
            entropy_coef=entropy_coef,
        )
    )
