"""The torchtitan loss under CP computes log-probs on each rank's shard and gathers only [T] vectors.
It must match log-probs and gradients computed from the full-sequence logits."""

import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.loss.hub import math_utils
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

_VOCAB = 37
_LENGTHS = [13, 11]
_RESPONSES = [6, 5]
_SEQ_LEN = 32


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _set_single_rank_tp(world_size: int) -> None:
    groups = [dist.new_group([r]) for r in range(world_size)]
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=GroupInfo(rank=0, size=1, group=groups[dist.get_rank()]),
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            is_pp_last_stage=True,
        )
    )


def _loss(logits, batch, args):
    out = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=_LENGTHS,
        response_lengths=_RESPONSES,
        with_entropy=True,
    )
    log_probs, entropy = torch.cat(out["log_probs"]), torch.cat(out["entropy"])
    weights = torch.linspace(0.5, 1.5, log_probs.numel())
    loss = (weights * log_probs).sum() + 0.1 * entropy.sum()
    return loss, {"log_probs": log_probs.detach(), "entropy": entropy.detach()}


def _worker(rank: int, world_size: int, port: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    init_gloo(rank, world_size, port=port)
    try:
        math_utils.compute_log_probs = _naive_compute_log_probs
        _set_single_rank_tp(world_size)
        mesh = init_device_mesh("cpu", (world_size,))
        args = make_args(true_on_policy_mode=False, rollout_temperature=0.8, log_probs_chunk_size=3)
        for dtype in (torch.float32, torch.bfloat16):
            _check_against_the_full_logits(mesh, args, dtype, world_size)
    finally:
        dist.destroy_process_group()


def _check_against_the_full_logits(mesh, args, dtype: torch.dtype, world_size: int) -> None:
    from torchtitan.distributed.context_parallel import cp_shard

    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    g = torch.Generator().manual_seed(0)
    tokens = [torch.randint(0, _VOCAB, (n,), generator=g) for n in _LENGTHS]
    full = (torch.randn((1, _SEQ_LEN, _VOCAB), generator=g) * 3).to(dtype)
    batch = {"tokens": torch.cat(tokens).unsqueeze(0), "unconcat_tokens": tokens}

    reference_logits = full.clone().requires_grad_(True)
    reference_loss, reference = _loss(reference_logits, batch, args)
    reference_loss.backward()

    (local_positions,), _ = cp_shard(mesh, (torch.arange(_SEQ_LEN).unsqueeze(0),), None, "headtail")
    local_positions = local_positions.flatten()
    local_logits = full[:, local_positions].clone().requires_grad_(True)
    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_context_parallel(mesh, "headtail")
    adapter.arm([batch], lambda pred, b: _loss(pred, b, args), is_training=True)
    loss, _ = adapter(local_logits, torch.zeros(1, local_positions.numel(), dtype=torch.long))
    loss.backward()
    (result,) = adapter.collect()

    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(result["log_probs"], reference["log_probs"], rtol=0, atol=0)
    torch.testing.assert_close(result["entropy"], reference["entropy"], rtol=0, atol=0)
    # every rank backprops the same full-sequence loss and the gather's backward sums the ranks,
    # the factor the trainer's 1 / cp gradient scale removes; the full-logit gather did the same
    expected_grad = world_size * reference_logits.grad[:, local_positions]
    torch.testing.assert_close(local_logits.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_cp_local_log_probs_match_the_full_logit_gather(world_size):
    pytest.importorskip("torchtitan")
    run_multiprocess(_worker, world_size=world_size)


def _headtail_indices(seq_len: int, cp: int, rank: int) -> torch.Tensor:
    chunks = torch.arange(seq_len).chunk(2 * cp)
    return torch.cat([chunks[rank], chunks[2 * cp - 1 - rank]])


def _wrapper_worker(rank: int, world_size: int, port: int) -> None:
    """The same check without torchtitan: the shard and its gather built by hand."""
    from torch.distributed._functional_collectives import all_gather_single_autograd

    from miles.backends.training_utils.loss.hub.context_parallel_logits import ContextParallelLogits

    init_gloo(rank, world_size, port=port)
    try:
        math_utils.compute_log_probs = _naive_compute_log_probs
        _set_single_rank_tp(world_size)
        args = make_args(true_on_policy_mode=False, rollout_temperature=1.0, log_probs_chunk_size=-1)
        g = torch.Generator().manual_seed(1)
        tokens = [torch.randint(0, _VOCAB, (n,), generator=g) for n in _LENGTHS]
        full = torch.randn((1, _SEQ_LEN, _VOCAB), generator=g)
        batch = {"unconcat_tokens": tokens}
        reference_logits = full.clone().requires_grad_(True)
        reference_loss, reference = _loss(reference_logits, batch, args)
        reference_loss.backward()

        local_positions = _headtail_indices(_SEQ_LEN, world_size, rank)
        restore = torch.cat([_headtail_indices(_SEQ_LEN, world_size, r) for r in range(world_size)]).argsort()
        targets = torch.zeros(_SEQ_LEN, dtype=torch.long)
        targets[: sum(_LENGTHS) - 1] = torch.cat(tokens)[1:]
        local_logits = full[:, local_positions].clone().requires_grad_(True)
        logits = ContextParallelLogits(
            local=local_logits,
            targets=targets[local_positions],
            gather_rows=lambda rows: all_gather_single_autograd(rows, 0, dist.group.WORLD).index_select(0, restore),
        )
        loss, result = _loss(logits, batch, args)
        loss.backward()

        torch.testing.assert_close(result["log_probs"], reference["log_probs"], rtol=0, atol=0)
        torch.testing.assert_close(result["entropy"], reference["entropy"], rtol=0, atol=0)
        torch.testing.assert_close(local_logits.grad, world_size * reference_logits.grad[:, local_positions])
    finally:
        dist.destroy_process_group()


def test_cp_local_log_probs_match_without_torchtitan():
    run_multiprocess(_wrapper_worker, world_size=2)
