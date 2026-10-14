"""The torchtitan gradient must equal the gradient of the loss averaged over the whole optimizer step.

Four gloo ranks run dp=2 x cp=2 through the real pieces: DataIterator, the per-step global token count,
loss_function (sft loss), RLLossAdapter with its CP gather and the trainer's 1 / (dp * cp) scale, and a
SUM all-reduce of the gradient over every rank (torchtitan's FSDP with gradient_divide_factor 1.0).
The reference computes the same loss on one process from the full logits of every rank's data."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.loss.hub import math_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.torch_native.actor import _global_loss_tokens, _step_loss

_DP, _CP = 2, 2
_VOCAB = 11
_SEQ_LEN = 32
_MICRO_BATCH_SIZE = 2
_NUM_MICROBATCHES = 2
_KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths"]

# four spawned processes each import torchtitan: 28.6 s on an M-series Mac, too close to the 30 s fast budget
register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["torchtitan"])


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _rank_data(dp_rank: int) -> dict:
    """Four samples with different response lengths and masks per DP rank; one mask is all zeros."""
    g = torch.Generator().manual_seed(100 + dp_rank)
    data = {key: [] for key in _KEYS}
    for i in range(_MICRO_BATCH_SIZE * _NUM_MICROBATCHES):
        prompt, response = 2 + i % 3, 3 + (i + dp_rank) % 4
        data["tokens"].append(torch.randint(0, _VOCAB, (prompt + response,), generator=g))
        mask = (torch.rand(response, generator=g) > 0.3).int()
        data["loss_masks"].append(torch.zeros_like(mask) if (dp_rank, i) == (1, 0) else mask)
        data["total_lengths"].append(prompt + response)
        data["response_lengths"].append(response)
    return data


def _base_logits(dp_rank: int, microbatch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + 10 * dp_rank + microbatch)
    return torch.randn((1, _SEQ_LEN, _VOCAB), generator=g) * 2


def _reference_grad(per_token: bool) -> torch.Tensor:
    """The gradient of the step's loss on one process: the token mean, or the mean of sample means."""
    weight = torch.zeros(_VOCAB, requires_grad=True)
    sums, counts = [], []
    for dp_rank in range(_DP):
        data = _rank_data(dp_rank)
        for microbatch in range(_NUM_MICROBATCHES):
            logits = _base_logits(dp_rank, microbatch) + weight
            offset = 0
            for i in range(microbatch * _MICRO_BATCH_SIZE, (microbatch + 1) * _MICRO_BATCH_SIZE):
                tokens, mask = data["tokens"][i], data["loss_masks"][i]
                total, response = data["total_lengths"][i], data["response_lengths"][i]
                rows = logits[0, offset + total - response - 1 : offset + total - 1]
                log_probs = torch.log_softmax(rows, dim=-1).gather(-1, tokens[-response:].unsqueeze(-1)).squeeze(-1)
                sums.append(-(log_probs * mask).sum())
                counts.append(max(int(mask.sum()), 1))
                offset += total
    loss = sum(sums) / sum(counts) if per_token else sum(s / c for s, c in zip(sums, counts, strict=True)) / len(sums)
    loss.backward()
    return weight.grad


def _worker(rank: int, world_size: int, port: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    init_gloo(rank, world_size, port=port)
    try:
        math_utils.compute_log_probs = _naive_compute_log_probs
        mesh = init_device_mesh("cpu", (_DP, _CP), mesh_dim_names=("dp", "cp"))
        tp_groups = [dist.new_group([r]) for r in range(world_size)]
        trivial = GroupInfo(rank=0, size=1, group=None)
        dp = GroupInfo(rank=mesh.get_local_rank("dp"), size=_DP, group=mesh.get_group("dp"))
        set_parallel_state(
            ParallelState(
                intra_dp=dp,
                intra_dp_cp=dp,
                cp=trivial,
                tp=GroupInfo(rank=0, size=1, group=tp_groups[rank]),
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
                is_pp_last_stage=True,
            )
        )
        for per_token in (True, False):
            grad = _step_gradient(mesh, dp, per_token=per_token)
            torch.testing.assert_close(grad, _reference_grad(per_token), rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


def _step_gradient(mesh, dp: GroupInfo, *, per_token: bool) -> torch.Tensor:
    from torchtitan.distributed.context_parallel import cp_shard

    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    args = make_args(
        loss_type="sft_loss",
        true_on_policy_mode=False,
        calculate_per_token_loss=per_token,
        global_batch_size=_DP * _MICRO_BATCH_SIZE * _NUM_MICROBATCHES,
    )
    data_iterator = DataIterator(_rank_data(dp.rank), micro_batch_size=_MICRO_BATCH_SIZE)
    global_num_tokens = _global_loss_tokens(data_iterator, _NUM_MICROBATCHES, group=dp.group) if per_token else None
    assert data_iterator.offset == 0
    batches = []
    for _ in range(_NUM_MICROBATCHES):
        batch = data_iterator.get_next(_KEYS)
        batch["unconcat_tokens"] = batch["tokens"]
        batch["tokens"] = torch.cat(batch["tokens"]).unsqueeze(0)
        batches.append(batch)

    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_gradient_scale(1.0 / (_DP * _CP))
    adapter.set_context_parallel(mesh["cp"], "headtail")
    adapter.arm(batches, partial(_step_loss, args, _NUM_MICROBATCHES, global_num_tokens), is_training=True)
    (local_positions,), _ = cp_shard(mesh["cp"], (torch.arange(_SEQ_LEN).unsqueeze(0),), None, "headtail")
    local_positions = local_positions.flatten()
    weight = torch.zeros(_VOCAB, requires_grad=True)
    for microbatch in range(_NUM_MICROBATCHES):
        local_logits = (_base_logits(dp.rank, microbatch) + weight)[:, local_positions]
        label = torch.full((1, local_positions.numel()), microbatch, dtype=torch.long)
        loss, _ = adapter(local_logits, label)
        loss.backward()
    dist.all_reduce(weight.grad)
    return weight.grad


def test_the_step_gradient_is_the_gradient_of_the_global_mean_loss():
    """Per-token: the mean over every loss token of the step. Per-sample: the mean of sample means."""
    pytest.importorskip("torchtitan")
    run_multiprocess(_worker, world_size=_DP * _CP)
