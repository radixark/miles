"""With per-sample loss, a rollout split into several samples must count once, like the unsplit rollout.

Three rollouts are split into 1, 3 and 4 samples (eight samples, a folded episode's segments). Each sample's
prompt is its rollout's tokens before the segment and its response is the segment, so every loss token sees
the same previous token as in the unsplit rollout. The fake model's logits depend only on the previous token
(an embedding table plus the trained bias), so a token's log-prob does not depend on how the rollout is cut.

Four gloo ranks run dp=2 x cp=2 through the real pieces: DataIterator with the actor's TRAIN_KEYS,
loss_function (sft loss) through the actor's _step_loss, RLLossAdapter with its CP gather and the trainer's
1 / (dp * cp) scale, and a SUM all-reduce of the gradient over every rank (torchtitan's FSDP with
gradient_divide_factor 1.0). The samples go to the DP ranks round-robin, as the training-side DP split does,
so one rollout's samples sit on both ranks. The reference is the gradient of the mean over the three
unsplit rollouts of each rollout's token-mean loss."""

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
from miles.backends.training_utils.torch_native.actor import TRAIN_KEYS, _step_loss
from miles.ray.rollout.train_data_conversion import _compute_rollout_mask_sums

_DP, _CP = 2, 2
_VOCAB = 11
_SEQ_LEN = 64
_MICRO_BATCH_SIZE = 2
_NUM_MICROBATCHES = 2
_SEGMENTS_PER_ROLLOUT = (1, 3, 4)
_NUM_ROLLOUTS = len(_SEGMENTS_PER_ROLLOUT)
_NUM_SAMPLES = sum(_SEGMENTS_PER_ROLLOUT)

# four spawned processes each import torchtitan, as in test_torchtitan_loss_scale.py
register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["torchtitan"])


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _rollouts() -> list[dict]:
    """Per rollout: a prompt, then one response segment and one loss mask per sample."""
    g = torch.Generator().manual_seed(7)
    rollouts = []
    for num_segments in _SEGMENTS_PER_ROLLOUT:
        prompt = torch.randint(0, _VOCAB, (2 + num_segments % 2,), generator=g)
        segments, masks = [], []
        for j in range(num_segments):
            length = 2 + (j + num_segments) % 3
            segments.append(torch.randint(0, _VOCAB, (length,), generator=g))
            masks.append((torch.rand(length, generator=g) > 0.25).int())
        rollouts.append({"prompt": prompt, "segments": segments, "masks": masks})
    return rollouts


def _split_samples() -> list[dict]:
    """The eight samples in rollout order, with their rollout index."""
    samples = []
    for rollout_id, rollout in enumerate(_rollouts()):
        context = rollout["prompt"]
        for segment, mask in zip(rollout["segments"], rollout["masks"], strict=True):
            samples.append({"tokens": torch.cat([context, segment]), "loss_mask": mask, "rollout_id": rollout_id})
            context = torch.cat([context, segment])
    return samples


def _rank_data(dp_rank: int, *, per_rollout: bool) -> dict:
    """One DP rank's shard. ``per_rollout`` adds what the DP split hands a multi-sample batch:
    each sample's whole-rollout loss-mask sum and the step's rollout count over all ranks."""
    samples = _split_samples()
    rollout_mask_sums = _compute_rollout_mask_sums(
        [s["rollout_id"] for s in samples], [s["loss_mask"].tolist() for s in samples]
    )
    picked = range(dp_rank, _NUM_SAMPLES, _DP)
    data = {
        "tokens": [samples[i]["tokens"] for i in picked],
        "loss_masks": [samples[i]["loss_mask"] for i in picked],
        "total_lengths": [len(samples[i]["tokens"]) for i in picked],
        "response_lengths": [len(samples[i]["loss_mask"]) for i in picked],
    }
    if per_rollout:
        data["rollout_mask_sums"] = torch.tensor([rollout_mask_sums[i] for i in picked], dtype=torch.float32)
        data["num_rollouts"] = [_NUM_ROLLOUTS]
    return data


def _logits(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[1, len(tokens), V]: row t depends only on token t (the previous token of the row's target)."""
    table = torch.randn((_VOCAB, _VOCAB), generator=torch.Generator().manual_seed(3)) * 2
    return (table[tokens] + weight).unsqueeze(0)


def _reference_grad() -> torch.Tensor:
    """The gradient of the mean over the unsplit rollouts of each rollout's token-mean loss."""
    weight = torch.zeros(_VOCAB, requires_grad=True)
    rollout_losses = []
    for rollout in _rollouts():
        response = torch.cat(rollout["segments"])
        mask = torch.cat(rollout["masks"])
        tokens = torch.cat([rollout["prompt"], response])
        rows = _logits(tokens, weight)[0, len(rollout["prompt"]) - 1 : len(tokens) - 1]
        log_probs = torch.log_softmax(rows, dim=-1).gather(-1, response.unsqueeze(-1)).squeeze(-1)
        rollout_losses.append(-(log_probs * mask).sum() / max(int(mask.sum()), 1))
    (sum(rollout_losses) / len(rollout_losses)).backward()
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
        reference = _reference_grad()
        torch.testing.assert_close(_step_gradient(mesh, dp, per_rollout=True), reference, rtol=1e-5, atol=1e-6)
        # Without the rollout count and sums every sample is its own mean over the sample count: the
        # 4-sample rollout weighs 4 / 8 of the step instead of 1 / 3.
        assert not torch.allclose(_step_gradient(mesh, dp, per_rollout=False), reference, rtol=1e-3, atol=1e-4)
    finally:
        dist.destroy_process_group()


def _step_gradient(mesh, dp: GroupInfo, *, per_rollout: bool) -> torch.Tensor:
    from torchtitan.distributed.context_parallel import cp_shard

    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    # The training-side split trains one step per round on a multi-sample batch: the global batch
    # size is the sample count.
    args = make_args(
        loss_type="sft_loss",
        true_on_policy_mode=False,
        calculate_per_token_loss=False,
        global_batch_size=_NUM_SAMPLES,
    )
    data = _rank_data(dp.rank, per_rollout=per_rollout)
    num_rollouts = data.pop("num_rollouts", [None])
    data_iterator = DataIterator(data, micro_batch_size=_MICRO_BATCH_SIZE)
    batches, logits_tokens = [], []
    for _ in range(_NUM_MICROBATCHES):
        batch = data_iterator.get_next([key for key in TRAIN_KEYS if key in data])
        batch["unconcat_tokens"] = batch["tokens"]
        batch["tokens"] = torch.cat(batch["tokens"]).unsqueeze(0)
        assert batch["tokens"].numel() <= _SEQ_LEN
        logits_tokens.append(torch.nn.functional.pad(batch["tokens"][0], (0, _SEQ_LEN - batch["tokens"].numel())))
        batches.append(batch)

    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_gradient_scale(1.0 / (_DP * _CP))
    adapter.set_context_parallel(mesh["cp"], "headtail")
    adapter.arm(
        batches, partial(_step_loss, args, _NUM_MICROBATCHES, None, num_rollouts=num_rollouts[0]), is_training=True
    )
    (local_positions,), _ = cp_shard(mesh["cp"], (torch.arange(_SEQ_LEN).unsqueeze(0),), None, "headtail")
    local_positions = local_positions.flatten()
    weight = torch.zeros(_VOCAB, requires_grad=True)
    for microbatch in range(_NUM_MICROBATCHES):
        local_logits = _logits(logits_tokens[microbatch], weight)[:, local_positions]
        label = torch.full((1, local_positions.numel()), microbatch, dtype=torch.long)
        loss, _ = adapter(local_logits, label)
        loss.backward()
    dist.all_reduce(weight.grad)
    return weight.grad


def test_a_split_rollout_counts_once_in_the_per_sample_loss():
    pytest.importorskip("torchtitan")
    run_multiprocess(_worker, world_size=_DP * _CP)
