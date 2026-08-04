"""Per-rollout DP/microbatch scheduling (pack first, distribute second), computed on the rollout side."""

from __future__ import annotations

import logging
from typing import Any

from miles.utils.flops_utils import calculate_fwd_flops
from miles.utils.seqlen_balancing import (
    expand_bins_by_splitting,
    first_fit_decreasing_pack,
    get_seqlen_balanced_partitions,
)

logger = logging.getLogger(__name__)

SCHEDULE_CONFIG_KEYS = ("dp_size", "cp_size", "vpp_size", "microbatch_group_size_per_vp_stage")


def has_full_schedule_config(train_parallel_config: dict | None) -> bool:
    """True when the backend advertised every field build_dp_schedule needs."""
    if not train_parallel_config:
        return False
    return all(key in train_parallel_config for key in SCHEDULE_CONFIG_KEYS)


def _calculate_workloads(step_lengths, args):
    return [calculate_fwd_flops([sl], args) for sl in step_lengths]


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute per-rank ``(partitions, micro_batch_indices, num_microbatches, num_rollouts)``;
    ``global_batch_size`` counts rollouts, not training samples."""
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"] or 1
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    # micro-batch size per step must divide evenly across dp and vpp
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # Rollout can include multiple samples (compaction, subagent, fork, etc.)
    # Samples in the same rollout should be trained in the same step, or dropped together.
    rollout_id_to_sample_index: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_sample_index.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_sample_index.keys())

    # Plan the rollout count of each training step: full steps of global_batch_size,
    # plus (under --allow-partial-train-step) one smaller final step for the trailing
    # rollouts that would otherwise be dropped.
    num_full_steps = len(rollout_ids) // global_batch_size
    assert num_full_steps >= 1, (
        f"total rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )
    num_rollouts = [global_batch_size] * num_full_steps
    leftover = len(rollout_ids) - num_full_steps * global_batch_size
    if leftover and getattr(args, "allow_partial_train_step", False) and args.use_dynamic_batch_size:
        leftover_samples = sum(len(rollout_id_to_sample_index[rid]) for rid in rollout_ids[-leftover:])
        if leftover_samples >= dp_size:
            num_rollouts.append(leftover)
        else:
            logger.info(f"partial step skipped: {leftover_samples} samples < dp_size {dp_size}")

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []

    step_start = 0
    for step_i, step_num_rollouts in enumerate(num_rollouts):
        picked_rollouts = rollout_ids[step_start : step_start + step_num_rollouts]
        step_start += step_num_rollouts
        sample_indices = [pos for rid in picked_rollouts for pos in rollout_id_to_sample_index[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        assert len(sample_indices) >= dp_size, (
            f"step of {step_num_rollouts} rollouts has {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        balance_by_flops = getattr(args, "balance_by_flops", False)
        # Shared by FLOPs-balanced packing and FLOPs-balanced distribution below.
        workloads = _calculate_workloads(step_lengths, args) if balance_by_flops else None
        if args.use_dynamic_batch_size:
            # Pack under the token budget: first-fit, or FLOPs-balanced partitions under
            # --balance-by-flops (which does not enforce the token cap per micro-batch).
            if balance_by_flops:
                total_tokens = sum(step_lengths)
                micro_batch_count = max(1, (total_tokens + max_per_bin - 1) // max_per_bin)
                if micro_batch_count >= len(step_lengths):
                    step_micro_batches = [[i] for i in range(len(step_lengths))]
                else:
                    step_micro_batches = get_seqlen_balanced_partitions(workloads, micro_batch_count, equal_size=False)
            else:
                step_micro_batches = first_fit_decreasing_pack(step_lengths, max_per_bin)
            # Grow the micro-batch count to a multiple of align_to by splitting multi-sample micro-batches.
            target = max((len(step_micro_batches) + align_to - 1) // align_to * align_to, align_to)
            if target != len(step_micro_batches):
                expand_bins_by_splitting(step_micro_batches, target, step_lengths)
                assert len(step_micro_batches) == target, (
                    f"dynamic path: could only produce {len(step_micro_batches)} micro-batches after maximal "
                    f"splitting; need {target}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
        else:
            # Fixed-size chunks of micro_batch_size samples.
            assert args.micro_batch_size is not None
            n = len(step_lengths)
            step_micro_batches = [
                list(range(i, min(i + args.micro_batch_size, n))) for i in range(0, n, args.micro_batch_size)
            ]
            if len(step_micro_batches) % align_to != 0:
                raise AssertionError(
                    f"static path: micro-batch count ({len(step_micro_batches)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static micro-batches would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        num_microbatches.append(len(step_micro_batches) // dp_size)

        # Distribute the micro-batches across DP ranks, len(step_micro_batches) / dp_size each: strided
        # round-robin, or Karmarkar-Karp on micro-batch weights (tokens under --balance-data,
        # FLOPs under --balance-by-flops).
        if args.balance_data or balance_by_flops:
            if balance_by_flops:
                weights = [sum(workloads[i] for i in micro_batch) for micro_batch in step_micro_batches]
            else:
                weights = [sum(step_lengths[i] for i in micro_batch) for micro_batch in step_micro_batches]
            rank_micro_batch_ids = get_seqlen_balanced_partitions(weights, dp_size, equal_size=True)
        else:
            rank_micro_batch_ids = [list(range(rank, len(step_micro_batches), dp_size)) for rank in range(dp_size)]

        for rank, micro_batch_ids in enumerate(rank_micro_batch_ids):
            for k in micro_batch_ids:
                micro_batch = step_micro_batches[k]
                local_start = len(partitions[rank])
                partitions[rank].extend(sample_indices[i] for i in micro_batch)
                micro_batch_indices[rank].append(list(range(local_start, local_start + len(micro_batch))))

    return partitions, micro_batch_indices, num_microbatches, num_rollouts
