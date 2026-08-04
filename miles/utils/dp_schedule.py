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


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute per-rank ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``;
    ``global_batch_size`` counts rollouts, not training samples."""
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"] or 1
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    # mbs count per step must divide evenly across ranks and, under VPP, mb groups.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    rollout_to_samples = _group_samples_by_rollout(rollout_indices)
    step_rollout_counts = _plan_step_rollout_counts(args, rollout_to_samples, global_batch_size, dp_size)

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    rollout_ids = list(rollout_to_samples.keys())
    rollout_cursor = 0
    for step_rollout_count in step_rollout_counts:
        step_rollouts = rollout_ids[rollout_cursor : rollout_cursor + step_rollout_count]
        rollout_cursor += step_rollout_count
        sample_indices = [pos for rid in step_rollouts for pos in rollout_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        assert len(sample_indices) >= dp_size, (
            f"step of {step_rollout_count} rollouts has {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        step_mbs = _pack_step(args, step_lengths, cp_size)
        step_mbs = _align_mbs_count(args, step_mbs, step_lengths, align_to)
        rank_mbs = _distribute_mbs(args, step_mbs, step_lengths, dp_size)

        num_microbatches.append(len(step_mbs) // dp_size)
        global_batch_sizes.append(step_rollout_count)
        for rank, mbs_list in enumerate(rank_mbs):
            for mbs in mbs_list:
                local_start = len(partitions[rank])
                partitions[rank].extend(sample_indices[i] for i in mbs)
                micro_batch_indices[rank].append(list(range(local_start, local_start + len(mbs))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes


def _group_samples_by_rollout(rollout_indices: list[int]) -> dict[int, list[int]]:
    """Sample positions per rollout id, preserving first-occurrence order, so a
    rollout's samples always land in the same training step."""
    rollout_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_to_samples.setdefault(rid, []).append(sample_pos)
    return rollout_to_samples


def _plan_step_rollout_counts(
    args: Any, rollout_to_samples: dict[int, list[int]], global_batch_size: int, dp_size: int
) -> list[int]:
    """Rollout count per training step: full steps of ``global_batch_size``, plus —
    under ``--allow-partial-train-step`` — one smaller final step for the trailing
    rollouts that would otherwise be dropped (dynamic batch only)."""
    num_rollouts = len(rollout_to_samples)
    num_full_steps = num_rollouts // global_batch_size
    assert num_full_steps >= 1, (
        f"num_rollouts ({num_rollouts}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )

    counts = [global_batch_size] * num_full_steps
    leftover = num_rollouts - num_full_steps * global_batch_size
    if leftover and getattr(args, "allow_partial_train_step", False) and args.use_dynamic_batch_size:
        leftover_rollouts = list(rollout_to_samples.keys())[-leftover:]
        leftover_samples = sum(len(rollout_to_samples[rid]) for rid in leftover_rollouts)
        if leftover_samples >= dp_size:
            counts.append(leftover)
        else:
            logger.info(f"partial step skipped: {leftover_samples} samples < dp_size {dp_size}")
    return counts


def _pack_step(args: Any, step_lengths: list[int], cp_size: int) -> list[list[int]]:
    """Group one step's samples into micro-batches; ``mbs[k]`` holds local indices
    into ``step_lengths``."""
    if not args.use_dynamic_batch_size:
        assert args.micro_batch_size is not None
        n = len(step_lengths)
        return [list(range(i, min(i + args.micro_batch_size, n))) for i in range(0, n, args.micro_batch_size)]

    assert args.max_tokens_per_gpu is not None
    max_per_bin = args.max_tokens_per_gpu * cp_size
    if getattr(args, "balance_by_flops", False):
        num_mbs = max(1, (sum(step_lengths) + max_per_bin - 1) // max_per_bin)
        if num_mbs >= len(step_lengths):
            return [[i] for i in range(len(step_lengths))]
        # NOTE: FLOPs balancing does not enforce the token cap per mbs.
        return get_seqlen_balanced_partitions(_workloads(args, step_lengths), num_mbs, equal_size=False)
    return first_fit_decreasing_pack(step_lengths, max_per_bin)


def _align_mbs_count(args: Any, step_mbs: list[list[int]], step_lengths: list[int], align_to: int) -> list[list[int]]:
    """Grow the mbs count to a multiple of ``align_to`` (dp_size x VPP mb group)."""
    target = max((len(step_mbs) + align_to - 1) // align_to * align_to, align_to)
    if target == len(step_mbs):
        return step_mbs
    if not args.use_dynamic_batch_size:
        raise AssertionError(
            f"static path: num_mbs ({len(step_mbs)}) is not a multiple of {align_to}; "
            f"adjust the config so step_size % (dp_size * micro_batch_size * mb_group) == 0."
        )
    expand_bins_by_splitting(step_mbs, target, step_lengths)
    assert len(step_mbs) == target, f"could only produce {len(step_mbs)} mbs after maximal splitting; need {target}."
    return step_mbs


def _distribute_mbs(
    args: Any, step_mbs: list[list[int]], step_lengths: list[int], dp_size: int
) -> list[list[list[int]]]:
    """Assign micro-batches to DP ranks, ``len(step_mbs) / dp_size`` each: strided
    round-robin, or Karmarkar-Karp on mbs weights (tokens under ``--balance-data``,
    FLOPs under ``--balance-by-flops``)."""
    if args.balance_data or getattr(args, "balance_by_flops", False):
        if getattr(args, "balance_by_flops", False):
            workloads = _workloads(args, step_lengths)
            weights = [sum(workloads[i] for i in mbs) for mbs in step_mbs]
        else:
            weights = [sum(step_lengths[i] for i in mbs) for mbs in step_mbs]
        rank_mbs_idx = get_seqlen_balanced_partitions(weights, dp_size, equal_size=True)
    else:
        rank_mbs_idx = [list(range(rank, len(step_mbs), dp_size)) for rank in range(dp_size)]
    return [[step_mbs[i] for i in mbs_idx] for mbs_idx in rank_mbs_idx]


def _workloads(args: Any, step_lengths: list[int]) -> list[int]:
    return [calculate_fwd_flops([length], args) for length in step_lengths]
