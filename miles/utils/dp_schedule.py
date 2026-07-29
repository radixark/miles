"""Per-rollout DP/microbatch scheduling, computed on the rollout side.

Pure Python (no ray/sglang imports) so it is unit-testable under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by rollout id (``rollout_indices[i]``, defaulting to
     ``samples[i].index``) and split rollouts into steps of
     ``global_batch_size`` rollouts each. In the common case one rollout
     emits one training sample so this is the same as a contiguous chunk;
     under compact / subagent one rollout may emit multiple training
     samples, in which case all of those samples stay in the same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     mbs token sums.

Invariants (asserted by tests/fast/utils/test_dp_schedule.py):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds <= ``max_tokens_per_gpu * cp_size``
    tokens, except an oversized sample, which lands alone in its own mbs;
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing rollouts (every kept sample placed exactly once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)``.
"""

from __future__ import annotations

import logging
from typing import Any

from miles.utils.seqlen_balancing import (
    expand_bins_by_splitting,
    first_fit_decreasing_pack,
    get_seqlen_balanced_partitions,
)

logger = logging.getLogger(__name__)

SCHEDULE_CONFIG_KEYS = ("dp_size", "cp_size", "vpp_size", "microbatch_group_size_per_vp_stage")


def has_full_schedule_config(train_parallel_config: dict | None) -> bool:
    """True when the backend advertised every field build_dp_schedule needs
    (fsdp/torchtitan report only ``dp_size``; indep_dp reports ``{}``)."""
    if not train_parallel_config:
        return False
    return all(key in train_parallel_config for key in SCHEDULE_CONFIG_KEYS)


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        # FFD packs at least as tightly as arrival-order first-fit; fewer
        # bins = fewer micro-batches per training step.
        return first_fit_decreasing_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample. Samples sharing the same
            id are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``:
          - ``partitions[r]`` — global sample indices of rank r, in mbs order;
          - ``micro_batch_indices[r][k]`` — local indices into ``partitions[r]``
            for the k-th mbs (flat across steps);
          - ``num_microbatches[s]`` — mbs count for step s, same on every rank;
          - ``global_batch_sizes[s]`` — rollout count for step s (equal to
            ``global_batch_size`` except for an opt-in smaller final partial step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"] or 1
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by rollout id (preserve first-occurrence order). All
    # samples from one rollout stay in a single step so the per-rollout loss
    # reducer is well-defined.
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples.keys())

    num_steps = len(rollout_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )

    # Per-step rollout counts. With --allow-partial-train-step, trailing
    # rollouts train as one smaller final step (dynamic batch only — the
    # static path can't satisfy its fixed-size alignment on an arbitrary
    # remainder) instead of being dropped.
    step_rollout_counts = [global_batch_size] * num_steps
    leftover = len(rollout_ids) - num_steps * global_batch_size
    if leftover and getattr(args, "allow_partial_train_step", False) and args.use_dynamic_batch_size:
        leftover_samples = sum(len(rollout_id_to_samples[rid]) for rid in rollout_ids[-leftover:])
        if leftover_samples >= dp_size:
            step_rollout_counts.append(leftover)
        else:
            logger.info(f"partial step skipped: {leftover_samples} samples < dp_size {dp_size}")

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    rollout_cursor = 0
    for step_i, step_rollout_count in enumerate(step_rollout_counts):
        step_rollouts = rollout_ids[rollout_cursor : rollout_cursor + step_rollout_count]
        rollout_cursor += step_rollout_count
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(step_rollout_count)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on mbs token sums when balance_data is on,
        # otherwise a strided round-robin. Both produce ``num_mbs_per_rank`` mbs per
        # rank (equal_size=True is what KK needs for PP to stay synced).
        if args.balance_data:
            mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_token_sums, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
