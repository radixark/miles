"""Per-rollout DP/microbatch scheduling, computed on the rollout side."""

from __future__ import annotations

from typing import Any

from miles.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

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
) -> tuple[list[list[int]], list[list[list[int]]], list[int]]:
    """Compute per-rank ``(partitions, micro_batch_indices, num_microbatches)``."""
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"] or 1
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    assert len(total_lengths) % global_batch_size == 0, (
        f"num_samples ({len(total_lengths)}) is not a multiple of global_batch_size "
        f"({global_batch_size}); trim upstream before scheduling."
    )
    num_steps = len(total_lengths) // global_batch_size

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []

    for step_i in range(num_steps):
        step_start = step_i * global_batch_size
        step_lengths = total_lengths[step_start : step_start + global_batch_size]

        if args.balance_data:
            rank_parts = get_seqlen_balanced_partitions(step_lengths, dp_size, equal_size=True)
        else:
            rank_parts = [list(range(r, global_batch_size, dp_size)) for r in range(dp_size)]

        if not args.use_dynamic_batch_size:
            mbs = args.micro_batch_size
            n = len(rank_parts[0])  # gbs / dp, same for every rank
            assert n % mbs == 0, (
                f"per-rank batch ({n} = global_batch_size {global_batch_size} / dp_size {dp_size}) "
                f"is not a multiple of micro_batch_size ({mbs})"
            )
            rank_mbs = [[list(range(i, i + mbs)) for i in range(0, n, mbs)] for _ in range(dp_size)]
            num_mbs_per_rank = n // mbs
        else:
            rank_lens = [[step_lengths[i] for i in rank_parts[r]] for r in range(dp_size)]
            rank_mbs = [first_fit_pack(rank_lens[r], max_per_bin) for r in range(dp_size)]
            num_mbs_per_rank = max(len(b) for b in rank_mbs)
            if vpp_size > 1:
                num_mbs_per_rank = max(num_mbs_per_rank // mb_group * mb_group, 1)
            for r in range(dp_size):
                expand_bins_by_splitting(rank_mbs[r], num_mbs_per_rank, rank_lens[r])

        num_microbatches.append(num_mbs_per_rank)

        for r in range(dp_size):
            for mbs_local in rank_mbs[r]:
                local_start = len(partitions[r])
                partitions[r].extend(step_start + rank_parts[r][i] for i in mbs_local)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_local))))

    return partitions, micro_batch_indices, num_microbatches
