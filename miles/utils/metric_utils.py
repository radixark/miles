import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    group_ids: list | None = None,
):
    """Compute pass@k from a flat list of rewards.

    Groups are formed one of two ways:

    * ``group_ids`` given: bucket ``flat_rewards`` by the per-sample group key.
      This is the ragged-safe path -- groups may have any (variable) number of
      samples, e.g. over-sampling, dynamic-sampling filters, aborted/partial
      groups, or list-expanded multi-turn/tool eval samples.
    * ``group_ids`` is ``None``: chunk ``flat_rewards`` into contiguous
      ``group_size`` blocks. This assumes a group-major layout (each group's
      rewards adjacent, every group full-size) and tolerates only trailing
      raggedness: a final partial block is kept as a smaller group, while
      interior raggedness silently mis-groups -- pass ``group_ids`` for that.
      For exact-multiple input this is numerically identical to the legacy
      ``reshape(num_groups, group_size)`` path.

    For each ``k`` in the pass@k list, pass@k is averaged only over groups with
    at least ``k`` samples; undersized groups are skipped and a rung is dropped
    entirely when no group qualifies.
    """
    if group_size == 1:
        return {}

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    if group_ids is not None:
        assert len(group_ids) == len(flat_rewards), f"{len(group_ids)=} {len(flat_rewards)=}"
        buckets: dict[Any, list[float]] = defaultdict(list)
        for reward, gid in zip(flat_rewards, group_ids, strict=True):
            buckets[gid].append(reward)
        groups = [np.array(rewards) for rewards in buckets.values()]
    else:
        rewards = np.asarray(flat_rewards)
        groups = [rewards[i : i + group_size] for i in range(0, len(rewards), group_size)]

    group_sizes = np.array([len(g) for g in groups])
    group_correct = np.array([int(np.sum(g == 1)) for g in groups])

    log_dict = {}
    for k in pass_rate_name_list:
        eligible = group_sizes >= k
        if not eligible.any():
            continue

        pass_k_estimates = _estimate_pass_at_k(group_sizes[eligible], group_correct[eligible], k)
        log_dict[f"pass@{k}"] = np.mean(pass_k_estimates)

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
