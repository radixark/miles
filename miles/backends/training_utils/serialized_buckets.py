"""Align per-rank serialized weight buckets after a colocated gather."""

from collections.abc import Sequence

import torch


def align_serialized_bucket_columns(per_rank_buckets: Sequence[Sequence[object]]) -> list[list[object | None]]:
    """Zip per-rank bucket lists into columns.

    A short rank (including an empty one) contributes ``None`` so the gather
    source can pad instead of indexing past the list.
    """
    if not per_rank_buckets:
        return []
    width = max(len(buckets) for buckets in per_rank_buckets)
    return [[buckets[i] if i < len(buckets) else None for buckets in per_rank_buckets] for i in range(width)]


def empty_flattened_tensor_data(*, device: torch.device | str | int) -> dict:
    """Placeholder flattened bucket for a rank that contributed no tensors."""
    return {
        "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=device),
        "metadata": [],
    }
