from collections.abc import Mapping, Sequence

import torch

from miles.utils.sampling_mask import RolloutSamplingMask


def get_rollout_sampling_masks(batch: Mapping[str, object]) -> list[RolloutSamplingMask]:
    """Reconstruct the per-sample masks carried by the training-data wire."""
    ids_batch = batch.get("rollout_sampling_mask_ids")
    offsets_batch = batch.get("rollout_sampling_mask_offsets")
    if ids_batch is None or offsets_batch is None:
        raise ValueError("top-p actor scoring requires both sampling-mask wire fields")
    if not isinstance(ids_batch, Sequence) or not isinstance(offsets_batch, Sequence):
        raise TypeError("rollout sampling-mask ids and offsets must be sequences with one entry per sample")
    if len(ids_batch) != len(offsets_batch):
        raise ValueError(f"sampling-mask ids batch size {len(ids_batch)} != offsets batch size {len(offsets_batch)}")
    return [
        RolloutSamplingMask(ids=torch.as_tensor(ids), offsets=torch.as_tensor(offsets))
        for ids, offsets in zip(ids_batch, offsets_batch, strict=True)
    ]


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask: RolloutSamplingMask,
    response_indices: Sequence[int] | torch.Tensor,
    *,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive.

    Args:
        logits: ``[local_rows, local_vocab_size]`` response-row logits this
            rank holds (TP vocab shard, CP row subset).
        sampling_mask: the sample's complete sampling mask.
        response_indices: ``[local_rows]`` global response position of each row.
        tp_rank: this rank's index in the TP group.

    Returns:
        Bool mask shaped like ``logits``; True marks ids inside the support.
    """
    if isinstance(response_indices, torch.Tensor) and (
        response_indices.ndim != 1
        or response_indices.dtype == torch.bool
        or torch.is_floating_point(response_indices)
        or torch.is_complex(response_indices)
    ):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
    if len(response_indices) != logits.size(0):
        raise ValueError(
            f"sampling-mask rows must align with logits: indices={len(response_indices)}, logits={logits.size(0)}"
        )

    if logits.size(0) == 0:
        return torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device).view_as(logits)

    # CP response rows form a small number of contiguous runs, so the CSR
    # gather is a handful of CPU slices before the GPU expansion.
    selected_ids, lengths = sampling_mask._select_masks(response_indices)
    selected_ids = selected_ids.to(logits.device)
    row_indices = torch.repeat_interleave(
        torch.arange(len(response_indices), dtype=torch.long, device=logits.device),
        lengths.to(device=logits.device, dtype=torch.long),
    )
    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    is_local = (selected_ids >= vocab_start) & (selected_ids < vocab_start + local_vocab_size)
    flat_local_indices = row_indices[is_local] * local_vocab_size + selected_ids[is_local].to(torch.long) - vocab_start
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[flat_local_indices] = True
    return mask.view_as(logits)
