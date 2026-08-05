from collections.abc import Mapping, Sequence

import torch


def get_rollout_sampling_mask(batch: Mapping[str, object]) -> tuple[object, object]:
    """Read the complete sampling mask required by an actor scoring pass."""
    sampling_mask_ids = batch.get("rollout_sampling_mask_ids")
    sampling_mask_offsets = batch.get("rollout_sampling_mask_offsets")
    if sampling_mask_ids is None or sampling_mask_offsets is None:
        raise ValueError("top-p actor scoring requires rollout_sampling_mask_ids and rollout_sampling_mask_offsets")
    return sampling_mask_ids, sampling_mask_offsets


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask_ids: Sequence[int] | torch.Tensor,
    sampling_mask_offsets: Sequence[int] | torch.Tensor,
    response_indices: Sequence[int] | torch.Tensor,
    *,
    response_length: int,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive."""
    ids = _to_int_list(sampling_mask_ids)
    offsets = _to_int_list(sampling_mask_offsets)
    indices = _to_int_list(response_indices)

    if len(indices) != logits.size(0):
        raise ValueError(f"sampling-mask rows must align with logits: indices={len(indices)}, logits={logits.size(0)}")
    if not offsets or offsets[0] != 0 or offsets[-1] != len(ids):
        raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
    if len(offsets) != response_length + 1:
        raise ValueError(f"sampling-mask offsets length {len(offsets)} != response length + 1 ({response_length + 1})")
    if any(start >= end for start, end in zip(offsets, offsets[1:])):
        raise ValueError("every response token must have a non-empty sampling support")

    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    row_indices: list[int] = []
    column_indices: list[int] = []

    for local_row, response_index in enumerate(indices):
        if response_index < 0 or response_index + 1 >= len(offsets):
            raise ValueError(
                f"response index {response_index} is outside sampling-mask offsets of length {len(offsets)}"
            )
        support = ids[offsets[response_index] : offsets[response_index + 1]]
        local_support = [token_id - vocab_start for token_id in support if vocab_start <= token_id < vocab_end]
        row_indices.extend([local_row] * len(local_support))
        column_indices.extend(local_support)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    if row_indices:
        mask[
            torch.tensor(row_indices, dtype=torch.long, device=logits.device),
            torch.tensor(column_indices, dtype=torch.long, device=logits.device),
        ] = True
    return mask


def _to_int_list(values: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [int(value) for value in values]
