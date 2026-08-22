from collections.abc import Mapping, Sequence
from typing import cast

import torch


def get_rollout_sampling_mask(batch: Mapping[str, object]) -> list[list[list[int]]]:
    """Read the complete sampling mask required by an actor scoring pass."""
    sampling_mask = batch.get("rollout_sampling_mask")
    if sampling_mask is None:
        raise ValueError("truncated-sampling actor scoring requires rollout_sampling_mask")
    return cast(list[list[list[int]]], sampling_mask)


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask: list[list[int]],
    response_indices: Sequence[int],
    *,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive."""
    indices = _to_cpu_integer_tensor(response_indices)

    if indices.numel() != logits.size(0):
        raise ValueError(
            f"sampling-mask rows must align with logits: indices={indices.numel()}, logits={logits.size(0)}"
        )
    if any(not support for support in sampling_mask):
        raise ValueError("every response token must have a non-empty sampling support")
    response_length = len(sampling_mask)
    if torch.any(indices < 0) or torch.any(indices >= response_length):
        raise ValueError(f"response indices must be in [0, {response_length})")

    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    if indices.numel() == 0:
        return mask.view_as(logits)

    selected_supports = [sampling_mask[index] for index in indices.tolist()]
    lengths = torch.tensor([len(support) for support in selected_supports], device=logits.device)
    selected_ids = _to_cpu_integer_tensor([token_id for support in selected_supports for token_id in support]).to(
        logits.device
    )
    row_indices = torch.repeat_interleave(
        torch.arange(indices.numel(), dtype=torch.long, device=logits.device),
        lengths,
    )
    is_local = (selected_ids >= vocab_start) & (selected_ids < vocab_end)
    flat_local_indices = row_indices[is_local] * local_vocab_size + selected_ids[is_local].to(torch.long) - vocab_start
    mask[flat_local_indices] = True
    return mask.view_as(logits)


def _to_cpu_integer_tensor(values: Sequence[int]) -> torch.Tensor:
    if len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask token ids and response indices must be one-dimensional integers")
    return tensor
