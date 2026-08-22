from collections.abc import Mapping, Sequence

import torch


def get_rollout_sampling_mask(batch: Mapping[str, object]) -> tuple[object, object]:
    """Read the complete sampling mask required by an actor scoring pass."""
    sampling_mask_ids = batch.get("rollout_sampling_mask_ids")
    sampling_mask_offsets = batch.get("rollout_sampling_mask_offsets")
    if sampling_mask_ids is None or sampling_mask_offsets is None:
        raise ValueError(
            "truncated-sampling actor scoring requires rollout_sampling_mask_ids and rollout_sampling_mask_offsets"
        )
    return sampling_mask_ids, sampling_mask_offsets


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask_ids: list[int],
    sampling_mask_offsets: list[int],
    response_indices: Sequence[int],
    *,
    response_length: int,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive."""
    ids = _to_cpu_integer_tensor(sampling_mask_ids)
    offsets = _to_cpu_integer_tensor(sampling_mask_offsets)
    indices = _to_cpu_integer_tensor(response_indices)

    if indices.numel() != logits.size(0):
        raise ValueError(
            f"sampling-mask rows must align with logits: indices={indices.numel()}, logits={logits.size(0)}"
        )
    if offsets.numel() == 0 or offsets[0] != 0 or offsets[-1] != ids.numel():
        raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
    if offsets.numel() != response_length + 1:
        raise ValueError(
            f"sampling-mask offsets length {offsets.numel()} != response length + 1 ({response_length + 1})"
        )
    if torch.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("every response token must have a non-empty sampling support")
    if torch.any(indices < 0) or torch.any(indices >= response_length):
        raise ValueError(f"response indices must be in [0, {response_length})")

    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    if indices.numel() == 0:
        return mask.view_as(logits)

    indices = indices.to(torch.long)
    lengths = offsets[indices + 1] - offsets[indices]
    # CP response rows form a small number of contiguous runs. Slice those
    # runs on CPU, then expand and TP-filter the CSR data on the GPU.
    run_starts = [0]
    run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
    run_starts.append(indices.numel())
    selected_parts = [
        ids[offsets[indices[start]] : offsets[indices[end - 1] + 1]]
        for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
    ]
    selected_ids = selected_parts[0] if len(selected_parts) == 1 else torch.cat(selected_parts)

    selected_ids = selected_ids.to(logits.device)
    row_indices = torch.repeat_interleave(
        torch.arange(indices.numel(), dtype=torch.long, device=logits.device),
        lengths.to(device=logits.device, dtype=torch.long),
    )
    is_local = (selected_ids >= vocab_start) & (selected_ids < vocab_end)
    flat_local_indices = row_indices[is_local] * local_vocab_size + selected_ids[is_local].to(torch.long) - vocab_start
    mask[flat_local_indices] = True
    return mask.view_as(logits)


def _to_cpu_integer_tensor(values: Sequence[int]) -> torch.Tensor:
    if isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
    return tensor
