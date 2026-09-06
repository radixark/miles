from collections.abc import Sequence

import torch

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks


def constrain_positive_advantages(
    advantages: list[torch.Tensor],
    spans_by_sample: Sequence[Sequence[Sequence[int]]] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Cap policy advantages at zero inside explicitly marked response spans.

    Call after normalization or any other advantage adjustment. Negative credit
    is preserved, and neither the input tensors (which may alias value targets)
    nor loss masks are modified. Separate KL/entropy losses retain their masks.
    Each half-open span indexes the full response, before context-parallel
    slicing, including generated stop tokens when they belong to the attempt.
    """
    if spans_by_sample is None:
        return advantages
    if not (len(spans_by_sample) == len(advantages) == len(response_lengths) == len(total_lengths)):
        raise ValueError("Non-positive-advantage spans must align with the training samples")
    if max_seq_lens is not None and len(max_seq_lens) != len(advantages):
        raise ValueError("max_seq_lens must align with the training samples")

    masks = []
    for advantage, spans, response_length in zip(advantages, spans_by_sample, response_lengths, strict=True):
        if not isinstance(spans, (list, tuple)):
            raise ValueError("Non-positive-advantage spans must be a list of [start, end] pairs")
        mask = torch.zeros(response_length, dtype=torch.bool, device=advantage.device)
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(offset) is not int for offset in span) or not 0 <= span[0] < span[1] <= response_length:
                raise ValueError(f"Invalid non-positive-advantage span {span!r} for response length {response_length}")
            mask[span[0] : span[1]] = True
        masks.append(mask)

    local_masks = get_local_response_loss_masks(total_lengths, response_lengths, masks, qkv_format, max_seq_lens)
    constrained = []
    for advantage, mask in zip(advantages, local_masks, strict=True):
        if advantage.shape != mask.shape:
            raise ValueError(f"Advantage shape {advantage.shape} does not match local credit mask {mask.shape}")
        constrained.append(torch.where(mask, advantage.clamp_max(0), advantage))
    return constrained
