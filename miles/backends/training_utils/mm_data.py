"""Multimodal-specific preprocessing and tensor movement for rollout data.

Two concerns live here, kept out of data.py (generic batching / CP slicing)
so additional VL models can land without inflating that module:

1. Token expansion (currently Kimi-VL / Kimi-K2.5-only): the rollout side
   emits one media placeholder token per image, and training expands it to
   grid-derived token counts so the LM sees a position per vision patch.
2. Multimodal tensor materialization and per-microbatch collation
   (model-agnostic): moves image payloads (``pixel_values``, ``image_grid_thw``,
   ...) between host and device. When ``--defer-multimodal-cuda-transfer`` is
   set these tensors stay on pinned CPU memory after the rollout fetch and only
   the active microbatch is copied to CUDA during collation, which avoids
   holding a whole minibatch of vision features in GPU memory at once.
"""

import logging
from collections.abc import Sequence

import numpy as np
import torch

from miles.utils.types import RolloutBatch

from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)

MultimodalValue = np.ndarray | torch.Tensor
MultimodalInput = dict[str, MultimodalValue] | None

# Kimi-K2.5 / Kimi-VL media placeholder token id.
KIMI_VL_MEDIA_TOKEN_ID = 163605
# Kimi-VL tpool_patch_merger collapses a 2x2 spatial patch into one token.
_KIMI_VL_MERGE_H = 2
_KIMI_VL_MERGE_W = 2


def _num_image_tokens_from_grid(
    grid_thw: torch.Tensor, merge_h: int = _KIMI_VL_MERGE_H, merge_w: int = _KIMI_VL_MERGE_W
) -> int:
    _, h, w = grid_thw.tolist()
    # tpool_patch_merger averages over the temporal dimension T, so the
    # actual number of tokens per image depends only on the spatial grid.
    return (h // merge_h) * (w // merge_w)


def _expand_image_tokens_for_sample(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grid_thws is None or len(grid_thws) == 0:
        return tokens, loss_mask

    placeholder_positions = (tokens == media_token_id).nonzero(as_tuple=True)[0]
    if len(placeholder_positions) == 0:
        return tokens, loss_mask

    num_placeholders = len(placeholder_positions)
    num_grids = len(grid_thws)
    expected_total_image_tokens = sum(_num_image_tokens_from_grid(grid_thw) for grid_thw in grid_thws)
    if num_placeholders == expected_total_image_tokens:
        # Already pre-expanded. Keep this helper idempotent because the same
        # rollout batch may pass through multiple normalization paths.
        return tokens, loss_mask
    if num_placeholders != num_grids:
        logger.warning(
            "K25 multimodal token mismatch before training: placeholders=%s, grids=%s",
            num_placeholders,
            num_grids,
        )

    merge_h, merge_w = _KIMI_VL_MERGE_H, _KIMI_VL_MERGE_W
    prompt_len = len(tokens) - len(loss_mask)

    expanded_tokens = tokens.clone()
    expanded_mask = loss_mask.clone()

    for i, pos in enumerate(reversed(placeholder_positions)):
        pos = pos.item()
        grid_idx = num_placeholders - 1 - i
        if grid_idx >= num_grids:
            continue

        _, h, w = grid_thws[grid_idx].tolist()
        num_image_tokens = (h // merge_h) * (w // merge_w)

        expanded_placeholder = torch.full(
            (num_image_tokens,), media_token_id, dtype=expanded_tokens.dtype, device=expanded_tokens.device
        )
        expanded_tokens = torch.cat([expanded_tokens[:pos], expanded_placeholder, expanded_tokens[pos + 1 :]])

        if pos >= prompt_len:
            mask_pos = pos - prompt_len
            expanded_mask_tokens = torch.zeros(
                num_image_tokens, dtype=expanded_mask.dtype, device=expanded_mask.device
            )
            expanded_mask = torch.cat([expanded_mask[:mask_pos], expanded_mask_tokens, expanded_mask[mask_pos + 1 :]])

    return expanded_tokens, expanded_mask


def _collect_multimodal_grid_inputs(
    multimodal_train_inputs: Sequence[dict[str, torch.Tensor] | None] | None,
) -> list[dict[str, torch.Tensor] | None]:
    if multimodal_train_inputs is None:
        return []

    mm_inputs_list = []
    for mm_dict in multimodal_train_inputs:
        if mm_dict is not None and "grid_thws" in mm_dict:
            mm_inputs_list.append(mm_dict)
        else:
            mm_inputs_list.append(None)
    return mm_inputs_list


def _batch_has_media_placeholders(
    tokens: Sequence[torch.Tensor],
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
) -> bool:
    return any((token_tensor == media_token_id).any().item() for token_tensor in tokens)


def expand_multimodal_rollout_data_in_place(
    rollout_data: RolloutBatch,
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
    qkv_format: str = "thd",
) -> None:
    multimodal_train_inputs = rollout_data.get("multimodal_train_inputs", None)
    mm_inputs_list = _collect_multimodal_grid_inputs(multimodal_train_inputs)
    if not mm_inputs_list or not any(mm is not None for mm in mm_inputs_list):
        return

    tokens = rollout_data["tokens"]
    if not _batch_has_media_placeholders(tokens, media_token_id=media_token_id):
        return

    loss_masks = rollout_data["loss_masks"]
    old_total_lengths = list(rollout_data["total_lengths"])
    old_response_lengths = list(rollout_data["response_lengths"])

    token_or_mask_changed = False
    expanded_tokens = []
    expanded_loss_masks = []
    expanded_total_lengths = []
    expanded_response_lengths = []

    for i, (token_tensor, loss_mask_tensor) in enumerate(zip(tokens, loss_masks, strict=False)):
        if mm_inputs_list[i] is not None:
            new_tokens, new_loss_mask = _expand_image_tokens_for_sample(
                token_tensor,
                loss_mask_tensor,
                mm_inputs_list[i]["grid_thws"],
                media_token_id=media_token_id,
            )
            token_or_mask_changed = token_or_mask_changed or (
                (new_tokens.size(0) != token_tensor.size(0)) or (new_loss_mask.size(0) != loss_mask_tensor.size(0))
            )
            expanded_tokens.append(new_tokens)
            expanded_loss_masks.append(new_loss_mask)
            expanded_total_lengths.append(new_tokens.size(0))
            expanded_response_lengths.append(new_loss_mask.size(0))
        else:
            expanded_tokens.append(token_tensor)
            expanded_loss_masks.append(loss_mask_tensor)
            expanded_total_lengths.append(old_total_lengths[i])
            expanded_response_lengths.append(old_response_lengths[i])

    rollout_data["tokens"] = expanded_tokens
    rollout_data["loss_masks"] = expanded_loss_masks
    rollout_data["total_lengths"] = expanded_total_lengths
    rollout_data["response_lengths"] = expanded_response_lengths

    metadata_changed = (expanded_total_lengths != old_total_lengths) or (
        expanded_response_lengths != old_response_lengths
    )
    if metadata_changed:
        parallel_state = get_parallel_state()
        cp_size = parallel_state.cp.size
        if cp_size > 1 and qkv_format == "thd":
            for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):
                values = rollout_data.get(key)
                if not values:
                    continue
                rollout_data[key] = [
                    slice_log_prob_with_cp(
                        all_gather_with_cp(value, old_total_length, old_response_length),
                        new_total_length,
                        new_response_length,
                        qkv_format,
                    )
                    for value, old_total_length, old_response_length, new_total_length, new_response_length in zip(
                        values,
                        old_total_lengths,
                        old_response_lengths,
                        expanded_total_lengths,
                        expanded_response_lengths,
                        strict=False,
                    )
                ]
        logger.info(
            "Adjusted multimodal rollout metadata for Kimi VL: "
            f"token_or_mask_changed={token_or_mask_changed}, "
            f"total_lengths_changed={expanded_total_lengths != old_total_lengths}, "
            f"response_lengths_changed={expanded_response_lengths != old_response_lengths}"
        )


def _as_multimodal_tensor(value: MultimodalValue, device: torch.device | int | None) -> torch.Tensor:
    """Convert one multimodal array-like value to a tensor on the requested device."""
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value.copy())
    elif isinstance(value, torch.Tensor):
        # Tensor inputs are already trainer-owned; cloning would duplicate large
        # image payloads, including CUDA tensors in preload mode.
        tensor = value
    else:
        raise TypeError(f"Expected multimodal value to be a tensor or ndarray, got {type(value).__name__}")
    return tensor.to(device=device)


def materialize_multimodal_inputs(
    multimodal_inputs: Sequence[MultimodalInput],
    device: torch.device | int,
) -> list[dict[str, torch.Tensor] | None]:
    """Resolve per-sequence multimodal sample dicts to tensors on ``device``.

    Called once per rollout fetch. With ``device`` set to the CUDA device this
    preloads every image payload (legacy behavior); with ``device`` set to CPU
    (deferred mode) the tensors stay on pinned host memory so the later
    non_blocking CPU->CUDA copy in :func:`collate_multimodal_train_inputs` can
    overlap with compute.
    """
    materialized: list[dict[str, torch.Tensor] | None] = []
    for multimodal_input in multimodal_inputs:
        if multimodal_input is None:
            materialized.append(None)
            continue

        materialized_input: dict[str, torch.Tensor] = {}
        for key, value in multimodal_input.items():
            tensor = _as_multimodal_tensor(value, device)
            if isinstance(device, torch.device) and device.type == "cpu":
                # Deferred tensors stay on CPU here; pin host memory so the
                # later non_blocking CPU->CUDA copy can overlap with compute.
                # pinning makes sure that the memory doesn't get paged out to disk
                # which would make the copy synchronous.
                tensor = tensor.pin_memory() if torch.cuda.is_available() else tensor
            materialized_input[key] = tensor
        materialized.append(materialized_input)
    return materialized


def _cat_multimodal_tensors_for_forward(tensors: list[torch.Tensor], device: torch.device | int) -> torch.Tensor:
    """Concatenate one multimodal field and place only the active microbatch on CUDA.

    The per-tensor ``.to()`` runs *before* the concat on purpose: ``torch.cat``
    on CPU allocates a fresh *pageable* output tensor, dropping the pinning that
    :func:`materialize_multimodal_inputs` applied. Concatenating first would
    therefore force a synchronous host->device copy. Moving each (pinned) tensor
    over first keeps the copy ``non_blocking`` so it overlaps with compute;
    already-CUDA tensors (preload mode) make ``.to()`` a no-op.
    """
    tensors = [tensor.to(device=device, non_blocking=True) for tensor in tensors]
    return torch.cat(tensors, dim=0)


def collate_multimodal_train_inputs(
    multimodal_train_inputs: Sequence[MultimodalInput],
    device: torch.device | int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Collate the already-sliced microbatch multimodal payload for model forward.

    Args:
        multimodal_train_inputs: Per-sequence multimodal dicts selected by
            ``DataIterator`` for the active microbatch.
        device: CUDA device used by the current Megatron rank.

    Returns:
        A tuple of ``(multimodal_data, multimodal_num_items)``. Tensor values are
        concatenated across the active microbatch and placed on ``device``.
    """
    values_by_key: dict[str, list[torch.Tensor]] = {}
    multimodal_num_items: dict[str, list[int]] = {}
    for mm_input_dict in multimodal_train_inputs:
        if mm_input_dict is None:
            continue
        if not isinstance(mm_input_dict, dict):
            raise TypeError(
                f"Expected multimodal_train_inputs entries to be dict or None, got {type(mm_input_dict).__name__}"
            )
        for key, value in mm_input_dict.items():
            tensor = _as_multimodal_tensor(value, None)
            values_by_key.setdefault(key, []).append(tensor)
            multimodal_num_items.setdefault(key, []).append(tensor.size(0))

    multimodal_data = {
        key: _cat_multimodal_tensors_for_forward(values, device) for key, values in values_by_key.items()
    }
    return multimodal_data, multimodal_num_items
