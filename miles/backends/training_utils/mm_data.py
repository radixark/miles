"""Multimodal-specific preprocessing for rollout data.

Currently Kimi-VL / Kimi-K2.5-only: the rollout side emits one media
placeholder token per image, and training expands it to grid-derived token
counts so the LM sees a position per vision patch.  Kept separate from
data.py (generic batching / CP slicing) so additional VL models can land
without inflating that module.
"""

import logging
from collections.abc import Sequence

import torch

from miles.utils.types import RolloutBatch

from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)

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
            for key in ("rollout_log_probs", "teacher_log_probs"):
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
