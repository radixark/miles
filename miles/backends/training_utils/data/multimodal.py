"""Multimodal-specific preprocessing for rollout data.

The rollout side emits one media placeholder/sentinel per media item; training
expands it to the per-item token count so the LM sees a position per vision
patch / audio frame. Two families live here: Kimi-VL / Kimi-K2.5 (grid-derived
expansion of an in-vocab placeholder) and Inkling (out-of-vocab sentinels expanded
to in-vocab placeholder runs with explicit positions). Kept separate from
data.py (generic batching / CP slicing).
"""

import logging
from collections.abc import Sequence

import torch

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch, WeightVersionSpan, WeightVersionsPerCall

logger = logging.getLogger(__name__)

# Kimi-K2.5 / Kimi-VL media placeholder token id.
KIMI_VL_MEDIA_TOKEN_ID = 163605
# Kimi-VL tpool_patch_merger collapses a 2x2 spatial patch into one token.
_KIMI_VL_MERGE_H = 2
_KIMI_VL_MERGE_W = 2


def _shifted_token_index(index: int, expansions: Sequence[tuple[int, int]]) -> int:
    """Map a pre-expansion token boundary ``index`` to its post-expansion position.

    Each entry ``(pos, delta)`` represents replacing a single token at ``pos``
    with ``1 + delta`` tokens. Any boundary strictly after ``pos`` shifts by
    ``delta``; a boundary at ``pos`` stays at the start of the replacement run.
    """
    offset = 0
    for pos, delta in expansions:
        if pos < index:
            offset += delta
    return index + offset


def _remap_weight_version_call(
    call: WeightVersionsPerCall | dict | list[dict],
    expansions: Sequence[tuple[int, int]],
) -> WeightVersionsPerCall | dict | list[dict]:
    """Shift output span positions and ``output_start`` for a single call after media expansion.

    ``prefill_spans`` and ``prompt_tokens`` stay untouched because the engine already
    reports them in expanded engine-token coordinates (see ``WeightVersionsPerCall``).
    Preserves the caller's representation (``WeightVersionsPerCall``, dict, or legacy span list).
    """
    if not expansions:
        return call

    def _remap_span_dict(span: dict) -> dict:
        return {
            **span,
            "abs_start": _shifted_token_index(span["abs_start"], expansions),
            "abs_end": _shifted_token_index(span["abs_end"], expansions),
        }

    if isinstance(call, WeightVersionsPerCall):
        return WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(
                    version=span.version,
                    abs_start=_shifted_token_index(span.abs_start, expansions),
                    abs_end=_shifted_token_index(span.abs_end, expansions),
                )
                for span in call.spans
            ],
            prefill_spans=list(call.prefill_spans),
            output_start=(None if call.output_start is None else _shifted_token_index(call.output_start, expansions)),
            prompt_tokens=call.prompt_tokens,
        )

    if isinstance(call, list):
        return [_remap_span_dict(span) for span in call]

    remapped_call = dict(call)
    remapped_call["spans"] = [_remap_span_dict(span) for span in call["spans"]]
    if call.get("output_start") is not None:
        remapped_call["output_start"] = _shifted_token_index(call["output_start"], expansions)
    return remapped_call


def _remap_weight_versions_for_sample(
    weight_versions: Sequence[WeightVersionsPerCall | dict | list[dict]] | None,
    expansions: Sequence[tuple[int, int]],
) -> list[WeightVersionsPerCall | dict | list[dict]] | None:
    if weight_versions is None or not expansions:
        return weight_versions
    return [_remap_weight_version_call(call, expansions) for call in weight_versions]


def _num_image_tokens_from_grid(
    grid_thw: torch.Tensor, merge_h: int = _KIMI_VL_MERGE_H, merge_w: int = _KIMI_VL_MERGE_W
) -> int:
    _, h, w = grid_thw.tolist()
    # tpool_patch_merger averages over the temporal dimension T, so the
    # actual number of tokens per image depends only on the spatial grid.
    return (h // merge_h) * (w // merge_w)


def _expand_image_tokens_and_record_shifts(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    if grid_thws is None or len(grid_thws) == 0:
        return tokens, loss_mask, []

    placeholder_positions = (tokens == media_token_id).nonzero(as_tuple=True)[0]
    if len(placeholder_positions) == 0:
        return tokens, loss_mask, []

    num_placeholders = len(placeholder_positions)
    num_grids = len(grid_thws)
    expected_total_image_tokens = sum(_num_image_tokens_from_grid(grid_thw) for grid_thw in grid_thws)
    if num_placeholders == expected_total_image_tokens:
        # Already pre-expanded. Keep this helper idempotent because the same
        # rollout batch may pass through multiple normalization paths.
        return tokens, loss_mask, []
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
    expansions: list[tuple[int, int]] = []

    for i, pos in enumerate(reversed(placeholder_positions)):
        pos = pos.item()
        grid_idx = num_placeholders - 1 - i
        if grid_idx >= num_grids:
            continue

        _, h, w = grid_thws[grid_idx].tolist()
        num_image_tokens = (h // merge_h) * (w // merge_w)
        if num_image_tokens != 1:
            expansions.append((pos, num_image_tokens - 1))

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

    expansions.reverse()
    return expanded_tokens, expanded_mask, expansions


def _expand_image_tokens_for_sample(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_tokens, expanded_mask, _ = _expand_image_tokens_and_record_shifts(
        tokens,
        loss_mask,
        grid_thws,
        media_token_id=media_token_id,
    )
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


INKLING_IMAGE_SENTINEL_ID = -101
INKLING_AUDIO_SENTINEL_ID = -102
INKLING_MM_PLACEHOLDER_TOKEN_ID = 200023
INKLING_MM_AUDIO_PLACEHOLDER_TOKEN_ID = 200025


def _expand_inkling_sample_and_record_shifts(
    token_tensor, loss_mask, mm, sample_idx: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Replace media sentinels with placeholder runs, recording positions into mm and token shifts; idempotent."""
    spec = (
        (INKLING_IMAGE_SENTINEL_ID, INKLING_MM_PLACEHOLDER_TOKEN_ID, "mm_vision_num_patches", "mm_vision_positions"),
        (
            INKLING_AUDIO_SENTINEL_ID,
            INKLING_MM_AUDIO_PLACEHOLDER_TOKEN_ID,
            "mm_audio_num_tokens",
            "mm_audio_positions",
        ),
    )
    splices = []
    for sentinel, placeholder, counts_key, positions_key in spec:
        counts = mm.get(counts_key) if mm else None
        positions = (token_tensor == sentinel).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        assert counts is not None and positions.numel() == len(counts), (
            f"sample {sample_idx}: {positions.numel()} sentinel(s) {sentinel} but "
            f"{counts_key}={'missing' if counts is None else len(counts)}"
        )
        splices.append((positions.tolist(), [int(c) for c in counts], placeholder, positions_key))

    if not splices:
        return token_tensor, []

    prompt_len = len(token_tensor) - len(loss_mask)
    flat = sorted(
        (pos, n, placeholder, positions_key)
        for positions, counts, placeholder, positions_key in splices
        for pos, n in zip(positions, counts, strict=True)
    )
    assert all(
        p < prompt_len for p, _, _, _ in flat
    ), "Inkling media sentinels must be in the prompt; found one in the response"

    pieces, prev, shift = [], 0, 0
    out_positions: dict[str, list[int]] = {}
    expansions: list[tuple[int, int]] = []
    for pos, n, placeholder, positions_key in flat:
        pieces.append(token_tensor[prev:pos])
        pieces.append(torch.full((n,), placeholder, dtype=token_tensor.dtype, device=token_tensor.device))
        start = pos + shift
        out_positions.setdefault(positions_key, []).extend(range(start, start + n))
        if n != 1:
            expansions.append((pos, n - 1))
        shift += n - 1
        prev = pos + 1
    pieces.append(token_tensor[prev:])
    for positions_key, plist in out_positions.items():
        mm[positions_key] = torch.tensor(plist, dtype=torch.long, device=token_tensor.device)
    return torch.cat(pieces), expansions


def _expand_inkling_sample(token_tensor, loss_mask, mm, sample_idx: int):
    """Replace media sentinels with placeholder runs, recording their sample-local positions into mm; idempotent."""
    expanded, _ = _expand_inkling_sample_and_record_shifts(token_tensor, loss_mask, mm, sample_idx)
    return expanded


def _expand_inkling_rollout_data_in_place(rollout_data: RolloutBatch) -> None:
    mm_list = rollout_data["multimodal_train_inputs"]
    tokens = rollout_data["tokens"]
    loss_masks = rollout_data["loss_masks"]
    old_total_lengths = list(rollout_data["total_lengths"])
    weight_versions = rollout_data.get("weight_versions")

    new_tokens_list, new_total_lengths = [], []
    remapped_weight_versions = [] if weight_versions is not None else None
    for i, (token_tensor, loss_mask) in enumerate(zip(tokens, loss_masks, strict=False)):
        mm = mm_list[i] if i < len(mm_list) else None
        expanded, expansions = _expand_inkling_sample_and_record_shifts(token_tensor, loss_mask, mm, i)
        new_tokens_list.append(expanded)
        new_total_lengths.append(expanded.size(0))
        if remapped_weight_versions is not None:
            sample_wv = weight_versions[i] if i < len(weight_versions) else None
            remapped_weight_versions.append(_remap_weight_versions_for_sample(sample_wv, expansions))

    if new_total_lengths != old_total_lengths:
        try:
            cp_size = get_parallel_state().cp.size
        except Exception:
            cp_size = 1
        assert cp_size == 1, "Inkling multimodal expansion does not support CP>1 yet"
        rollout_data["tokens"] = new_tokens_list
        rollout_data["total_lengths"] = new_total_lengths
        if remapped_weight_versions is not None:
            rollout_data["weight_versions"] = remapped_weight_versions
        logger.info(
            "Expanded Inkling image sentinels: total_lengths %s -> %s",
            old_total_lengths,
            new_total_lengths,
        )


def expand_multimodal_rollout_data_in_place(
    rollout_data: RolloutBatch,
    media_token_id: int = KIMI_VL_MEDIA_TOKEN_ID,
    qkv_format: str = "thd",
) -> None:
    multimodal_train_inputs = rollout_data.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None and any(
        mm is not None and ("mm_vision_num_patches" in mm or "mm_audio_num_tokens" in mm)
        for mm in multimodal_train_inputs
    ):
        _expand_inkling_rollout_data_in_place(rollout_data)
        return
    mm_inputs_list = _collect_multimodal_grid_inputs(multimodal_train_inputs)
    if not mm_inputs_list or not any(mm is not None for mm in mm_inputs_list):
        return

    tokens = rollout_data["tokens"]
    if not _batch_has_media_placeholders(tokens, media_token_id=media_token_id):
        return

    loss_masks = rollout_data["loss_masks"]
    old_total_lengths = list(rollout_data["total_lengths"])
    old_response_lengths = list(rollout_data["response_lengths"])
    weight_versions = rollout_data.get("weight_versions")

    token_or_mask_changed = False
    expanded_tokens = []
    expanded_loss_masks = []
    expanded_total_lengths = []
    expanded_response_lengths = []
    remapped_weight_versions = [] if weight_versions is not None else None

    for i, (token_tensor, loss_mask_tensor) in enumerate(zip(tokens, loss_masks, strict=False)):
        expansions: list[tuple[int, int]] = []
        if mm_inputs_list[i] is not None:
            new_tokens, new_loss_mask, expansions = _expand_image_tokens_and_record_shifts(
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
        if remapped_weight_versions is not None:
            sample_wv = weight_versions[i] if i < len(weight_versions) else None
            remapped_weight_versions.append(_remap_weight_versions_for_sample(sample_wv, expansions))

    rollout_data["tokens"] = expanded_tokens
    rollout_data["loss_masks"] = expanded_loss_masks
    rollout_data["total_lengths"] = expanded_total_lengths
    rollout_data["response_lengths"] = expanded_response_lengths
    if remapped_weight_versions is not None:
        rollout_data["weight_versions"] = remapped_weight_versions

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
