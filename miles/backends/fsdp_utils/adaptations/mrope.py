from collections.abc import Mapping
from functools import partial

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations.arch_adapter import PositionIdsBuilder

_TEXT, _IMAGE, _VIDEO = 0, 1, 2

_ROPE_INPUT_KEYS = ("image_grid_thw", "video_grid_thw", "second_per_grid_ts")


class MropePositionIds:
    def position_ids_builder(self, model: nn.Module) -> PositionIdsBuilder:
        return partial(mrope_position_ids, model.model)


def mrope_position_ids(
    model: nn.Module, tokens: torch.Tensor, multimodal_inputs: Mapping[str, torch.Tensor] | None
) -> torch.Tensor:
    """One document's ``[text, temporal, height, width]`` positions, as ``[4, seq]``."""
    text_positions = torch.arange(tokens.size(0), device=tokens.device, dtype=torch.long)

    rope_inputs = {
        key: multimodal_inputs[key]
        for key in _ROPE_INPUT_KEYS
        if multimodal_inputs is not None and multimodal_inputs.get(key) is not None
    }
    if "image_grid_thw" not in rope_inputs and "video_grid_thw" not in rope_inputs:
        return text_positions.expand(4, -1)

    mrope_positions, _ = model.get_rope_index(
        tokens.unsqueeze(0), mm_token_type_ids=_mm_token_type_ids(model.config, tokens).unsqueeze(0), **rope_inputs
    )
    return torch.cat([text_positions.unsqueeze(0), mrope_positions.squeeze(1)], dim=0)


def _mm_token_type_ids(config, tokens: torch.Tensor) -> torch.Tensor:
    image = tokens == config.image_token_id
    mm_token_type_ids = torch.full_like(tokens, _TEXT)
    mm_token_type_ids[image] = _IMAGE
    if (video_token_id := getattr(config, "video_token_id", None)) is not None:
        mm_token_type_ids[tokens == video_token_id] = _VIDEO
    # GLM-V marks video frames with the image token, between the video's start and end markers
    start, end = getattr(config, "video_start_token_id", None), getattr(config, "video_end_token_id", None)
    if start is not None and end is not None:
        in_video = torch.cumsum(tokens == start, dim=0) > torch.cumsum(tokens == end, dim=0)
        mm_token_type_ids[image & in_video] = _VIDEO
    return mm_token_type_ids
