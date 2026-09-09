"""Frozen Qwen3.8-Flash-Next visual tower and VLM forward wiring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from miles.utils.hf_config import load_hf_config

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

_VISUAL_PREFIX = "model.visual."


def _load_visual_state_dict(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    with open(f"{hf_checkpoint}/model.safetensors.index.json", encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    weights_by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        if name.startswith(_VISUAL_PREFIX):
            weights_by_file.setdefault(filename, []).append(name)
    if not weights_by_file:
        raise RuntimeError(f"No {_VISUAL_PREFIX} weights found in {hf_checkpoint}")

    state_dict = {}
    for filename, names in weights_by_file.items():
        with safe_open(f"{hf_checkpoint}/{filename}", framework="pt") as shard:
            for name in names:
                state_dict[name.removeprefix(_VISUAL_PREFIX)] = shard.get_tensor(name)
    return state_dict


def build_qwen3_8_next_visual(
    hf_checkpoint: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3_5MoeVisionModel:
    """Construct and strictly load the checkpoint's frozen visual tower."""
    hf_config = load_hf_config(hf_checkpoint)
    vision_config = hf_config.vision_config
    vision_config._attn_implementation = "sdpa"
    visual = Qwen3_5MoeVisionModel(vision_config)

    visual.load_state_dict(_load_visual_state_dict(hf_checkpoint), strict=True)
    visual.to(device=device, dtype=dtype)
    visual.requires_grad_(False)
    visual.eval()
    return visual


def _scatter_visual_embeddings(
    model: GPTModel,
    decoder_input: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    from megatron.core import parallel_state

    positions = torch.nonzero(input_ids.reshape(-1) == token_id, as_tuple=False).flatten()
    if positions.numel() != embeddings.shape[0]:
        raise ValueError(
            f"Qwen3.8-Flash-Next visual token/feature mismatch: {positions.numel()} token(s), "
            f"{embeddings.shape[0]} feature row(s)"
        )

    embeddings = embeddings.to(device=decoder_input.device, dtype=decoder_input.dtype)
    if model.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
        sequence_length = decoder_input.shape[0]
        rank = parallel_state.get_tensor_model_parallel_rank()
        local_positions = positions.to(decoder_input.device) - rank * sequence_length
        selected = (local_positions >= 0) & (local_positions < sequence_length)
        local_positions = local_positions[selected]
        embeddings = embeddings[selected]
    else:
        local_positions = positions.to(decoder_input.device)

    decoder_input = decoder_input.clone()
    if local_positions.numel():
        decoder_input[local_positions, 0] = embeddings
    return decoder_input


def _position_ids(
    hf_config,
    input_ids: torch.Tensor,
    packed_seq_params,
    image_grid_thw: torch.Tensor | None,
) -> torch.Tensor:
    from miles_plugins.models.qwen3_vl import get_qwen3_vl_position_ids

    return get_qwen3_vl_position_ids(
        input_ids,
        packed_seq_params=packed_seq_params,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
    )


def _wire_mrope(model: GPTModel, hf_config) -> None:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding

    if model.config.context_parallel_size != 1:
        raise NotImplementedError("Qwen3.8-Flash-Next VLM training currently requires context parallel size 1")

    rope_parameters = hf_config.text_config.rope_parameters
    mrope_section = list(rope_parameters["mrope_section"])
    model.config.position_embedding_type = "mrope"
    model.config.mrope_section = mrope_section
    model.config.apply_rope_fusion = False
    model.position_embedding_type = "mrope"
    model.mrope_section = mrope_section
    model.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(
        kv_channels=model.config.kv_channels,
        rotary_percent=model.rotary_percent,
        rotary_interleaved=model.config.rotary_interleaved,
        rotary_base=model.rotary_base,
        cp_group=model.pg_collection.cp,
    )


def wire_qwen3_8_next_visual(model: GPTModel, hf_checkpoint: str) -> None:
    """Attach image handling without registering frozen tower parameters."""
    hf_config = load_hf_config(hf_checkpoint)
    _wire_mrope(model, hf_config)
    original_forward = model.forward

    if model.pre_process:
        device = torch.device("cuda", torch.cuda.current_device())
        visual = build_qwen3_8_next_visual(hf_checkpoint, device=device, dtype=model.config.params_dtype)
        model.__dict__["_qwen3_8_next_visual"] = visual

    def _multimodal_forward(*args, pixel_values=None, image_grid_thw=None, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen3.8-Flash-Next VLM training expects packed input_ids with shape [1, sequence]")
        kwargs["position_ids"] = _position_ids(
            hf_config,
            input_ids,
            kwargs.get("packed_seq_params"),
            image_grid_thw,
        )

        if not model.pre_process or pixel_values is None:
            return original_forward(*args, **kwargs)

        if image_grid_thw is None:
            raise ValueError("pixel_values requires image_grid_thw")
        decoder_input = model.embedding(input_ids=input_ids, position_ids=kwargs["position_ids"])
        with torch.no_grad():
            image_embeddings = model.__dict__["_qwen3_8_next_visual"](
                pixel_values.to(device=decoder_input.device, dtype=model.config.params_dtype),
                grid_thw=image_grid_thw.to(device=decoder_input.device),
            ).pooler_output
        kwargs["decoder_input"] = _scatter_visual_embeddings(
            model,
            decoder_input,
            input_ids,
            hf_config.image_token_id,
            image_embeddings,
        )
        return original_forward(*args, **kwargs)

    model.forward = _multimodal_forward
