"""The frozen GLM-5.3 vision tower, built on transformers' GlmOcr vision classes.

Ported from fti (fleet-ai/theseus ``services/fti/src/fti/trainers/miles/vision/glm5_next.py``), which
trains GLM-5.3-Flash on the Megatron path with this tower; that code is in turn a port of the tower in
radixark/miles#2792. The tower is frozen: its weights load from the checkpoint's ``model.visual.*``
tensors and never enter the optimizer, the state-dict adapter or a training checkpoint.
"""

import functools
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchMerger,
)

from miles.utils.hf_utils.config import load_hf_config

_VISUAL_PREFIX = "model.visual."


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config) -> None:
        super().__init__(config, bias=config.attention_bias)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mlp = Glm5NextVisionMLP(config)


class Glm5NextVisionPatchMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config) -> None:
        super().__init__(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            bias=False,
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act1(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionModel(GlmOcrVisionModel):
    """The vision part of the official GLM-5.3 model: GlmOcr with clamped SwiGLU MLPs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(Glm5NextVisionBlock(config) for _ in range(config.depth))
        self.merger = Glm5NextVisionPatchMerger(config)


def _visual_weight_files(hf_checkpoint: str) -> dict[str, list[str]]:
    index_path = Path(hf_checkpoint) / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]
    names_by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        if name.startswith(_VISUAL_PREFIX):
            names_by_file.setdefault(filename, []).append(name)
    return names_by_file


def has_visual_weights(hf_checkpoint: str) -> bool:
    """Whether the sharded checkpoint carries ``model.visual.*`` tensors. The config cannot tell: the
    glm5_next alias builds a default vision_config for text-only checkpoints too."""
    return bool(_visual_weight_files(hf_checkpoint))


def _load_visual_state_dict(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    names_by_file = _visual_weight_files(hf_checkpoint)
    if not names_by_file:
        raise RuntimeError(f"no {_VISUAL_PREFIX} weights in {hf_checkpoint}")

    state_dict = {}
    for filename, names in names_by_file.items():
        with safe_open(f"{hf_checkpoint}/{filename}", framework="pt") as shard:
            for name in names:
                state_dict[name.removeprefix(_VISUAL_PREFIX)] = shard.get_tensor(name)
    return state_dict


@functools.cache
def build_glm5_next_visual(hf_checkpoint: str, *, device: torch.device, dtype: torch.dtype) -> Glm5NextVisionModel:
    """The checkpoint's frozen tower, strictly loaded. Cached so the actor and the reference model,
    which read the same checkpoint, share one copy on the GPU."""
    vision_config = load_hf_config(hf_checkpoint).vision_config
    vision_config._attn_implementation = "sdpa"

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            visual = Glm5NextVisionModel(vision_config)
    finally:
        torch.set_default_dtype(previous_dtype)

    visual.load_state_dict(_load_visual_state_dict(hf_checkpoint), strict=True)
    visual.requires_grad_(False)
    visual.eval()
    return visual
