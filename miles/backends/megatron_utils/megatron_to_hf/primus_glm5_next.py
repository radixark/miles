"""Megatron -> HF weight-update conversion for Primus' GLM-5.3-Flash (`model_type: glm5_next`).

The inverse of Primus' `glm5_next_hf_loader`. Primus keeps the HF names for the KDA and mHC
parameters and Megatron's MLA names for the DSA layers, which differs from Miles' own
`glm5_next` model, hence a separate converter. Names are emitted text-only
(`model.layers.N...`); SGLang's `Glm5NextForConditionalGeneration` strips `language_model.`
from checkpoint names, so these load the same way.

Primus may hold a tensor in a different but same-sized shape (e.g. the KDA short-conv
weights), so each output is reshaped to the checkpoint's own shape.
"""

import functools
import json
import os
import re
import struct

import torch

_LAYER_PATTERN = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_EXPERT_PATTERN = re.compile(r"mlp\.experts\.linear_fc([12])\.weight(\d+)")

_DSA_TO_HF = {
    "linear_q_down_proj": "q_a_proj",
    "q_layernorm": "q_a_layernorm",
    "linear_q_up_proj": "q_b_proj",
    "linear_kv_down_proj": "kv_a_proj_with_mqa",
    "kv_layernorm": "kv_a_layernorm",
    "linear_kv_up_proj": "kv_b_proj",
    "linear_proj": "o_proj",
}

_TOP_LEVEL = {
    "module.module.embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "module.module.output_layer.weight": "lm_head.weight",
    "module.module.decoder.final_layernorm.weight": "model.norm.weight",
}


@functools.cache
def _hf_shapes(hf_checkpoint: str) -> dict[str, tuple[int, ...]]:
    """Tensor shapes from the safetensors headers, keyed by text-only HF name."""
    with open(os.path.join(hf_checkpoint, "model.safetensors.index.json")) as f:
        files = sorted(set(json.load(f)["weight_map"].values()))
    shapes = {}
    for file_name in files:
        with open(os.path.join(hf_checkpoint, file_name), "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name.replace("model.language_model.", "model.")] = tuple(meta["shape"])
    return shapes


def _layer_to_hf(layer_idx: str, rest: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    prefix = f"model.layers.{layer_idx}."

    expert = _EXPERT_PATTERN.fullmatch(rest)
    if expert is not None:
        fc, expert_idx = expert.groups()
        expert_prefix = f"{prefix}mlp.experts.{expert_idx}."
        if fc == "1":
            gate, up = param.chunk(2, dim=0)
            return [(expert_prefix + "gate_proj.weight", gate), (expert_prefix + "up_proj.weight", up)]
        return [(expert_prefix + "down_proj.weight", param)]

    if rest.startswith("hc_") or rest == "input_layernorm.weight":
        return [(prefix + rest, param)]
    if rest == "pre_mlp_layernorm.weight":
        return [(prefix + "post_attention_layernorm.weight", param)]

    if rest.startswith("mlp."):
        sub = rest[len("mlp.") :]
        mlp = prefix + "mlp."
        if sub in ("linear_fc1.weight", "shared_experts.linear_fc1.weight"):
            owner = mlp + ("shared_experts." if sub.startswith("shared_experts.") else "")
            gate, up = param.chunk(2, dim=0)
            return [(owner + "gate_proj.weight", gate), (owner + "up_proj.weight", up)]
        if sub == "linear_fc2.weight":
            return [(mlp + "down_proj.weight", param)]
        if sub == "shared_experts.linear_fc2.weight":
            return [(mlp + "shared_experts.down_proj.weight", param)]
        if sub == "router.weight":
            return [(mlp + "gate.weight", param)]
        if sub == "router.expert_bias":
            return [(mlp + "gate.e_score_correction_bias", param)]

    if rest.startswith("self_attention."):
        sub = rest[len("self_attention.") :]
        attn = prefix + "self_attn."
        head, _, tail = sub.partition(".")
        if head == "indexer":
            return [(attn + sub, param)]
        if head in _DSA_TO_HF:
            return [(attn + _DSA_TO_HF[head] + "." + tail, param)]
        # KDA keeps the HF names, apart from the output norm.
        return [(attn + sub.replace("out_norm.", "o_norm."), param)]

    raise ValueError(f"Unknown Primus GLM-5.3 parameter: layer {layer_idx}, {rest}")


def convert_primus_glm5_next_to_hf(args, name, param):
    if name in _TOP_LEVEL:
        converted = [(_TOP_LEVEL[name], param)]
    else:
        match = _LAYER_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Unknown Primus GLM-5.3 parameter: {name}")
        converted = _layer_to_hf(*match.groups(), param)

    shapes = _hf_shapes(args.hf_checkpoint)
    out = []
    for hf_name, tensor in converted:
        shape = shapes.get(hf_name)
        if shape is None:
            raise ValueError(f"{name} converts to {hf_name}, which is not in {args.hf_checkpoint}")
        if tuple(tensor.shape) != shape:
            tensor = tensor.reshape(shape)
        out.append((hf_name, tensor))
    return out
