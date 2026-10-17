"""Map Step5 Megatron weights to HF names for export and SGLang weight updates.

Routed experts use per-expert names; gathered QKV/gate blocks are split by TP rank.
The frozen indexer is exported and synced like any other weight.
"""

import re

from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES as INDEXER_CHECKPOINT_NAMES
from miles_plugins.models.step5.qkv_layout import split_qkv_and_gate

_DIRECT_NAMES = {
    "module.module.embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "module.module.output_layer.weight": "lm_head.weight",
    "module.module.decoder.final_layernorm.weight": "model.norm.weight",
}

_LAYER = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_ROUTED_EXPERT = re.compile(r"mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)")

_LAYER_NAMES = {
    "input_layernorm.weight": "input_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    "mlp.router.weight": "moe.gate.weight",
    "mlp.router.expert_bias": "moe.router_bias",
    "mlp.shared_experts.linear_fc2.weight": "share_expert.down_proj.weight",
    **{
        f"self_attention.indexer.{name}": f"self_attn.{checkpoint_name}"
        for name, checkpoint_name in INDEXER_CHECKPOINT_NAMES.items()
    },
}

# Gathered fc1 weights are [gate; up] (the gather undoes the per-rank GLU interleave).
_GATE_UP_MODULES = {
    "mlp.linear_fc1.weight": "mlp",
    "mlp.shared_experts.linear_fc1.weight": "share_expert",
}


def convert_step5_to_hf(args, name, param):
    if name in _DIRECT_NAMES:
        return [(_DIRECT_NAMES[name], param)]
    match = _LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = match.groups()
    prefix = f"model.layers.{layer_idx}"

    if expert := _ROUTED_EXPERT.fullmatch(rest):
        projection, expert_idx = expert.groups()
        expert_prefix = f"{prefix}.moe.experts.{expert_idx}"
        if projection == "linear_fc1":
            gate, up = param.chunk(2, dim=0)
            return [(f"{expert_prefix}.gate_proj.weight", gate), (f"{expert_prefix}.up_proj.weight", up)]
        return [(f"{expert_prefix}.down_proj.weight", param)]
    if rest in _LAYER_NAMES:
        return [(f"{prefix}.{_LAYER_NAMES[rest]}", param)]
    if rest in _GATE_UP_MODULES:
        module = _GATE_UP_MODULES[rest]
        gate, up = param.chunk(2, dim=0)
        return [(f"{prefix}.{module}.gate_proj.weight", gate), (f"{prefix}.{module}.up_proj.weight", up)]
    if rest == "self_attention.linear_qkv.weight":
        query, key, value, gate = split_qkv_and_gate(
            param,
            num_query_groups=args.num_query_groups,
            num_attention_heads=args.num_attention_heads,
            head_dim=args.kv_channels,
            tp_size=args.tensor_model_parallel_size,
        )
        return [
            (f"{prefix}.self_attn.q_proj.weight", query),
            (f"{prefix}.self_attn.k_proj.weight", key),
            (f"{prefix}.self_attn.v_proj.weight", value),
            (f"{prefix}.self_attn.g_proj.weight", gate),
        ]
    raise ValueError(f"Unknown parameter name: {name}")
