import re

import torch

_TEXT_MODEL = "language_model.model"

_DIRECT = {
    "module.module.embedding.word_embeddings.weight": f"{_TEXT_MODEL}.embed_tokens.weight",
    "module.module.output_layer.weight": "language_model.lm_head.weight",
    "module.module.decoder.final_layernorm.weight": f"{_TEXT_MODEL}.norm.weight",
}

_LAYER_RENAMES = {
    "input_layernorm.weight": "input_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    "self_attention.indexer.index_q_proj.weight": "self_attn.index_q_proj.weight",
    "self_attention.indexer.index_k_proj.weight": "self_attn.index_k_proj.weight",
    "self_attention.indexer.index_q_norm": "self_attn.index_q_norm.weight",
    "self_attention.indexer.index_k_norm": "self_attn.index_k_norm.weight",
    "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    "mlp.router.weight": "block_sparse_moe.gate.weight",
    "mlp.router.expert_bias": "block_sparse_moe.e_score_correction_bias",
    "mlp.shared_experts.linear_fc2.weight": "block_sparse_moe.shared_experts.down_proj.weight",
}


def convert_minimax_m3_to_hf(args, name, param):
    if name in _DIRECT:
        return [(_DIRECT[name], param)]

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = match.groups()
    prefix = f"{_TEXT_MODEL}.layers.{layer_idx}"

    expert = re.match(r"mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)", rest)
    if expert:
        kind, expert_idx = expert.groups()
        base = f"{prefix}.block_sparse_moe.experts.{expert_idx}"
        if kind == "linear_fc1":
            gate, up = param.chunk(2, dim=0)
            return [(f"{base}.w1.weight", gate), (f"{base}.w3.weight", up)]
        return [(f"{base}.w2.weight", param)]

    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{prefix}.block_sparse_moe.shared_experts.gate_proj.weight", gate),
            (f"{prefix}.block_sparse_moe.shared_experts.up_proj.weight", up),
        ]
    if rest == "mlp.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [(f"{prefix}.mlp.gate_proj.weight", gate), (f"{prefix}.mlp.up_proj.weight", up)]
    if rest == "self_attention.linear_qkv.weight":
        head_dim = args.kv_channels
        per_group = args.num_attention_heads // args.num_query_groups
        param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
        q, k, v = torch.split(param, [per_group, 1, 1], dim=1)
        return [
            (f"{prefix}.self_attn.q_proj.weight", q.reshape(-1, args.hidden_size)),
            (f"{prefix}.self_attn.k_proj.weight", k.reshape(-1, args.hidden_size)),
            (f"{prefix}.self_attn.v_proj.weight", v.reshape(-1, args.hidden_size)),
        ]
    if rest in _LAYER_RENAMES:
        return [(f"{prefix}.{_LAYER_RENAMES[rest]}", param)]
    raise ValueError(f"Unknown parameter name: {name}")
