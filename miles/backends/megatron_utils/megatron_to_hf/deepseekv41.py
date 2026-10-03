import re


from .deepseekv4 import _packed_alphas


def convert_deepseekv41_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("embed.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("norm.weight", param)]
    if name.startswith("module.module.decoder.hc_head_"):
        return []

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = match.groups()
    prefix = f"layers.{layer_idx}"

    if rest == "self_attention_hyper_connection.mapping_proj.weight":
        return [(f"{prefix}.hc_attn_fn", param)]
    if rest == "self_attention_hyper_connection.bias":
        return [(f"{prefix}.hc_attn_base", param)]
    if rest.startswith("self_attention_hyper_connection.alpha_"):
        packed = _packed_alphas(name, param)
        return [] if packed is None else [(f"{prefix}.hc_attn_scale", packed)]
    if rest == "mlp_hyper_connection.mapping_proj.weight":
        return [(f"{prefix}.hc_ffn_fn", param)]
    if rest == "mlp_hyper_connection.bias":
        return [(f"{prefix}.hc_ffn_base", param)]
    if rest.startswith("mlp_hyper_connection.alpha_"):
        packed = _packed_alphas(name, param)
        return [] if packed is None else [(f"{prefix}.hc_ffn_scale", packed)]

    if rest in ("v41_engram.table", "v41_engram.table_scale"):
        return []
    if rest == "v41_engram.linear_wkv.weight":
        return [(f"{prefix}.engram.wkv.weight", param)]
    if rest == "v41_engram.q_weight":
        return [(f"{prefix}.engram.q_weight", param)]
    if rest == "v41_engram.k_weight":
        return [(f"{prefix}.engram.k_weight", param)]

    expert = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    if expert:
        proj, expert_idx = expert.groups()
        if proj == "linear_fc1":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.ffn.experts.{expert_idx}.w1.weight", gate_weight),
                (f"{prefix}.ffn.experts.{expert_idx}.w3.weight", up_weight),
            ]
        if proj == "linear_fc2":
            return [(f"{prefix}.ffn.experts.{expert_idx}.w2.weight", param)]
        raise ValueError(f"Unknown expert parameter name: {name}")

    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{prefix}.ffn.shared_experts.w1.weight", gate_weight),
            (f"{prefix}.ffn.shared_experts.w3.weight", up_weight),
        ]
    if rest == "mlp.shared_experts.linear_fc2.weight":
        return [(f"{prefix}.ffn.shared_experts.w2.weight", param)]

    direct_names = {
        "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
        "self_attention.q_layernorm.weight": "attn.q_norm.weight",
        "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
        "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
        "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
        "self_attention.linear_o_group_proj": "attn.wo_a.weight",
        "self_attention.linear_proj.weight": "attn.wo_b.weight",
        "self_attention.core_attention.attn_sink": "attn.attn_sink",
        "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
        "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
        "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
        "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
        "self_attention.core_attention.indexer.linear_wk.weight": "attn.indexer.wk.weight",
        "self_attention.core_attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "pre_mlp_layernorm.weight": "ffn_norm.weight",
        "mlp.router.weight": "ffn.gate.weight",
        "mlp.router.expert_bias": "ffn.gate.bias",
    }
    if rest in direct_names:
        return [(f"{prefix}.{direct_names[rest]}", param)]
    raise ValueError(f"Unknown parameter name: {name}")
