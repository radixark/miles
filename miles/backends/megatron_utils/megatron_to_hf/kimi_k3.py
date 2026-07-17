import re

from ..update_weight.common import AtomicUpdateGroup


_PREFIX = "language_model.model"


def get_kimi_k3_atomic_update_groups():
    return [
        AtomicUpdateGroup(
            "qkv_a_proj",
            (
                ".self_attention.q_a_proj.weight",
                ".self_attention.kv_a_proj_with_mqa.weight",
            ),
        )
    ]


def convert_kimi_k3_to_hf(args, name, param):
    del args

    direct = {
        "module.module.embedding.word_embeddings.weight": f"{_PREFIX}.embed_tokens.weight",
        "module.module.decoder.final_layernorm.weight": f"{_PREFIX}.norm.weight",
        "module.module.output_layer.weight": "language_model.lm_head.weight",
    }
    if name in direct:
        return [(direct[name], param)]

    match = re.fullmatch(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unknown Kimi K3 parameter name: {name}")
    layer_idx, rest = match.groups()
    layer_prefix = f"{_PREFIX}.layers.{layer_idx}"

    expert_match = re.fullmatch(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)", rest)
    if expert_match is not None:
        projection, expert_idx = expert_match.groups()
        expert_prefix = f"{layer_prefix}.block_sparse_moe.experts.{expert_idx}"
        if projection == "linear_fc1":
            gate, up = param.chunk(2, dim=0)
            return [
                (f"{expert_prefix}.w1.weight", gate),
                (f"{expert_prefix}.w3.weight", up),
            ]
        return [(f"{expert_prefix}.w2.weight", param)]

    if rest == "mlp.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        return [
            (f"{layer_prefix}.mlp.gate_proj.weight", gate),
            (f"{layer_prefix}.mlp.up_proj.weight", up),
        ]

    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate, up = param.chunk(2, dim=0)
        shared_prefix = f"{layer_prefix}.block_sparse_moe.shared_experts"
        return [
            (f"{shared_prefix}.gate_proj.weight", gate),
            (f"{shared_prefix}.up_proj.weight", up),
        ]

    output_mapping = {
        "output_attn_res_norm.weight": f"{_PREFIX}.output_attn_res_norm.weight",
        "output_attn_res_proj.weight": f"{_PREFIX}.output_attn_res_proj.weight",
    }
    if rest in output_mapping:
        return [(output_mapping[rest], param)]

    mapping = {
        "input_layernorm.weight": "input_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention_res_norm.weight": "self_attention_res_norm.weight",
        "self_attention_res_proj.weight": "self_attention_res_proj.weight",
        "mlp_res_norm.weight": "mlp_res_norm.weight",
        "mlp_res_proj.weight": "mlp_res_proj.weight",
        "self_attention.q_proj.weight": "self_attn.q_proj.weight",
        "self_attention.k_proj.weight": "self_attn.k_proj.weight",
        "self_attention.v_proj.weight": "self_attn.v_proj.weight",
        "self_attention.q_conv1d.weight": "self_attn.q_conv1d.weight",
        "self_attention.k_conv1d.weight": "self_attn.k_conv1d.weight",
        "self_attention.v_conv1d.weight": "self_attn.v_conv1d.weight",
        "self_attention.A_log": "self_attn.A_log",
        "self_attention.dt_bias": "self_attn.dt_bias",
        "self_attention.f_a_proj.weight": "self_attn.f_a_proj.weight",
        "self_attention.f_b_proj.weight": "self_attn.f_b_proj.weight",
        "self_attention.b_proj.weight": "self_attn.b_proj.weight",
        "self_attention.g_proj.weight": "self_attn.g_proj.weight",
        "self_attention.o_norm.weight": "self_attn.o_norm.weight",
        "self_attention.o_proj.weight": "self_attn.o_proj.weight",
        "self_attention.q_a_proj.weight": "self_attn.q_a_proj.weight",
        "self_attention.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
        "self_attention.q_b_proj.weight": "self_attn.q_b_proj.weight",
        "self_attention.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "self_attention.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
        "self_attention.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        "mlp.router.weight": "block_sparse_moe.gate.weight",
        "mlp.router.expert_bias": "block_sparse_moe.gate.e_score_correction_bias",
        "mlp.fc1_latent_proj.weight": "block_sparse_moe.routed_expert_down_proj.weight",
        "mlp.routed_expert_norm.weight": "block_sparse_moe.routed_expert_norm.weight",
        "mlp.fc2_latent_proj.weight": "block_sparse_moe.routed_expert_up_proj.weight",
        "mlp.shared_experts.linear_fc2.weight": "block_sparse_moe.shared_experts.down_proj.weight",
    }
    if rest not in mapping:
        raise ValueError(f"Unknown Kimi K3 layer parameter name: {name}")
    return [(f"{layer_prefix}.{mapping[rest]}", param)]
