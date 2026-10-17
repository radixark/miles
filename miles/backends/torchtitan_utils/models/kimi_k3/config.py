"""Read the Kimi K3 architecture from HF metadata; a flavor only labels the run."""


def model_config_kwargs(hf_config) -> dict:
    text = getattr(hf_config, "text_config", None) or hf_config
    if getattr(hf_config, "quantization_config", None) or getattr(text, "quantization_config", None):
        raise ValueError("Kimi K3 needs a BF16 checkpoint; run tools/convert_mxfp4_to_bf16.py first")
    linear = text.linear_attn_config
    supported = {
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "moe_layer_freq": 1,
        "num_expert_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 0,
        "tie_word_embeddings": False,
    }
    for name, expected in supported.items():
        actual = getattr(text, name)
        if actual != expected:
            raise ValueError(f"Unsupported K3 {name}={actual!r}; expected {expected!r}")
    if not linear["use_full_rank_gate"] or linear["gate_lower_bound"] is None:
        raise ValueError("K3 requires full-rank output gates and a KDA gate lower bound")
    if not text.attn_res_block_size or text.attn_res_block_size < 1:
        raise ValueError("K3 requires positive attn_res_block_size")
    if not 0 < text.num_experts_per_token <= text.num_experts:
        raise ValueError("K3 num_experts_per_token must be in [1, num_experts]")
    if linear["num_heads"] > linear["head_dim"]:
        raise ValueError("K3 checkpoint A_log stores head_dim slots; num_heads must fit those slots")
    # Pruned checkpoints retain the full model's layer lists. Only their prefix is active.
    active = set(range(1, text.num_hidden_layers + 1))
    full = active.intersection(linear["full_attn_layers"])
    kda = active.intersection(linear["kda_layers"])
    if full & kda or full | kda != active:
        raise ValueError("K3 full_attn_layers and kda_layers must partition the active layers")
    return dict(
        dim=text.hidden_size,
        vocab_size=text.vocab_size,
        n_layers=text.num_hidden_layers,
        n_heads=linear["num_heads"],
        head_dim=linear["head_dim"],
        mla_heads=text.num_attention_heads,
        full_attn_layers=tuple(sorted(full)),
        q_lora_rank=text.q_lora_rank,
        kv_lora_rank=text.kv_lora_rank,
        qk_nope_head_dim=text.qk_nope_head_dim,
        qk_rope_head_dim=text.qk_rope_head_dim,
        v_head_dim=text.v_head_dim,
        conv_kernel_size=linear["short_conv_kernel_size"],
        gate_lower_bound=linear["gate_lower_bound"],
        norm_eps=text.rms_norm_eps,
        num_experts=text.num_experts,
        top_k=text.num_experts_per_token,
        num_shared_experts=text.num_shared_experts,
        route_scale=text.routed_scaling_factor,
        latent_dim=text.routed_expert_hidden_size,
        dense_layers=text.first_k_dense_replace,
        dense_intermediate=text.intermediate_size,
        moe_intermediate=text.moe_intermediate_size,
        attn_res_block_size=text.attn_res_block_size,
    )
