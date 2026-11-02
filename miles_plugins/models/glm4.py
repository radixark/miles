from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


def get_glm_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.backend.num_experts,
        moe_grouped_gemm=args.backend.moe_grouped_gemm,
        qk_layernorm=args.backend.qk_layernorm,
        multi_latent_attention=args.backend.multi_latent_attention,
        post_self_attn_layernorm=args.backend.post_self_attn_layernorm,
        post_mlp_layernorm=args.backend.post_mlp_layernorm,
    )
    return transformer_layer_spec
