def model_args(num_layers: int = 48, num_experts: int = 384) -> str:
    # Inkling2 (48 layers, 384 routed experts, top-8). The model provider builds the
    # TransformerConfig from the checkpoint's text_config; these flags keep megatron's
    # argument validation and checkpoint metadata consistent with it.
    return (
        "--custom-model-provider-path miles_plugins.models.inkling2.model.inkling2_model_provider "
        "--disable-bias-linear "
        f"--num-layers {num_layers} "
        "--hidden-size 4608 "
        "--ffn-hidden-size 2304 "
        "--num-attention-heads 48 "
        "--group-query-attention "
        "--num-query-groups 8 "
        "--kv-channels 128 "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 201024 "
        "--position-embedding-type none "
        "--max-position-embeddings 1048576 "
        # MoE
        f"--num-experts {num_experts} "
        "--moe-ffn-hidden-size 2304 "
        "--moe-router-topk 8 "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-dtype fp32 "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-grouped-gemm "
    )
