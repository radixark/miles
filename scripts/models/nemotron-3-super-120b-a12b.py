def model_args() -> str:
    return (
        "--disable-bias-linear "
        "--group-query-attention "
        "--num-attention-heads 32 "
        "--num-query-groups 2 "
        "--kv-channels 128 "
        "--num-layers 88 "
        "--hidden-size 4096 "
        "--ffn-hidden-size 2688 "
        "--normalization RMSNorm "
        "--position-embedding-type none "
        "--vocab-size 131072 "
        "--make-vocab-size-divisible-by 128 "
        "--untie-embeddings-and-output-weights "
        # MoE specifics
        "--num-experts 512 "
        "--moe-router-topk 22 "
        "--moe-ffn-hidden-size 2688 "
        "--moe-shared-expert-intermediate-size 5376 "
        # Super-120B bottlenecks expert input/output through a 1024-dim latent.
        # Routed experts run on moe_latent_size, NOT hidden_size, with two extra
        # fc1/fc2 latent projections per MoE layer. The miles NemotronH bridge
        # surfaces this from HF config; the CLI arg keeps Megatron's parser happy.
        "--moe-latent-size 1024 "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-grouped-gemm "
        "--moe-router-dtype fp32 "
        # Routing: HF config has n_group=1 (MoE groups), topk_group=1,
        # routed_scaling_factor=5.0. With n_group=1, group-limited routing is a
        # no-op (single group of 512). `n_groups=8` in HF is Mamba groups —
        # unrelated to MoE.
        "--moe-router-num-groups 1 "
        "--moe-router-group-topk 1 "
        "--moe-router-topk-scaling-factor 5.0 "
        "--moe-router-pre-softmax "
        # Match nano-30b-a3b (known-working MoE RL on nemotron_h) settings.
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-router-bias-update-rate 0 "
        "--moe-aux-loss-coeff 0 "
    )
