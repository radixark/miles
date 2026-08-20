from types import SimpleNamespace


def calculate_embedding_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size


def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    return 2 * seqlen * hidden_size * vocab_size


def calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups):
    if args.q_lora_rank is None:
        q_flops = 2 * seqlen * hidden_size * num_attention_heads * args.kv_channels
    else:
        q_flops = (
            2
            * seqlen
            * args.q_lora_rank
            * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
        )
    if args.kv_lora_rank is None:
        kv_flops = 2 * 2 * seqlen * hidden_size * num_query_groups * args.kv_channels
    else:
        kv_flops = (
            2
            * seqlen
            * (
                args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim))
                + args.hidden_size * args.qk_pos_emb_head_dim
            )
        )

    return q_flops + kv_flops


def calculate_attention_flops(args, seqlen, num_attention_heads):
    # QK^T with causal
    if args.qk_pos_emb_head_dim:
        flops = 2 * num_attention_heads * seqlen * seqlen * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
    else:
        flops = 2 * num_attention_heads * seqlen * seqlen * args.kv_channels / 2
    # A*V
    if args.v_head_dim:
        flops += num_attention_heads * seqlen * seqlen * args.v_head_dim
    else:
        flops += num_attention_heads * seqlen * seqlen * args.kv_channels
    return flops


def calculate_output_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size * hidden_size


def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3


def calculate_layer_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size):
    return (
        calculate_qkv_projection_flops(args, seqlen, hidden_size, num_attention_heads, num_query_groups)
        + calculate_attention_flops(args, seqlen, num_attention_heads)
        + calculate_output_flops(seqlen, hidden_size)
        + calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size)
    )


def calculate_fwd_flops(
    seqlens,
    args,
):
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    vocab_size = args.vocab_size

    total_flops = 0

    dense_ffn = args.ffn_hidden_size
    if args.num_experts is None:
        num_dense_layers = args.num_layers
        num_moe_layers = 0
    else:
        shared_expert_ffn = getattr(args, "moe_shared_expert_intermediate_size", None)
        if shared_expert_ffn is None:
            shared_expert_ffn = 0

        moe_ffn = args.moe_ffn_hidden_size * args.moe_router_topk + shared_expert_ffn
        if hasattr(args, "moe_layer_freq"):
            if isinstance(args.moe_layer_freq, list):
                num_dense_layers = sum(1 for freq in args.moe_layer_freq if freq == 0)
                num_moe_layers = sum(1 for freq in args.moe_layer_freq if freq > 0)
            else:
                num_dense_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq != 0)
                num_moe_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq == 0)
        else:
            num_dense_layers = 0
            num_moe_layers = args.num_layers

    for seqlen in seqlens:
        if num_dense_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    dense_ffn,
                )
                * num_dense_layers
            )

        if num_moe_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    args,
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    moe_ffn,
                )
                * num_moe_layers
            )

        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)

    return total_flops


def fwd_tflops_per_gpu(seqlens, args, world_size):
    return calculate_fwd_flops(seqlens, args) / world_size / 1e12


def _first(config, *names, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _moe_layer_pattern(config, num_layers, num_experts):
    if num_experts is None:
        return None
    first_dense = getattr(config, "first_k_dense_replace", None)
    if first_dense is not None:
        return [0 if i < first_dense else 1 for i in range(num_layers)]
    mlp_only = set(getattr(config, "mlp_only_layers", None) or ())
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return [0 if i in mlp_only or (i + 1) % step != 0 else 1 for i in range(num_layers)]


def flops_args_from_hf_config(config):
    getter = getattr(config, "get_text_config", None)
    config = (getter() if callable(getter) else getattr(config, "text_config", None)) or config

    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_layers = _first(config, "num_hidden_layers", "num_layers")
    assert num_layers is not None, f"no layer count on {type(config).__name__}; cannot size the FLOPs model"
    num_experts = _first(config, "n_routed_experts", "num_experts", "num_local_experts")

    dense_ffn = getattr(config, "intermediate_size", None)

    moe_ffn = getattr(config, "moe_intermediate_size", None)
    if num_experts is not None and moe_ffn is None:
        moe_ffn = dense_ffn

    shared_ffn = getattr(config, "shared_expert_intermediate_size", None)
    if shared_ffn is None and getattr(config, "n_shared_experts", None):
        shared_ffn = config.n_shared_experts * moe_ffn

    moe_layer_freq = _moe_layer_pattern(config, num_layers, num_experts)
    needs_dense = moe_layer_freq is None or any(f == 0 for f in moe_layer_freq)
    needs_moe = moe_layer_freq is not None and any(f > 0 for f in moe_layer_freq)
    if (needs_dense and dense_ffn is None) or (needs_moe and moe_ffn is None):
        raise ValueError(
            f"{type(config).__name__} does not expose the FFN widths the FLOPs model needs "
            f"(dense={dense_ffn}, moe={moe_ffn}); it likely nests them under a sub-config"
        )

    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=_first(config, "num_key_value_heads", default=num_attention_heads),
        vocab_size=config.vocab_size,
        num_layers=num_layers,
        ffn_hidden_size=dense_ffn,
        kv_channels=_first(config, "head_dim", default=hidden_size // num_attention_heads),
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=_first(config, "num_experts_per_tok", "moe_topk", default=1),
        moe_shared_expert_intermediate_size=shared_ffn,
        moe_layer_freq=moe_layer_freq,
        q_lora_rank=getattr(config, "q_lora_rank", None),
        kv_lora_rank=getattr(config, "kv_lora_rank", None),
        qk_head_dim=getattr(config, "qk_nope_head_dim", None) or 0,
        qk_pos_emb_head_dim=getattr(config, "qk_rope_head_dim", None) or 0,
        v_head_dim=getattr(config, "v_head_dim", None) or 0,
    )
