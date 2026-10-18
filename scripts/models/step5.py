"""StepFun Step-5-Preview text decoder.

92 layers, hidden 4096, GQA with 64 query and 4 KV heads of 192 and a head-wise sigmoid output
gate. Three layers in four attend over a 512-token sliding window, every fourth is full (DSA)
attention; layers 3-90 are MoE with 352 routed experts (top-8, sigmoid, x3.0) plus a shared
expert. ``layers`` names original layer indices, so a sliced checkpoint keeps each layer's
attention type and MLP. Per-layer RoPE comes from the spec, keyed by attention type.
"""

NUM_LAYERS = 92
MOE_LAYERS = range(3, 91)


def _per_layer(flags) -> str:
    return "[" + ",".join(str(int(flag)) for flag in flags) + "]"


def model_args(layers: tuple[int, ...] = tuple(range(NUM_LAYERS))) -> str:
    return (
        "--spec miles_plugins.models.step5 get_step5_spec "
        f"--num-layers {len(layers)} "
        "--hidden-size 4096 "
        "--ffn-hidden-size 13824 "
        "--num-attention-heads 64 "
        "--group-query-attention "
        "--num-query-groups 4 "
        "--kv-channels 192 "
        "--head-wise-attn-gate "
        "--qk-layernorm "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--fp32-residual-connection "
        "--swiglu "
        "--disable-bias-linear "
        "--hidden-dropout 0 "
        "--attention-dropout 0 "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 128896 "
        # 128896 = 32 * 4028, so TP 1, 2 and 4 all run without vocab padding.
        "--make-vocab-size-divisible-by 32 "
        # attention: a 512-token window is the query plus 511 before it
        "--position-embedding-type rope "
        "--rotary-base 10000 "
        "--window-size 511,0 "
        f"--window-attn-skip-freq {_per_layer(layer % 4 != 3 for layer in layers)} "
        # moe
        "--num-experts 352 "
        f"--moe-layer-freq {_per_layer(layer in MOE_LAYERS for layer in layers)} "
        "--moe-ffn-hidden-size 1536 "
        "--moe-shared-expert-intermediate-size 1536 "
        "--moe-router-topk 8 "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type none "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-topk-scaling-factor 3.0 "
        "--moe-router-dtype fp32 "
        "--moe-token-dispatcher-type alltoall "
        "--moe-grouped-gemm "
        "--moe-combine-in-fp32 "
    )
