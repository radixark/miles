"""Full 62-layer Muse Spark text model; the launcher supplies parallelism and attention backend.

``num_layers`` serves the last-N-layer partials, whose schedules also count from the top.
"""


def model_args(num_layers=62, **kwargs) -> str:
    no_rope_freq = "[" + ",".join(str(int((num_layers - layer - 1) % 4 == 0)) for layer in range(num_layers)) + "]"
    window_attn_skip_freq = (
        "[" + ",".join(str(int((num_layers - layer - 1) % 4 != 0)) for layer in range(num_layers)) + "]"
    )
    # The converted checkpoint stores each layer under its own numbered key.
    moe_layer_freq = "[" + ",".join("1" for _ in range(num_layers)) + "]"
    return (
        "--custom-model-provider-path miles_plugins.models.muse_spark.muse_spark_1x_model_provider "
        "--megatron-to-hf-mode raw "
        f"--num-layers {num_layers} --hidden-size 8192 --ffn-hidden-size 16384 "
        "--num-attention-heads 128 --num-query-groups 16 --kv-channels 64 "
        "--group-query-attention --qk-layernorm --attention-output-gate "
        "--normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --swiglu "
        "--vocab-size 202048 --make-vocab-size-divisible-by 1 --untie-embeddings-and-output-weights "
        "--position-embedding-type rope --rotary-base 500000 --rotary-percent 1.0 --rotary-interleaved "
        f"--no-rope-freq {no_rope_freq} "
        f"--window-size 2047,0 --window-attn-skip-freq {window_attn_skip_freq} "
        "--post-self-attn-layernorm --post-mlp-layernorm "
        "--num-experts 256 --moe-router-topk 8 --moe-ffn-hidden-size 4096 --moe-latent-size 4096 "
        f"--moe-layer-freq {moe_layer_freq} --moe-grouped-gemm --moe-token-dispatcher-type alltoall "
        "--moe-router-score-function sigmoid --moe-router-dtype fp32 --moe-router-use-torch-mm "
        "--moe-router-enable-expert-bias --freeze-e-score-correction-bias --moe-router-topk-scaling-factor 1 "
        "--moe-combine-in-fp32 --moe-deterministic-combine "
        "--expert-tensor-parallel-size 1 --microbatch-group-size-per-virtual-pipeline-stage 1 "
        "--moe-router-load-balancing-type none --moe-aux-loss-coeff 0 "
        "--fp32-residual-connection --accumulate-allreduce-grads-in-fp32 "
        "--attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32 "
        "--no-gradient-accumulation-fusion --no-bias-swiglu-fusion --no-bias-dropout-fusion "
        "--no-masked-softmax-fusion"
    )
