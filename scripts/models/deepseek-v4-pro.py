import os

from model_args_utils import moe_layer_freq


COMPRESS_RATIOS = "[128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0]"
SWIGLU_LIMIT_ARGS = "--activation-func-clamp-value 10 --no-bias-swiglu-fusion --no-activation-func-clamp-shared-expert"


def model_args(
    nlayers: int | None = None, rotary_scaling_factor: str | None = None, compress_ratios: str = COMPRESS_RATIOS
) -> str:
    nlayers = nlayers if nlayers is not None else int(os.environ.get("MODEL_ARGS_NUM_LAYERS") or 61)
    rotary_scaling_factor = (
        rotary_scaling_factor if rotary_scaling_factor is not None else os.environ.get("ROTARY_SCALING_FACTOR") or "16"
    )
    return (
        "--disable-bias-linear "
        f"--num-layers {nlayers} "
        "--hidden-size 7168 "
        "--ffn-hidden-size 3072 "
        "--num-attention-heads 128 "
        "--normalization RMSNorm "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 129280 "
        "--hidden-dropout 0.0 "
        "--attention-dropout 0.0 "
        # MLA params (reused by V4)
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 512 "
        "--qk-pos-emb-head-dim 64 "
        "--v-head-dim 512 "
        "--qk-layernorm "
        f"--rotary-scaling-factor {rotary_scaling_factor} "
        "--rotary-base 10000 "
        "--original-max-position-embeddings 65536 "
        "--beta-fast 32 "
        "--beta-slow 1 "
        "--attention-softmax-in-fp32 "
        "--no-rope-fusion "
        # MoE
        "--num-experts 384 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=0)} "
        "--moe-ffn-hidden-size 3072 "
        "--moe-router-topk 6 "
        "--moe-shared-expert-intermediate-size 3072 "
        "--moe-router-pre-softmax "
        "--moe-router-score-function sqrtsoftplus "
        "--moe-router-enable-expert-bias "
        "--moe-router-load-balancing-type seq_aux_loss "
        "--moe-token-dispatcher-type alltoall "
        "--moe-aux-loss-coeff 0 "
        "--moe-grouped-gemm "
        "--moe-router-topk-scaling-factor 2.5 "
        # DSV4 specific
        "--num-residual-streams 4 "
        "--mhc-sinkhorn-iterations 20 "
        f"--csa-compress-ratios {compress_ratios} "
        "--csa-compress-rotary-base 160000 "
        "--o-groups 16 "
        "--o-lora-rank 1024 "
        "--moe-n-hash-layers 3 "
        "--csa-window-size 128 "
        # DSA Indexer
        "--dsa-indexer-n-heads 64 "
        "--dsa-indexer-head-dim 128 "
        "--dsa-indexer-topk 1024 "
        # V4 model spec (plugin)
        "--spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec "
        f"{SWIGLU_LIMIT_ARGS} "
    )
