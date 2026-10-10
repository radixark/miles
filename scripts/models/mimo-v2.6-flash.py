# XiaomiMiMo MiMo-V2.6-Flash-RL text decoder. Train through Megatron-Bridge
# (`--megatron-to-hf-mode bridge`) on the BF16 checkpoint written by
# tools/convert_mimo_v2_to_bf16.py: the bridge (miles_plugins/megatron_bridge/mimo_v2.py) supplies the
# per-layer SWA/global attention, sink, V head dim and RoPE bases, which these flags cannot express.
# The flags mirror the global-attention shape and the MoE layout so Miles and Megatron size the model
# and its pipeline split the same way the bridge builds it.
from model_args_utils import moe_layer_freq

NHIDDEN = 4096
FFN_HIDDEN = 16384
NHEADS = 64
GLOBAL_KV_HEADS = 4
QK_HEAD_DIM = 192
MOE_ROUTED_EXPERTS = 256
MOE_ACTIVE_ROUTED_EXPERTS = 8
MOE_FFN_HIDDEN = 2048
N_LAYERS = 48
FIRST_K_DENSE_REPLACE = 1


def model_args(nlayers: int = N_LAYERS) -> str:
    return (
        f"--num-layers {nlayers} "
        f"--hidden-size {NHIDDEN} "
        f"--ffn-hidden-size {FFN_HIDDEN} "
        f"--num-attention-heads {NHEADS} "
        "--group-query-attention "
        f"--num-query-groups {GLOBAL_KV_HEADS} "
        f"--kv-channels {QK_HEAD_DIM} "
        "--position-embedding-type rope "
        "--rotary-percent 0.334 "
        "--rotary-base 10000000 "
        "--no-rope-fusion "
        "--max-position-embeddings 1048576 "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--swiglu "
        "--disable-bias-linear "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 152576 "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        f"--num-experts {MOE_ROUTED_EXPERTS} "
        f"--moe-router-topk {MOE_ACTIVE_ROUTED_EXPERTS} "
        f"--moe-ffn-hidden-size {MOE_FFN_HIDDEN} "
        "--moe-router-score-function sigmoid "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-load-balancing-type none "
        "--moe-aux-loss-coeff 0 "
        "--moe-router-dtype fp32 "
        "--moe-grouped-gemm "
    )
