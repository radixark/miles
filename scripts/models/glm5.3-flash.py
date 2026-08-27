"""GLM-5.3-Flash: 45 layers (34 KDA linear attention + 11 DSA), 288-expert MoE
with 3 dense lead layers, mHC on every layer. The hyper-connection / kpool
indexer / KDA gate fields have no Megatron CLI flags and are derived from the
checkpoint by the spec (miles_plugins/models/glm5_next/glm5_next.py).
--mtp-num-layers is omitted: MTP is rollout-side EAGLE only, never trained.
--rope-type rope keeps Megatron off the yarn path; both attention paths skip
rope entirely (qk-pos-emb-head-dim 0)."""

from model_args_utils import moe_layer_freq

FIRST_K_DENSE_REPLACE = 3


def model_args(nlayers: int = 45) -> str:
    return (
        "--spec miles_plugins.models.glm5_next.glm5_next get_glm5_next_spec "
        f"--num-layers {nlayers} "
        "--hidden-size 4096 "
        "--ffn-hidden-size 12288 "
        "--num-attention-heads 64 "
        "--multi-latent-attention "
        "--q-lora-rank 1536 "
        "--kv-lora-rank 512 "
        "--qk-head-dim 256 "
        "--qk-pos-emb-head-dim 0 "
        "--v-head-dim 256 "
        "--kv-channels 256 "
        "--qk-layernorm "
        f"--moe-layer-freq {moe_layer_freq(nlayers=nlayers, first_k_dense_replace=FIRST_K_DENSE_REPLACE)} "
        "--num-experts 288 "
        "--moe-router-topk 8 "
        "--moe-router-score-function sigmoid "
        "--moe-router-pre-softmax "
        "--moe-router-enable-expert-bias "
        "--moe-router-bias-update-rate 0 "
        "--moe-router-topk-scaling-factor 2.5 "
        "--moe-ffn-hidden-size 2048 "
        "--moe-shared-expert-intermediate-size 2048 "
        "--moe-router-dtype fp32 "
        "--moe-grouped-gemm "
        "--moe-permute-fusion "
        "--vocab-size 154880 "
        "--rotary-base 800000 "
        "--rope-type rope "
        "--position-embedding-type rope "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-5 "
        "--swiglu "
        "--disable-bias-linear "
        "--untie-embeddings-and-output-weights "
        "--make-vocab-size-divisible-by 16 "
        "--enable-experimental "
    )
