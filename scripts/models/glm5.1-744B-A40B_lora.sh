MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
N_DENSE_LAYERS=3
N_MOE_LAYERS=75
NHEADS=64

# GLM-5.1 744B-A40B (zai-org/GLM-5.1, glm_moe_dsa). MODEL_ARGS are identical to
# glm5-744B-A40B.sh (the full-FT GLM-5/5.1 registry): GLM-5.1 uses rotary-base 1e6;
# the ONLY architecture difference vs the glm5.2-744B-A40B* registries is
# --rotary-base (5.2 uses 8e6) -- the DSA cross-layer index-sharing schedule is read
# from the HF config by the shared glm5 provider, not from these args.
MODEL_ARGS=(
   --spec "miles_plugins.models.glm5.glm5" "get_glm5_spec"
    --moe-layer-freq "[0]*${N_DENSE_LAYERS}+[1]*${N_MOE_LAYERS}"
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --make-vocab-size-divisible-by 16
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 2048
    --kv-lora-rank 512
    --qk-head-dim 192
    --v-head-dim 256
    --kv-channels 192
    --qk-pos-emb-head-dim 64
    --vocab-size 154880
    --rotary-base 1000000
    --enable-experimental
)

# ─────────────────────────────────────────────────────────────────────────────
# LoRA defaults (GLM-5.1 LoRA registry, consumed by scripts/run_glm5_1_744b_a40b_lora.py)
#
# How this file is consumed: miles.utils.external_utils.command_utils.execute_train()
# does `source scripts/models/<megatron_model_type>.sh` and prepends ONLY the
# ${MODEL_ARGS[@]} array to the train.py command line, BEFORE the args emitted by the
# runner .py; argparse takes the LAST occurrence, so anything the runner emits always
# wins over this file. (U.convert_checkpoint sources the same array, but the bridge-LoRA
# runner never converts a megatron checkpoint -- it loads the HF checkpoint directly via
# --megatron-to-hf-mode bridge.)
#
# Division of labor -- MODEL_ARGS above carries the model ARCHITECTURE only (identical to
# glm5-744B-A40B.sh, the full-FT registry); every LoRA / run-mode flag is emitted by the
# runner .py and is deliberately NOT duplicated here, because (a) value flags set here
# would be silently overridden anyway (last-occurrence-wins) and (b) boolean flags set
# here could NOT be turned back off by the runner's knobs (e.g. KEEP_MOE_LORA=0 must be
# able to drop --experts-shared-outer-loras). For reference, the runner emits:
#   * bridge + DSA backend:  --megatron-to-hf-mode bridge --dsa-attention-backend {glm-native|megatron-bridge-native}
#                            (+ the matching --qkv-format thd|bshd and --micro-batch-size 1)
#   * adapter shape:         --lora-rank 16 --lora-alpha 32 --lora-dropout 0.0
#                            --target-modules "q,k,v,o + MLA q_a/kv_a/q_b/kv_b + gate/up/down"
#                            (DSA indexer wq_b/wk/weights_proj excluded by default)
#   * MoE-expert LoRA pair:  --experts-shared-outer-loras (train side) +
#                            --sglang-lora-use-virtual-experts (serve side); both gated by
#                            KEEP_MOE_LORA=1 (default)
#   * LoRA misc:             --no-gradient-accumulation-fusion; sglang serving:
#                            --sglang-max-lora-rank <rank> --sglang-lora-backend triton
#   * opt-in via --extra-args: --lora-base-cpu-backup (host-RAM base mirror; watch pod RAM)
#
# NOTE on --spec (above): in the LoRA-via-bridge path (--megatron-to-hf-mode bridge +
# --lora-rank>0) the model is built by the Megatron-Bridge provider + miles' "dsa"
# experimental-attention-variant monkey-patch (bridge_lora_helpers.py), NOT get_glm5_spec --
# args.spec is never imported/invoked there. It is kept for parity with the other glm5
# registry entries (non-bridge / full-spec paths still use it).
# ─────────────────────────────────────────────────────────────────────────────
