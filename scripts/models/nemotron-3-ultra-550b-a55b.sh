# NVIDIA Nemotron-3-Ultra-550B-A55B (BF16, MoE nemotron_h = hybrid Mamba2 + Attention + MoE).
# HF config:
#   num_hidden_layers=108  hidden_size=8192  num_attention_heads=64  num_key_value_heads=2
#   head_dim=128  intermediate_size=5120  moe_intermediate_size=5120
#   n_routed_experts=512  num_experts_per_tok=22  n_shared_experts=1
#   moe_shared_expert_intermediate_size=10240  routed_scaling_factor=5.0
#   moe_latent_size=2048  n_group=1  topk_group=1  sigmoid routing + aux-free expert bias
#   num_nextn_predict_layers=1 (MTP head)   mamba n_groups=8
# Same AutoBridge path as Super-120B (--megatron-to-hf-mode bridge) + miles
# NemotronHBridge MoE/latent shim (miles_plugins/megatron_bridge/nemotron_h.py).
# NOTE: Mamba n_groups=8 forces attention/mamba tensor-parallel <= 8 (n_groups % tp == 0).

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 64
   --num-query-groups 2
   --kv-channels 128
   --num-layers 108
   --hidden-size 8192
   --ffn-hidden-size 5120
   --normalization RMSNorm
   --position-embedding-type none
   --vocab-size 131072
   --make-vocab-size-divisible-by 128
   --untie-embeddings-and-output-weights

   # MoE specifics
   --num-experts 512
   --moe-router-topk 22
   --moe-ffn-hidden-size 5120
   --moe-shared-expert-intermediate-size 10240
   # Ultra-550B bottlenecks expert input/output through a 2048-dim latent
   # (routed experts run on moe_latent_size, not hidden_size; two extra fc1/fc2
   # latent projections per MoE layer). Surfaced from HF config by the miles
   # NemotronH bridge; the CLI arg keeps Megatron's parser happy.
   --moe-latent-size 2048
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-grouped-gemm
   --moe-router-dtype fp32
   # n_group=1 (MoE groups) -> group-limited routing is a no-op (single group of
   # 512). HF n_groups=8 is the Mamba groups, unrelated to MoE.
   --moe-router-num-groups 1
   --moe-router-group-topk 1
   --moe-router-topk-scaling-factor 5.0
   --moe-router-pre-softmax
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-bias-update-rate 0
   --moe-aux-loss-coeff 0
)
