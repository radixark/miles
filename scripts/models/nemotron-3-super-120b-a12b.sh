# NVIDIA Nemotron-3-Super-120B-A12B (BF16, MoE nemotron_h = hybrid Mamba + Attention + MoE).
# HF config (verified 2026-05-01):
#   num_hidden_layers=88  hidden_size=4096  num_attention_heads=32  num_key_value_heads=2
#   head_dim=128  intermediate_size=2688  moe_intermediate_size=2688
#   n_routed_experts=512  num_experts_per_tok=22  n_shared_experts=1
#   moe_shared_expert_intermediate_size=5376  routed_scaling_factor=5.0
#   n_group=1  topk_group=1  sigmoid routing + aux-free expert bias
# The AutoBridge path (--megatron-to-hf-mode bridge) + miles NemotronHBridge MoE shim
# (see miles_plugins/megatron_bridge/nemotron_h.py) construct the provider and
# HF<->Megatron mapping_registry at load time. Attention-side structural args go
# in MODEL_ARGS for miles' arg parser.

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 2
   --kv-channels 128
   --num-layers 88
   --hidden-size 4096
   --ffn-hidden-size 2688
   --normalization RMSNorm
   --position-embedding-type none
   --vocab-size 131072
   --make-vocab-size-divisible-by 128
   --untie-embeddings-and-output-weights

   # MoE specifics
   --num-experts 512
   --moe-router-topk 22
   --moe-ffn-hidden-size 2688
   --moe-shared-expert-intermediate-size 5376
   # Super-120B bottlenecks expert input/output through a 1024-dim latent.
   # Routed experts run on moe_latent_size, NOT hidden_size, with two extra
   # fc1/fc2 latent projections per MoE layer. The miles NemotronH bridge
   # surfaces this from HF config; the CLI arg keeps Megatron's parser happy.
   --moe-latent-size 1024
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-grouped-gemm
   --moe-router-dtype fp32
   # Routing: HF config has n_group=1 (MoE groups), topk_group=1,
   # routed_scaling_factor=5.0. With n_group=1, group-limited routing is a
   # no-op (single group of 512). `n_groups=8` in HF is Mamba groups —
   # unrelated to MoE.
   --moe-router-num-groups 1
   --moe-router-group-topk 1
   --moe-router-topk-scaling-factor 5.0
   --moe-router-pre-softmax
   # Match nano-30b-a3b (known-working MoE RL on nemotron_h) settings.
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-bias-update-rate 0
   --moe-aux-loss-coeff 0
)
