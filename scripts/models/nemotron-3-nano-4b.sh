# NVIDIA Nemotron-3-Nano-4B (BF16, dense `nemotron_h` = hybrid Mamba + Attention).
# HF config (verified 2026-04-21):
#   num_hidden_layers=42  hidden_size=3136  num_attention_heads=40  num_key_value_heads=8
#   vocab_size=131072     max_position_embeddings=262144   no RoPE   squared-relu FFN
# The AutoBridge path (--megatron-to-hf-mode bridge) constructs the full Megatron
# provider from the HF config.json at load time, including all Mamba-specific
# fields (mamba_num_heads, mamba_state_dim, hybrid_override_pattern, etc.), so we
# only keep the attention-side structural args here for miles' arg parser.

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 40
   --num-query-groups 8
   --kv-channels 128
   --num-layers 42
   --hidden-size 3136
   --ffn-hidden-size 12544
   --normalization RMSNorm
   --position-embedding-type none
   --vocab-size 131072
   --make-vocab-size-divisible-by 128
   --untie-embeddings-and-output-weights
)
