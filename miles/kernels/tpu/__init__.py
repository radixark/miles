# here, not in flash_attention.py, so argument validation reads it without importing jax and torch_tpu
FLASH_ATTENTION_BLOCK = 1024  # fastest fwd+bwd on v7x for 4k-8k packed sequences
