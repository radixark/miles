from megatron.core.transformer import TransformerConfig
from .ref_model import precompute_freqs_cis


def wrapped_precompute_freqs_cis(config: TransformerConfig, rope_head_dim: int, base: float):
    max_seq_len = 65536

    inputs = dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=config.original_max_position_embeddings,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )

    assert inputs == dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=65536,
        base=base,
        factor=4,
        beta_fast=32,
        beta_slow=1,
    )

    return precompute_freqs_cis(**inputs)
