from megatron.core.transformer import TransformerConfig
from .rope import precompute_freqs_cis


def wrapped_precompute_freqs_cis(
    config: TransformerConfig, rope_head_dim: int, base: float, yarn_disabled: bool = False
):
    max_seq_len = 65536

    # yarn_disabled=True → original_seq_len=0, which makes precompute_freqs_cis skip the YaRN
    # correction-range interpolation. Used by 0415 for pure-window (compress_ratio==0) layers.
    original_seq_len = 0 if yarn_disabled else config.original_max_position_embeddings

    inputs = dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )

    assert config.rotary_scaling_factor in (4, 16), f"Unexpected rotary_scaling_factor: {config.rotary_scaling_factor}"
    expected_original = 0 if yarn_disabled else 65536
    assert inputs == dict(
        dim=rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=expected_original,
        base=base,
        factor=config.rotary_scaling_factor,
        beta_fast=32,
        beta_slow=1,
    )

    return precompute_freqs_cis(**inputs)
