import math
from functools import lru_cache

import torch
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import get_args


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precompute the complex rotary frequencies for RoPE, with optional YaRN smoothing.

    When ``original_seq_len > 0``, applies YaRN factor rescaling interpolated
    by a linear ramp between ``beta_fast`` and ``beta_slow``. Otherwise the
    base frequencies are used verbatim.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE in-place to the last dim of ``x``.

    ``x`` has shape ``[..., dim]`` where ``dim`` is even; the last-dim pairs are
    treated as complex numbers multiplied by ``freqs_cis``. When ``inverse=True``
    the conjugate rotation is applied (used for the indexer's inverse rope).
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def wrapped_precompute_freqs_cis(
    config: TransformerConfig, rope_head_dim: int, base: float, yarn_disabled: bool = False, min_seq_len: int = 0
):
    # The freqs_cis table must cover the global sequence length that
    # get_freqs_cis_for_cp slices (positions up to cp_size * seqlen_local). In
    # packed-THD dynamic-batch mode the real maximum is max_tokens_per_gpu *
    # cp_size (args.seq_length does not reflect the packed length), so derive it
    # from the run config rather than hardcoding. min_seq_len lets the forward
    # pass grow the table on demand (see ensure_freqs_cis).
    args = get_args()
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    if getattr(args, "max_tokens_per_gpu", None):
        budget = args.max_tokens_per_gpu * cp_size
    else:
        budget = args.seq_length
    max_seq_len = max(budget, config.original_max_position_embeddings, min_seq_len)

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


def ensure_freqs_cis(module, config, rope_head_dim, base, yarn_disabled, seqlen_global):
    # A single packed sample may exceed the max_tokens_per_gpu * cp_size budget
    # (while still fitting in memory), so the table built at init can be too
    # short. Grow module.freqs_cis monotonically to cover seqlen_global.
    if seqlen_global > module.freqs_cis.size(0):
        grown = wrapped_precompute_freqs_cis(
            config, rope_head_dim=rope_head_dim, base=base, yarn_disabled=yarn_disabled, min_seq_len=seqlen_global
        )
        module.freqs_cis = grown.to(device=module.freqs_cis.device, dtype=module.freqs_cis.dtype)
