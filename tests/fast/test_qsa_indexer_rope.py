"""The trainer's QSA indexer RoPE must reproduce sglang's eager neox rotation bit for bit."""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import apply_indexer_rope  # noqa: E402

BASE = 1e7
ROTARY_DIM = 64  # head_dim 256 * partial_rotary_factor 0.25
HEAD_DIM = 256
MAX_POS = 4096


def _megatron_rotary_pos_emb(max_pos: int) -> torch.Tensor:
    """Megatron ``RotaryEmbedding.get_emb`` for rotary_interleaved=False."""
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
    seq = torch.arange(max_pos, dtype=inv_freq.dtype)
    freqs = torch.outer(seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)[:, None, None, :]


def _sglang_reference(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """sglang RotaryEmbedding._compute_cos_sin_cache + QSAIndexer.apply_rope + apply_rotary_emb."""
    inv_freq = 1.0 / (BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float) / ROTARY_DIM))
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    last_dim = cos_sin.size()[-1]
    cos, sin = cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)
    half_rotary_dim = ROTARY_DIM // 2
    cos = cos.reshape(positions.numel(), -1)[:, :half_rotary_dim]
    sin = sin.reshape(positions.numel(), -1)[:, :half_rotary_dim]

    rot = x[..., :ROTARY_DIM]
    cos = cos.unsqueeze(-2).to(rot.dtype)
    sin = sin.unsqueeze(-2).to(rot.dtype)
    x1, x2 = torch.chunk(rot, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([torch.cat((o1, o2), dim=-1), x[..., ROTARY_DIM:]], dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads", [1, 8])
def test_indexer_rope_matches_sglang(dtype, heads):
    g = torch.Generator().manual_seed(heads)
    tokens = 257
    x = torch.randn(tokens, heads, HEAD_DIM, generator=g).to(dtype)
    positions = torch.randint(0, MAX_POS, (tokens,), generator=g)

    got = apply_indexer_rope(x, _megatron_rotary_pos_emb(MAX_POS), positions)
    want = _sglang_reference(x, positions)

    assert got.dtype == dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got[..., ROTARY_DIM:], x[..., ROTARY_DIM:], rtol=0, atol=0)


def test_indexer_rope_rotates_position_zero_to_identity():
    x = torch.randn(4, 2, HEAD_DIM).to(torch.bfloat16)
    got = apply_indexer_rope(x, _megatron_rotary_pos_emb(16), torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(got, x, rtol=0, atol=0)
