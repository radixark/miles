"""Qwen's enforced SelfAttention RoPE seam matches stock SGLang CUDA."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytest.importorskip("megatron.core")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from megatron.core.transformer import attention
from sglang.kernels.ops.attention.rope import apply_rope_with_cos_sin_cache_inplace

from miles_plugins.top.rope import enforce_fused_rope


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("rows", [1, 33, 201])
def test_enforced_rope_is_exact(packed, width, rows, monkeypatch):
    from miles_plugins.top import install

    monkeypatch.setattr(attention, "apply_rotary_pos_emb", attention.apply_rotary_pos_emb)
    monkeypatch.setattr(install, "_INSTALLED", {})
    config = SimpleNamespace(apply_rope_fusion=False, rotary_interleaved=False, mrope_section=None)
    enforce_fused_rope(SimpleNamespace(), config)
    torch.manual_seed(71)
    x = torch.randn(rows, 4, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.arange(rows, device="cuda")
    inv_freq = (1.0 / (1000000 ** (torch.arange(0, width, 2).float() / width))).cuda()
    angles = torch.outer(positions.float(), inv_freq)
    freqs = torch.cat((angles, angles), dim=-1)[:, None, None, :]
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32) if packed else None
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    got = attention.apply_rotary_pos_emb(
        x if packed else x.unsqueeze(1), freqs, config, cu_seqlens=cu, cp_group=group
    ).reshape_as(x)
    rollout = x.detach().clone()
    key = torch.randn(rows, 1, width, device="cuda", dtype=torch.bfloat16)
    apply_rope_with_cos_sin_cache_inplace(rollout, key, cache, positions, is_neox=True)
    assert torch.equal(got, rollout)
    seed = torch.randn_like(got)
    got.backward(seed)
    reference = x.detach().double().requires_grad_(True)
    cos, sin = cache.double().chunk(2, dim=-1)
    even, odd = reference.chunk(2, dim=-1)
    output = torch.cat((even * cos[:, None] - odd * sin[:, None], even * sin[:, None] + odd * cos[:, None]), dim=-1)
    output.backward(seed.double())
    torch.testing.assert_close(x.grad.double(), reference.grad, rtol=0.005, atol=0.002)
