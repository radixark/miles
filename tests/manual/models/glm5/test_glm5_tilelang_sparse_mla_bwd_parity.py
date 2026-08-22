"""Numerical parity between the GLM-5 DSA sparse-MLA backward kernels.

Guards the equivalence of:
  * ``tilelang_sparse_mla_bwd``             -- the vendored reference kernel.
  * ``tilelang_sparse_mla_bwd_opt`` -- the GEMM-operand-reuse / split-D
    optimized variant that the production ``sparse_mla.py`` autograd path now uses.

Both share the same public API and math; dQ/dKV stay strictly fp32-accumulated, so
the two agree to bf16-cast noise (cosine ~1.0). Mirrors Automodel's
``tests/unit_tests/models/glm_moe_dsa/test_glm_moe_dsa_tilelang_bwd_parity.py``.

GPU + tilelang required; skipped otherwise. Run:
  pixi run -e nemo python -m pytest tests/manual/models/glm5/test_glm5_tilelang_sparse_mla_bwd_parity.py -v -s
"""

import pytest
import torch

try:
    import tilelang  # noqa: F401
except ImportError:
    tilelang = None

if tilelang is not None:
    from miles_plugins.models.glm5.ops.tilelang_sparse_mla_bwd import sparse_mla_bwd as sparse_mla_bwd_ref
    from miles_plugins.models.glm5.ops.tilelang_sparse_mla_bwd_opt import sparse_mla_bwd as sparse_mla_bwd_opt
    from miles_plugins.models.glm5.ops.tilelang_sparse_mla_fwd import sparse_mla_fwd_interface
else:
    sparse_mla_bwd_ref = None
    sparse_mla_bwd_opt = None
    sparse_mla_fwd_interface = None


# GLM-5 DSA sparse-MLA latent geometry (fixed by the model / kernel asserts).
KV_LORA = 512
ROPE = 64
HEAD_DIM = KV_LORA + ROPE  # 576 -- the score/dQ/dKV width; DV=512 is the dP/dV width


def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def requires_tilelang():
    return pytest.mark.skipif(tilelang is None, reason="tilelang not installed")


def _causal_topk_indices(num_tokens, topk, device):
    """Causal top-k gather indices ``[S, topk]`` int32: token ``t`` selects
    ``min(t+1, topk)`` random keys from ``[0, t]``; the tail stays ``-1`` (masked)."""
    idx = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        n = min(t + 1, topk)
        idx[t, :n] = torch.randperm(t + 1, device=device)[:n].to(torch.int32)
    return idx


# (seqlen, heads, topk) -- topk must be a multiple of 64; heads must be a multiple
# of 64 when > 64 (forward kernel constraint used to build a consistent softmax state).
PARITY_CONFIGS = [
    (256, 16, 64),
    (512, 16, 128),
    (512, 64, 256),
    (1024, 64, 512),
]
PARITY_IDS = [f"s{s}_h{h}_top{tk}" for s, h, tk in PARITY_CONFIGS]


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("seqlen,heads,topk", PARITY_CONFIGS, ids=PARITY_IDS)
def test_sparse_mla_bwd_opt_matches_reference(seqlen, heads, topk):
    """The fusion backward kernel matches the vendored reference (cosine > 0.999)."""
    torch.manual_seed(0)
    dev = "cuda"
    scale = HEAD_DIM**-0.5

    q = torch.randn(seqlen, heads, HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(seqlen, 1, HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    indices = _causal_topk_indices(seqlen, topk, dev).view(seqlen, 1, topk).contiguous()
    o, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale)
    o, lse = o.contiguous(), lse.contiguous()
    do = torch.randn_like(o)

    dq_ref, dkv_ref = sparse_mla_bwd_ref(q, kv, o, do, indices, lse, sm_scale=scale)
    dq_opt, dkv_opt = sparse_mla_bwd_opt(q, kv, o, do, indices, lse, sm_scale=scale)

    assert dq_opt.shape == dq_ref.shape
    assert dkv_opt.shape == dkv_ref.shape

    cos = torch.nn.functional.cosine_similarity
    dq_cos = cos(dq_opt.float().flatten(), dq_ref.float().flatten(), dim=0).item()
    dkv_cos = cos(dkv_opt.float().flatten(), dkv_ref.float().flatten(), dim=0).item()
    print(f"\n[BWD-PARITY] s={seqlen} h={heads} topk={topk}  dq_cos={dq_cos:.6f} dkv_cos={dkv_cos:.6f}")

    assert dq_cos > 0.999, f"dq parity too low: {dq_cos:.6f}"
    assert dkv_cos > 0.999, f"dkv parity too low: {dkv_cos:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
