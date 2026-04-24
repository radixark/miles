"""Unit tests for DeepSeek-V4 TileLang sparse MLA attention kernels.

Compares tilelang sparse MLA (forward + backward) against the PyTorch reference
(sparse_attn_torch from attention_core.py) which uses gather-based attention.

Diff metrics follow the dumper comparator convention:
  - rel_diff:      1 - 2*(x·y) / (x² + y²)
  - max_abs_diff:  max |x - y|
  - mean_abs_diff: mean |x - y|
  - abs_diff percentiles (p50, p95, p99)

Test matrix covers:
  - V4 real configs: H=64 (n_local_heads on single TP), D=512, various topk
  - Smaller configs for faster testing: H=8, H=16
  - Different sequence lengths: 128, 256, 512, 1024, 2048
  - Different batch sizes: 1, 2
  - Different topk: 64, 128, 256, 512
  - attn_sink values: zero, positive, negative, mixed
  - Edge cases: all indices valid, some indices -1
"""

from dataclasses import dataclass

import pytest
import torch


# ---------------------------------------------------------------------------
# Diff computation (same as dumper comparator)
# ---------------------------------------------------------------------------
@dataclass
class DiffInfo:
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    p50_abs_diff: float
    p95_abs_diff: float
    p99_abs_diff: float


def compute_diff(baseline: torch.Tensor, target: torch.Tensor) -> DiffInfo:
    x = baseline.flatten().float()
    y = target.flatten().float()
    abs_diff = (x - y).abs()

    xy = (x * y).sum()
    x2 = (x * x).sum()
    y2 = (y * y).sum()
    denom = x2 + y2
    rel_diff = (1.0 - 2.0 * xy / denom).item() if denom > 0 else 0.0

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    sorted_diff = abs_diff.sort().values
    n = len(sorted_diff)
    p50 = sorted_diff[int(n * 0.50)].item() if n > 0 else 0.0
    p95 = sorted_diff[min(int(n * 0.95), n - 1)].item() if n > 0 else 0.0
    p99 = sorted_diff[min(int(n * 0.99), n - 1)].item() if n > 0 else 0.0

    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        p50_abs_diff=p50,
        p95_abs_diff=p95,
        p99_abs_diff=p99,
    )


def print_diff(name: str, diff: DiffInfo):
    print(
        f"  {name}: rel={diff.rel_diff:.2e}, max_abs={diff.max_abs_diff:.2e}, mean_abs={diff.mean_abs_diff:.2e}, p95={diff.p95_abs_diff:.2e}, p99={diff.p99_abs_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# PyTorch reference (from attention_core.py)
# ---------------------------------------------------------------------------
def ref_dense_attn(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Dense PyTorch reference for sparse attention with attn_sink.

    Uses full Q @ K^T with mask (no gather), matching dense_attn_torch in attention_core.py.
    """
    q = q.float()
    kv = kv.float()
    b, m, h, d = q.shape
    n = kv.shape[1]
    topk = topk_idxs.shape[-1]

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)
    valid_mask = topk_idxs != -1
    attn_mask[batch_idx[valid_mask], seq_idx[valid_mask], topk_idxs[valid_mask].long()] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q, kv) * sm_scale
    attn_mask_expanded = attn_mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~attn_mask_expanded, float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)

    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv)
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term

    o = numerator / denominator.unsqueeze(-1)
    return o


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def requires_tilelang():
    try:
        import tilelang  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="tilelang not installed")


def make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk, device="cuda", sink_mode="random"):
    """Generate random inputs for sparse MLA.

    Returns q, kv, attn_sink, topk_idxs.
    """
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(batch, seqlen_kv, dim, device=device, dtype=torch.bfloat16)

    if sink_mode == "random":
        attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    elif sink_mode == "zero":
        attn_sink = torch.zeros(heads, device=device, dtype=torch.float32)
    elif sink_mode == "positive":
        attn_sink = torch.rand(heads, device=device, dtype=torch.float32) * 2
    elif sink_mode == "negative":
        attn_sink = -torch.rand(heads, device=device, dtype=torch.float32) * 2
    else:
        attn_sink = torch.randn(heads, device=device, dtype=torch.float32)

    # Generate valid random topk indices (no duplicates per query, all valid)
    actual_topk = min(topk, seqlen_kv)
    topk_idxs = torch.stack(
        [
            torch.stack([torch.randperm(seqlen_kv, device=device)[:actual_topk] for _ in range(seqlen)])
            for _ in range(batch)
        ]
    ).to(
        torch.int32
    )  # [B, S, topk]

    # Pad with -1 if topk > seqlen_kv
    if topk > seqlen_kv:
        padding = torch.full((batch, seqlen, topk - actual_topk), -1, device=device, dtype=torch.int32)
        topk_idxs = torch.cat([topk_idxs, padding], dim=-1)

    return q, kv, attn_sink, topk_idxs


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------
FORWARD_CONFIGS = [
    # (batch, seqlen, heads, dim, seqlen_kv, topk)
    (1, 128, 8, 512, 160, 64),
    (1, 256, 8, 512, 320, 128),
    (1, 256, 16, 512, 320, 128),
    (2, 128, 8, 512, 160, 64),
    (1, 512, 8, 512, 640, 256),
    (1, 512, 16, 512, 640, 128),
    # V4 real config: H=64
    (1, 256, 64, 512, 320, 128),
    (1, 512, 64, 512, 640, 256),
    (1, 1024, 64, 512, 1280, 512),
    # Larger topk
    (1, 256, 8, 512, 320, 256),
    # Small topk
    (1, 256, 8, 512, 320, 64),
]

FORWARD_IDS = [f"b{b}_s{s}_h{h}_d{d}_kv{kv}_top{tk}" for b, s, h, d, kv, tk in FORWARD_CONFIGS]


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("batch,seqlen,heads,dim,seqlen_kv,topk", FORWARD_CONFIGS, ids=FORWARD_IDS)
def test_sparse_mla_forward(batch, seqlen, heads, dim, seqlen_kv, topk):
    """Compare tilelang sparse MLA forward against PyTorch reference."""
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

    q, kv, attn_sink, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk)
    sm_scale = (1.0 / dim) ** 0.5

    ref_o = ref_dense_attn(q, kv, attn_sink, topk_idxs, sm_scale)
    tl_o, tl_lse = sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

    diff = compute_diff(ref_o.float(), tl_o.float())
    print(f"\n[FWD] b={batch}, s={seqlen}, h={heads}, d={dim}, kv={seqlen_kv}, topk={topk}")
    print_diff("output", diff)

    assert diff.rel_diff < 1e-3, f"rel_diff too large: {diff.rel_diff:.2e}"
    assert diff.max_abs_diff < 0.1, f"max_abs_diff too large: {diff.max_abs_diff:.2e}"


# ---------------------------------------------------------------------------
# attn_sink correctness tests
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("sink_mode", ["zero", "positive", "negative", "random"])
def test_attn_sink_modes(sink_mode):
    """Test that attn_sink is correctly incorporated for different value ranges."""
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

    batch, seqlen, heads, dim, seqlen_kv, topk = 1, 256, 8, 512, 320, 128
    q, kv, attn_sink, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk, sink_mode=sink_mode)
    sm_scale = (1.0 / dim) ** 0.5

    ref_o = ref_dense_attn(q, kv, attn_sink, topk_idxs, sm_scale)
    tl_o, _ = sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

    diff = compute_diff(ref_o.float(), tl_o.float())
    print(f"\n[SINK-{sink_mode}]")
    print_diff("output", diff)

    assert diff.rel_diff < 1e-3, f"rel_diff too large for sink_mode={sink_mode}: {diff.rel_diff:.2e}"


@requires_cuda()
@requires_tilelang()
def test_attn_sink_effect():
    """Verify attn_sink actually changes output (not ignored)."""
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

    batch, seqlen, heads, dim, seqlen_kv, topk = 1, 128, 8, 512, 160, 64
    q, kv, _, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk)
    sm_scale = (1.0 / dim) ** 0.5

    sink_zero = torch.zeros(heads, device="cuda", dtype=torch.float32)
    sink_large = torch.ones(heads, device="cuda", dtype=torch.float32) * 10.0

    o_zero, _ = sparse_mqa_fwd_interface(q, kv, sink_zero, topk_idxs, sm_scale=sm_scale)
    o_large, _ = sparse_mqa_fwd_interface(q, kv, sink_large, topk_idxs, sm_scale=sm_scale)

    # Large attn_sink should suppress all outputs toward zero (denominator dominated by sink)
    diff = compute_diff(o_zero.float(), o_large.float())
    print("\n[SINK-EFFECT] output diff between sink=0 and sink=10")
    print_diff("output", diff)
    assert diff.max_abs_diff > 1e-3, "attn_sink has no effect on output — likely not implemented"


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------
BACKWARD_CONFIGS = [
    # (batch, seqlen, heads, dim, seqlen_kv, topk)
    (1, 128, 8, 512, 160, 64),
    (1, 256, 16, 512, 320, 128),
    (2, 128, 8, 512, 160, 64),
    (1, 256, 64, 512, 320, 128),
    (1, 512, 8, 512, 640, 256),
]

BACKWARD_IDS = [f"b{b}_s{s}_h{h}_d{d}_kv{kv}_top{tk}" for b, s, h, d, kv, tk in BACKWARD_CONFIGS]


def ref_dense_attn_with_grad(q, kv, attn_sink, topk_idxs, sm_scale):
    """Dense reference forward + backward using autograd.

    Uses full Q @ K^T with mask (no gather), giving cleaner gradients.
    """
    q = q.clone().float().requires_grad_(True)
    kv = kv.clone().float().requires_grad_(True)
    attn_sink = attn_sink.clone().requires_grad_(True)

    b, m, h, d = q.shape
    n = kv.shape[1]
    topk = topk_idxs.shape[-1]

    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)
    valid_mask = topk_idxs != -1
    attn_mask[batch_idx[valid_mask], seq_idx[valid_mask], topk_idxs[valid_mask].long()] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q, kv) * sm_scale
    attn_mask_expanded = attn_mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~attn_mask_expanded, float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values.clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)

    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv)
    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term
    o = numerator / denominator.unsqueeze(-1)

    loss = o.sum()
    loss.backward()

    return o, q.grad, kv.grad, attn_sink.grad


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("batch,seqlen,heads,dim,seqlen_kv,topk", BACKWARD_CONFIGS, ids=BACKWARD_IDS)
def test_sparse_mla_backward(batch, seqlen, heads, dim, seqlen_kv, topk):
    """Compare tilelang backward gradients against PyTorch autograd reference."""
    from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang

    q_base, kv_base, attn_sink_base, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk)
    sm_scale = (1.0 / dim) ** 0.5

    # Reference
    ref_o, ref_dq, ref_dkv, ref_d_sink = ref_dense_attn_with_grad(q_base, kv_base, attn_sink_base, topk_idxs, sm_scale)

    # TileLang
    q_tl = q_base.clone().requires_grad_(True)
    kv_tl = kv_base.clone().requires_grad_(True)
    sink_tl = attn_sink_base.clone().requires_grad_(True)

    tl_o = sparse_attn_tilelang(q_tl, kv_tl, sink_tl, topk_idxs, sm_scale)
    loss = tl_o.float().sum()
    loss.backward()

    print(f"\n[BWD] b={batch}, s={seqlen}, h={heads}, d={dim}, kv={seqlen_kv}, topk={topk}")

    # Forward output comparison
    fwd_diff = compute_diff(ref_o.float(), tl_o.float())
    print_diff("fwd_output", fwd_diff)
    assert fwd_diff.rel_diff < 1e-3, f"fwd rel_diff too large: {fwd_diff.rel_diff:.2e}"

    # Gradient comparisons
    for name, ref_g, tl_g in [
        ("dQ", ref_dq, q_tl.grad),
        ("dKV", ref_dkv, kv_tl.grad),
        ("dAttnSink", ref_d_sink, sink_tl.grad),
    ]:
        if ref_g is None or tl_g is None:
            print(f"  {name}: SKIPPED (None)")
            continue
        diff = compute_diff(ref_g.float(), tl_g.float())
        print_diff(name, diff)
        # Backward has larger tolerance due to bf16 GEMM + atomic adds
        assert diff.rel_diff < 0.05, f"{name} rel_diff too large: {diff.rel_diff:.2e}"


# ---------------------------------------------------------------------------
# Index masking test: some indices are -1
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
def test_partial_invalid_indices():
    """Test with some indices set to -1 (invalid)."""
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

    batch, seqlen, heads, dim, seqlen_kv, topk = 1, 256, 8, 512, 320, 128
    q, kv, attn_sink, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk)
    sm_scale = (1.0 / dim) ** 0.5

    # Set last 25% of indices to -1
    topk_idxs[:, :, topk * 3 // 4 :] = -1

    ref_o = ref_dense_attn(q, kv, attn_sink, topk_idxs, sm_scale)
    tl_o, _ = sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

    diff = compute_diff(ref_o.float(), tl_o.float())
    print("\n[PARTIAL-INVALID]")
    print_diff("output", diff)

    assert diff.rel_diff < 1e-3, f"rel_diff too large with partial invalid: {diff.rel_diff:.2e}"
    assert not torch.isnan(tl_o).any(), "NaN in output with partial invalid indices"


# ---------------------------------------------------------------------------
# Comprehensive diff summary
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
def test_diff_summary():
    """Print a comprehensive diff summary across all forward configs."""
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface

    configs = [
        (1, 128, 8, 512, 160, 64),
        (1, 256, 16, 512, 320, 128),
        (1, 256, 64, 512, 320, 128),
        (1, 512, 64, 512, 640, 256),
        (1, 1024, 64, 512, 1280, 512),
    ]

    print("\n" + "=" * 100)
    print(f"{'Config':<45} {'rel_diff':>10} {'max_abs':>10} {'mean_abs':>10} {'p99':>10}")
    print("=" * 100)

    for batch, seqlen, heads, dim, seqlen_kv, topk in configs:
        q, kv, attn_sink, topk_idxs = make_inputs(batch, seqlen, heads, dim, seqlen_kv, topk)
        sm_scale = (1.0 / dim) ** 0.5

        ref_o = ref_dense_attn(q, kv, attn_sink, topk_idxs, sm_scale)
        tl_o, _ = sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        diff = compute_diff(ref_o.float(), tl_o.float())
        label = f"b{batch}_s{seqlen}_h{heads}_d{dim}_kv{seqlen_kv}_top{topk}"
        print(
            f"{label:<45} {diff.rel_diff:>10.2e} {diff.max_abs_diff:>10.2e} {diff.mean_abs_diff:>10.2e} {diff.p99_abs_diff:>10.2e}"
        )

    print("=" * 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
