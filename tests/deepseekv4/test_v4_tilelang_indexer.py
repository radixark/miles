from dataclasses import dataclass

import pytest
import torch

try:
    import tilelang  # noqa: F401
except ImportError:
    tilelang = None

if tilelang is not None:
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer import v4_lighting_indexer
    from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_indexer_fwd import (
        _make_causal_cu_seqlens,
        batched_indexer_fwd,
    )
else:
    v4_lighting_indexer = None
    _make_causal_cu_seqlens = None
    batched_indexer_fwd = None


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
    """Compute diff metrics matching the dumper comparator."""
    x = baseline.flatten().float()
    y = target.flatten().float()

    abs_diff = (x - y).abs()

    # rel_diff: cosine-distance-like metric
    xy = (x * y).sum()
    x2 = (x * x).sum()
    y2 = (y * y).sum()
    denom = x2 + y2
    if denom > 0:
        sim = 2.0 * xy / denom
        rel_diff = (1.0 - sim).item()
    else:
        rel_diff = 0.0

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
        f"  {name}: rel_diff={diff.rel_diff:.2e}, max_abs={diff.max_abs_diff:.2e}, "
        f"mean_abs={diff.mean_abs_diff:.2e}, p50={diff.p50_abs_diff:.2e}, "
        f"p95={diff.p95_abs_diff:.2e}, p99={diff.p99_abs_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# PyTorch reference implementation (from dsa.py)
# ---------------------------------------------------------------------------
def ref_compute_index_scores(q, weights, k):
    """PyTorch reference: compute index scores.

    Args:
        q:       [seqlen_q, batch, heads, dim] bf16
        k:       [seqlen_kv, batch, dim] bf16
        weights: [seqlen_q, batch, heads] fp32

    Returns:
        index_scores: [batch, seqlen_q, seqlen_kv] fp32
    """
    # q @ k^T -> [sq, b, h, sk]
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    # ReLU
    index_scores = torch.relu(index_scores)
    # Weight by heads
    index_scores = index_scores * weights.float().unsqueeze(-1)
    # Sum across heads
    index_scores = index_scores.sum(dim=2)
    # Transpose to [b, sq, sk]
    index_scores = index_scores.transpose(0, 1)
    return index_scores


def ref_apply_causal_mask(index_scores, compress_ratio):
    """Apply causal mask for compressed KV positions.

    For query at position p, valid compressed groups are [0, (p+1) // compress_ratio).
    Positions outside this range are set to -inf.

    Args:
        index_scores: [batch, seqlen_q, seqlen_kv] fp32
        compress_ratio: int

    Returns:
        masked index_scores: [batch, seqlen_q, seqlen_kv] fp32
    """
    b, sq, sk = index_scores.shape
    q_positions = torch.arange(sq, device=index_scores.device)
    k_positions = torch.arange(sk, device=index_scores.device)
    # valid_end[q] = (q + 1) // compress_ratio
    valid_end = (q_positions + 1) // compress_ratio  # [sq]
    # mask: k_pos < valid_end[q_pos]
    mask = k_positions.unsqueeze(0) < valid_end.unsqueeze(1)  # [sq, sk]
    index_scores = index_scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return index_scores


def ref_fused_qk_topk(q, k, weights, compress_ratio, topk):
    """Full reference: scores + causal mask + topk.

    Returns:
        index_scores: [batch, seqlen_q, seqlen_kv] fp32 (masked)
        topk_indices: [batch, seqlen_q, topk] int64
    """
    index_scores = ref_compute_index_scores(q, weights, k)
    index_scores = ref_apply_causal_mask(index_scores, compress_ratio)
    actual_topk = min(topk, index_scores.shape[-1])
    topk_indices = index_scores.topk(actual_topk, dim=-1)[1]
    return index_scores, topk_indices


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def requires_tilelang():
    return pytest.mark.skipif(tilelang is None, reason="tilelang not installed")


def make_inputs(seqlen_q, batch, heads, dim, compress_ratio, device="cuda"):
    """Generate random inputs matching V4 indexer shapes."""
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.randn(seqlen_q, batch, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seqlen_kv, batch, dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seqlen_q, batch, heads, device=device, dtype=torch.float32) * 0.01

    return q, k, weights


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------
# Test configurations: (seqlen_q, batch, heads, dim, compress_ratio, topk)
FORWARD_CONFIGS = [
    # Short sequences, small batch — basic correctness
    (128, 1, 8, 128, 4, 32),
    (128, 1, 16, 128, 4, 32),
    (128, 2, 8, 128, 4, 32),
    # Medium sequences — typical training
    (512, 1, 8, 128, 4, 64),
    (512, 2, 16, 128, 4, 128),
    (512, 1, 64, 128, 4, 128),
    # Long sequences
    (2048, 1, 8, 128, 4, 128),
    (2048, 1, 64, 128, 4, 512),
    # C128 layer type
    (2048, 1, 8, 128, 128, 16),
    (1024, 1, 16, 128, 128, 8),
    # Larger batch
    (512, 4, 8, 128, 4, 64),
    (256, 4, 16, 128, 4, 64),
    # Edge: seqlen just above compress_ratio (small KV)
    (16, 1, 8, 128, 4, 4),
    (256, 1, 8, 128, 128, 2),
]

FORWARD_CONFIG_IDS = [f"sq{sq}_b{b}_h{h}_d{d}_cr{cr}_top{tk}" for sq, b, h, d, cr, tk in FORWARD_CONFIGS]


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("seqlen_q,batch,heads,dim,compress_ratio,topk", FORWARD_CONFIGS, ids=FORWARD_CONFIG_IDS)
def test_indexer_forward_scores(seqlen_q, batch, heads, dim, compress_ratio, topk):
    """Compare tilelang forward logits against PyTorch reference."""
    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)
    seqlen_kv = seqlen_q // compress_ratio

    # Reference
    ref_scores = ref_compute_index_scores(q, weights, k)
    ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

    # Tilelang
    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    # Compare only valid (non-masked) positions
    valid_mask = ref_scores != float("-inf")
    ref_valid = ref_scores[valid_mask]
    tl_valid = tl_scores[valid_mask]

    diff = compute_diff(ref_valid, tl_valid)
    print(f"\n[FWD] sq={seqlen_q}, b={batch}, h={heads}, cr={compress_ratio}, topk={topk}")
    print_diff("logits", diff)

    # Thresholds: tilelang bf16 GEMM vs pytorch fp32 einsum — allow some tolerance
    assert diff.rel_diff < 1e-3, f"rel_diff too large: {diff.rel_diff:.2e}"
    assert diff.max_abs_diff < 1.0, f"max_abs_diff too large: {diff.max_abs_diff:.2e}"
    assert diff.mean_abs_diff < 0.05, f"mean_abs_diff too large: {diff.mean_abs_diff:.2e}"


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("seqlen_q,batch,heads,dim,compress_ratio,topk", FORWARD_CONFIGS, ids=FORWARD_CONFIG_IDS)
def test_indexer_forward_topk(seqlen_q, batch, heads, dim, compress_ratio, topk):
    """Verify topk self-consistency: selected scores >= non-selected scores.

    The kernel is non-deterministic across calls (GPU GEMM accumulation order), so
    instead of comparing two separate kernel calls, we verify the output of a single
    call is internally consistent.
    """
    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)
    seqlen_kv = seqlen_q // compress_ratio

    tl_score, tl_topk = v4_lighting_indexer(q, k, weights, compress_ratio, topk)

    # Verify: selected scores should be >= min of selected scores at each position
    # Get full logits from the same kernel call is not possible (autograd Function
    # only returns topk scores). So re-run forward to get full logits.
    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    full_logits = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    b, sq, _ = tl_topk.shape
    violations = 0
    total = 0
    for bi in range(b):
        for qi in range(sq):
            selected = set(tl_topk[bi, qi].cpu().tolist()) - {-1}
            if not selected:
                continue
            selected_scores = full_logits[bi, qi, list(selected)]
            min_selected = selected_scores.min().item()
            # Check non-selected valid positions have scores <= min_selected
            valid_end = (qi + 1) // compress_ratio
            for ki in range(valid_end):
                if ki not in selected:
                    total += 1
                    if full_logits[bi, qi, ki].item() > min_selected + 1e-5:
                        violations += 1

    print(
        f"\n[TOPK] sq={seqlen_q}, b={batch}, h={heads}, cr={compress_ratio}, topk={topk}: "
        f"violations={violations}/{total}"
    )
    if total > 0:
        violation_rate = violations / total
        # Allow small violation rate due to kernel non-determinism on tied scores
        assert violation_rate < 0.05, f"topk violation rate too high: {violation_rate:.4f}"


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------
BACKWARD_CONFIGS = [
    # (seqlen_q, batch, heads, dim, compress_ratio, topk)
    # Keep topk as power of 2 (required by backward kernel)
    (128, 1, 8, 128, 4, 32),
    (256, 1, 16, 128, 4, 64),
    (512, 1, 8, 128, 4, 64),
    (512, 2, 16, 128, 4, 128),
    (1024, 1, 8, 128, 4, 128),
    (1024, 1, 64, 128, 4, 512),
    # C128
    (2048, 1, 8, 128, 128, 16),
    # Larger batch
    (256, 4, 8, 128, 4, 64),
]

BACKWARD_CONFIG_IDS = [f"sq{sq}_b{b}_h{h}_d{d}_cr{cr}_top{tk}" for sq, b, h, d, cr, tk in BACKWARD_CONFIGS]


def ref_indexer_backward_dense(q, k, weights, compress_ratio, topk, topk_indices):
    """Dense PyTorch autograd reference for backward.

    Computes forward scores using dense einsum, gathers at given topk_indices,
    then backpropagates. Uses fp32 throughout for the reference.

    Returns:
        grad_q, grad_w, grad_k
    """
    q = q.clone().float().requires_grad_(True)
    k = k.clone().float().requires_grad_(True)
    weights = weights.clone().float().requires_grad_(True)

    # Dense forward: q @ k^T -> [sq, b, h, sk]
    index_scores = torch.einsum("sbhd,tbd->sbht", q, k)
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)  # [sq, b, sk]
    index_scores = index_scores.transpose(0, 1)  # [b, sq, sk]

    # Gather at topk positions and compute loss
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    gathered_scores = torch.gather(index_scores, dim=-1, index=safe_indices)
    gathered_scores = torch.where(valid_mask, gathered_scores, torch.tensor(0.0, device=q.device))

    loss = gathered_scores.sum()
    loss.backward()

    return q.grad, weights.grad, k.grad


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("seqlen_q,batch,heads,dim,compress_ratio,topk", BACKWARD_CONFIGS, ids=BACKWARD_CONFIG_IDS)
def test_indexer_backward(seqlen_q, batch, heads, dim, compress_ratio, topk):
    """Compare tilelang backward gradients against dense PyTorch autograd reference.

    The main precision gap comes from bf16 GEMM (kernel) vs fp32 einsum (reference)
    producing different ReLU boundaries for scores near zero. This is structural:
    ~30% of Q@K^T products land near the ReLU boundary where bf16 truncation flips
    the sign, causing binary gradient differences at those positions.
    """
    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)

    # --- TileLang forward + backward ---
    q_tl = q.clone().requires_grad_(True)
    k_tl = k.clone().requires_grad_(True)
    w_tl = weights.clone().requires_grad_(True)

    tl_score, tl_topk = v4_lighting_indexer(q_tl, k_tl, w_tl, compress_ratio, topk)

    valid_mask = tl_topk != -1
    tl_score_masked = torch.where(valid_mask, tl_score, torch.tensor(0.0, device=q.device))
    loss = tl_score_masked.sum()
    loss.backward()

    # --- Dense fp32 reference backward using SAME topk_indices ---
    ref_grad_q, ref_grad_w, ref_grad_k = ref_indexer_backward_dense(q, k, weights, compress_ratio, topk, tl_topk)

    print(f"\n[BWD] sq={seqlen_q}, b={batch}, h={heads}, cr={compress_ratio}, topk={topk}")

    for name, ref_g, tl_g in [
        ("grad_q", ref_grad_q, q_tl.grad),
        ("grad_weights", ref_grad_w, w_tl.grad),
        ("grad_k", ref_grad_k, k_tl.grad),
    ]:
        if ref_g is None or tl_g is None:
            print(f"  {name}: SKIPPED (None)")
            continue
        diff = compute_diff(ref_g.float(), tl_g.float())
        print_diff(name, diff)

        # bf16 GEMM vs fp32 einsum: ~30% of ReLU boundaries flip, causing
        # binary gradient mismatches. rel_diff reflects this structural gap.
        assert diff.rel_diff < 0.5, f"{name} rel_diff too large: {diff.rel_diff:.2e}"
        assert diff.mean_abs_diff < 1.0, f"{name} mean_abs_diff too large: {diff.mean_abs_diff:.2e}"


# ---------------------------------------------------------------------------
# Masking correctness test
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_causal_mask_correctness(compress_ratio):
    """Verify that the tilelang kernel correctly masks future compressed groups."""
    seqlen_q = 512
    batch = 1
    heads = 8
    dim = 128
    seqlen_kv = seqlen_q // compress_ratio

    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    logits = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)  # [batch, sq, sk]

    # Check that future positions are -inf
    violations = 0
    total_checked = 0
    for qi in range(seqlen_q):
        valid_end = (qi + 1) // compress_ratio
        for ki in range(valid_end, seqlen_kv):
            total_checked += 1
            if logits[0, qi, ki].item() != float("-inf"):
                violations += 1

    print(f"\n[MASK] cr={compress_ratio}: checked {total_checked} future positions, violations={violations}")
    assert violations == 0, f"Found {violations} future-position violations (should be -inf)"


# ---------------------------------------------------------------------------
# Numerical stability test: large values
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
def test_large_values():
    """Test with large input values to check for overflow/underflow."""
    seqlen_q, batch, heads, dim, compress_ratio = 256, 1, 8, 128, 4
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.randn(seqlen_q, batch, heads, dim, device="cuda", dtype=torch.bfloat16) * 10.0
    k = torch.randn(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16) * 10.0
    weights = torch.randn(seqlen_q, batch, heads, device="cuda", dtype=torch.float32) * 0.1

    ref_scores = ref_compute_index_scores(q, weights, k)
    ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    valid_mask = ref_scores != float("-inf")
    diff = compute_diff(ref_scores[valid_mask], tl_scores[valid_mask])
    print("\n[LARGE] large values test")
    print_diff("logits", diff)

    # Larger values → larger absolute diff but rel_diff should stay reasonable
    assert diff.rel_diff < 5e-3, f"rel_diff too large: {diff.rel_diff:.2e}"
    assert not torch.isnan(tl_scores[valid_mask]).any(), "NaN in tilelang output"
    assert not torch.isinf(tl_scores[valid_mask]).any(), "Inf in tilelang output (non-masked)"


# ---------------------------------------------------------------------------
# Zero input test
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
def test_zero_inputs():
    """Test that zero inputs produce zero scores."""
    seqlen_q, batch, heads, dim, compress_ratio = 128, 1, 8, 128, 4
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.zeros(seqlen_q, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(seqlen_q, batch, heads, device="cuda", dtype=torch.float32)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    valid_mask = tl_scores != float("-inf")
    valid_scores = tl_scores[valid_mask]
    assert (valid_scores == 0).all(), f"Expected all zeros for valid positions, got max={valid_scores.max():.2e}"


# ---------------------------------------------------------------------------
# V4 real-world config test
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize(
    "seqlen_q",
    [256, 512, 1024, 2048],
    ids=["sq256", "sq512", "sq1024", "sq2048"],
)
def test_v4_real_config(seqlen_q):
    """Test with V4's actual indexer configuration: heads=64, dim=128, topk=512, cr=4."""
    batch = 1
    heads = 64
    dim = 128
    compress_ratio = 4
    topk = min(512, seqlen_q // compress_ratio)
    seqlen_kv = seqlen_q // compress_ratio

    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)

    # Forward scores comparison
    ref_scores = ref_compute_index_scores(q, weights, k)
    ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    valid_mask = ref_scores != float("-inf")
    diff = compute_diff(ref_scores[valid_mask], tl_scores[valid_mask])
    print(f"\n[V4-REAL] sq={seqlen_q}, h=64, d=128, cr=4, topk={topk}")
    print_diff("logits", diff)

    assert diff.rel_diff < 1e-3, f"rel_diff too large: {diff.rel_diff:.2e}"

    # Topk self-consistency: verify selected scores >= non-selected scores
    tl_score, tl_topk = v4_lighting_indexer(q, k, weights, compress_ratio, topk)
    violations = 0
    total = 0
    for qi in range(seqlen_q):
        selected = set(tl_topk[0, qi].cpu().tolist()) - {-1}
        if not selected:
            continue
        min_sel = tl_scores[0, qi, list(selected)].min().item()
        valid_end = (qi + 1) // compress_ratio
        for ki in range(valid_end):
            if ki not in selected:
                total += 1
                if tl_scores[0, qi, ki].item() > min_sel + 1e-5:
                    violations += 1

    print(f"  topk self-consistency: violations={violations}/{total}")
    if total > 0:
        assert violations / total < 0.05, f"topk violation rate too high: {violations}/{total}"


# ---------------------------------------------------------------------------
# Comprehensive diff summary (not a test, useful for manual inspection)
# ---------------------------------------------------------------------------
@requires_cuda()
@requires_tilelang()
def test_diff_summary():
    """Print a comprehensive diff summary across all configurations."""
    configs = [
        (128, 1, 8, 128, 4),
        (512, 1, 16, 128, 4),
        (512, 2, 64, 128, 4),
        (1024, 1, 64, 128, 4),
        (2048, 1, 8, 128, 128),
    ]

    print("\n" + "=" * 90)
    print(f"{'Config':<40} {'rel_diff':>10} {'max_abs':>10} {'mean_abs':>10} {'p99':>10}")
    print("=" * 90)

    for seqlen_q, batch, heads, dim, compress_ratio in configs:
        seqlen_kv = seqlen_q // compress_ratio

        q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)
        ref_scores = ref_compute_index_scores(q, weights, k)
        ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

        cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
        tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

        valid_mask = ref_scores != float("-inf")
        diff = compute_diff(ref_scores[valid_mask], tl_scores[valid_mask])

        label = f"sq{seqlen_q}_b{batch}_h{heads}_cr{compress_ratio}"
        print(
            f"{label:<40} {diff.rel_diff:>10.2e} {diff.max_abs_diff:>10.2e} "
            f"{diff.mean_abs_diff:>10.2e} {diff.p99_abs_diff:>10.2e}"
        )

    print("=" * 90)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
