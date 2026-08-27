"""GPU test for the GLM-5.3 triton kpool indexer selection against the torch
reference it replaced (the pre-triton production implementation, kept here
verbatim as the oracle): pooled-key parity at bf16 floor, end-to-end selection
identity on non-tied scores, boundary-tie rate measurement on random and
realistic-scale inputs, bit-determinism across repeated runs, and a >=1.5x
speed bar over the torch path at realistic shapes.

Usage: python tests/glm5_next/test_kpool_indexer.py
"""

import sys

import torch

sys.path.insert(0, ".")

from miles_plugins.models.glm5.ops.tilelang_indexer_fwd import indexer_fwd_interface
from miles_plugins.models.glm5_next.ops.kpool_indexer import (
    SPARSE_MLA_BLOCK,
    build_pooled_keys,
    kpool_select_topk,
    pool_boundaries,
)

INDEX_HEADS = 32
HEAD_DIM = 128
KPOOL = 4
INDEX_TOPK = 2048


# ---- torch reference (the previous production implementation, verbatim) ----


def ref_build_pooled_keys(index_k, gate_score, ape, cu_seqlens, kpool):
    ape = ape.float()
    pooled = []
    boundaries = cu_seqlens.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        num_pools = (end - start) // kpool
        if num_pools == 0:
            continue
        span = slice(start, start + num_pools * kpool)
        keys = index_k[span].float().view(num_pools, kpool, -1)
        scores = gate_score[span].float().view(num_pools, kpool, -1) + ape
        weights = torch.softmax(scores, dim=1)
        pooled.append((weights * keys).sum(dim=1))
    if not pooled:
        return index_k.new_zeros((0, index_k.shape[-1]))
    return torch.cat(pooled, dim=0).to(index_k.dtype)


def ref_pool_topk_to_token(pool_logits, topk, seq_token_base, pool_base, local_positions, shortcut, kpool):
    num_tokens, num_pools = pool_logits.shape
    device = pool_logits.device
    tokens = torch.full((num_tokens, topk), -1, dtype=torch.int64, device=device)

    group_topk = min(topk // kpool, num_pools)
    if group_topk > 0:
        scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
        valid = torch.isfinite(scores)
        token_base = seq_token_base.unsqueeze(1) + (pools - pool_base.unsqueeze(1)) * kpool
        candidates = token_base.unsqueeze(-1) + torch.arange(kpool, device=device)
        candidates = torch.where(valid.unsqueeze(-1), candidates, -1)
        tokens[:, : group_topk * kpool] = candidates.reshape(num_tokens, group_topk * kpool)

    if shortcut.any():
        offsets = torch.arange(topk, device=device)
        shortcut_tokens = seq_token_base.unsqueeze(1) + offsets
        shortcut_valid = offsets.unsqueeze(0) <= local_positions.unsqueeze(1)
        shortcut_tokens = torch.where(shortcut_valid, shortcut_tokens, -1)
        tokens = torch.where(shortcut.unsqueeze(1), shortcut_tokens, tokens)
    return tokens.to(torch.int32)


def ref_append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool, pad_multiple=SPARSE_MLA_BLOCK):
    num_tokens = tokens.shape[0]
    device = tokens.device
    slots = torch.arange(kpool - 1, device=device)
    tail_start = seq_token_base + torch.div(local_positions + 1, kpool, rounding_mode="floor") * kpool
    tail_len = (local_positions + 1) % kpool
    tail = tail_start.unsqueeze(1) + slots
    tail_valid = (slots.unsqueeze(0) < tail_len.unsqueeze(1)) & ~shortcut.unsqueeze(1)
    tail = torch.where(tail_valid, tail, tail.new_full((), -1))
    out = torch.cat([tokens.long(), tail], dim=1)

    width = out.shape[1]
    padded_width = (width + pad_multiple - 1) // pad_multiple * pad_multiple
    if padded_width != width:
        pad = out.new_full((num_tokens, padded_width - width), -1)
        out = torch.cat([out, pad], dim=1)
    return out.to(torch.int32)


def ref_kpool_select_topk(index_q, pooled_k, head_weights, cu_seqlens, pool_cu_seqlens, index_topk, kpool):
    num_tokens = index_q.shape[0]
    device = index_q.device
    token_ids = torch.arange(num_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices]
    pool_base = pool_cu_seqlens[seq_indices]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")
    shortcut = (local_positions + 1) <= index_topk

    if pooled_k.shape[0] > 0:
        with torch.no_grad():
            pool_logits = indexer_fwd_interface(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
                clean_logits=True,
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    tokens = ref_pool_topk_to_token(
        pool_logits, index_topk, seq_token_base, pool_base, local_positions, shortcut, kpool
    )
    tokens = ref_append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    return tokens.unsqueeze(1), pool_logits


# ---- input builders ----


def make_inputs(cu_list, realistic=False, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    total = cu_list[-1]
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)
    index_k = torch.randn(total, HEAD_DIM, device=device, dtype=torch.float32)
    if realistic:
        index_k = torch.nn.functional.rms_norm(index_k, (HEAD_DIM,))
    index_k = index_k.bfloat16()
    gate_score = (torch.randn(total, HEAD_DIM, device=device) * (0.3 if realistic else 1.0)).bfloat16()
    ape = torch.randn(KPOOL, HEAD_DIM, device=device, dtype=torch.float32) * (0.5 if realistic else 1.0)
    index_q = torch.randn(total, INDEX_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    if realistic:
        index_q = index_q / (HEAD_DIM**0.25)
    head_weights = torch.rand(total, INDEX_HEADS, device=device, dtype=torch.float32) * (
        (INDEX_HEADS**-0.5) * (HEAD_DIM**-0.5)
    )
    return index_k, gate_score, ape, index_q, head_weights, cu_seqlens


def run_triton(index_k, gate_score, ape, index_q, head_weights, cu_seqlens):
    pool_cu = pool_boundaries(cu_seqlens, KPOOL)
    pooled = build_pooled_keys(index_k, gate_score, ape, cu_seqlens, KPOOL)
    tokens = kpool_select_topk(index_q, pooled, head_weights, cu_seqlens, pool_cu, INDEX_TOPK, KPOOL)
    return pooled, tokens


def run_ref(index_k, gate_score, ape, index_q, head_weights, cu_seqlens):
    pool_cu = pool_boundaries(cu_seqlens, KPOOL)
    pooled = ref_build_pooled_keys(index_k, gate_score, ape, cu_seqlens, KPOOL)
    tokens, pool_logits = ref_kpool_select_topk(index_q, pooled, head_weights, cu_seqlens, pool_cu, INDEX_TOPK, KPOOL)
    return pooled, tokens, pool_logits


CU_CASES = {
    "single_8192": [0, 8192],
    "packed_mixed": [0, 4097, 4097 + 2048, 4097 + 2048 + 1023, 4097 + 2048 + 1023 + 130, 4097 + 2048 + 1023 + 130 + 2],
    "all_short": [0, 3, 3 + 64, 3 + 64 + 511, 3 + 64 + 511 + 1],
    "tiny": [0, 2],
}


def _row_sets(tokens):
    return [set(row[row >= 0].tolist()) for row in tokens[:, 0, :]]


def _boundary_tie_rows(pool_logits, group_topk):
    if pool_logits.shape[1] <= group_topk:
        return torch.zeros(pool_logits.shape[0], dtype=torch.bool)
    svals, _ = torch.sort(pool_logits, dim=-1, descending=True)
    kth = svals[:, group_topk - 1]
    nxt = svals[:, group_topk]
    return torch.isfinite(kth) & (kth == nxt)


def test_parity_and_selection_identity():
    for case_idx, (name, cu) in enumerate(CU_CASES.items()):
        for realistic in (False, True):
            inputs = make_inputs(cu, realistic=realistic, seed=100 + case_idx)
            pooled_t, tokens_t = run_triton(*inputs)
            pooled_r, tokens_r, pool_logits = run_ref(*inputs)

            n_real = pooled_r.shape[0]
            assert pooled_t.shape[0] >= n_real
            exact = torch.eq(pooled_t[:n_real], pooled_r).float().mean().item() if n_real else 1.0
            if n_real:
                diff = (pooled_t[:n_real].float() - pooled_r.float()).abs()
                rel = (diff.max() / pooled_r.float().abs().max().clamp_min(1e-6)).item()
            else:
                rel = 0.0
            assert rel <= 2**-7, (name, realistic, rel)
            assert (pooled_t[n_real:] == 0).all()

            assert tokens_t.shape == tokens_r.shape, (name, tokens_t.shape, tokens_r.shape)
            byte_equal = torch.equal(tokens_t, tokens_r)
            if not byte_equal:
                group_topk = min(INDEX_TOPK // KPOOL, n_real)
                ties = _boundary_tie_rows(pool_logits, group_topk).cpu()
                sets_t, sets_r = _row_sets(tokens_t.cpu()), _row_sets(tokens_r.cpu())
                bad = [i for i, (a, b) in enumerate(zip(sets_t, sets_r, strict=True)) if a != b and not ties[i]]
                assert not bad, (name, realistic, bad[:5], exact)
            print(
                f"[parity] {name} realistic={realistic}: pooled exact-match {exact * 100:.3f}%, "
                f"rel_err {rel:.2e}, tokens byte-equal {byte_equal}"
            )


def test_tie_rate():
    for realistic in (False, True):
        inputs = make_inputs(CU_CASES["single_8192"], realistic=realistic, seed=7)
        pooled_r, _, pool_logits = run_ref(*inputs)
        group_topk = min(INDEX_TOPK // KPOOL, pooled_r.shape[0])
        eligible = torch.arange(pool_logits.shape[0], device=pool_logits.device) + 1 > INDEX_TOPK
        ties = _boundary_tie_rows(pool_logits, group_topk) & eligible
        rate = ties.float().sum().item() / max(int(eligible.sum().item()), 1)
        print(f"[ties] realistic={realistic}: boundary-tie rate {rate * 100:.4f}% over {int(eligible.sum())} rows")


def test_determinism():
    inputs = make_inputs(CU_CASES["packed_mixed"], realistic=True, seed=3)
    pooled0, tokens0 = run_triton(*inputs)
    for _ in range(2):
        pooled, tokens = run_triton(*inputs)
        assert torch.equal(pooled, pooled0)
        assert torch.equal(tokens, tokens0)
    print("[determinism] 3 runs byte-identical")


def test_replay_off_matches_topk_fn_path():
    from miles_plugins.models.glm5_next.ops.kpool_indexer import _pool_topk_to_token_fn, append_tail_and_pad

    inputs = make_inputs(CU_CASES["packed_mixed"], realistic=False, seed=11)
    index_k, gate_score, ape, index_q, head_weights, cu_seqlens = inputs
    pool_cu = pool_boundaries(cu_seqlens, KPOOL)
    pooled = build_pooled_keys(index_k, gate_score, ape, cu_seqlens, KPOOL)
    fused = kpool_select_topk(index_q, pooled, head_weights, cu_seqlens, pool_cu, INDEX_TOPK, KPOOL)

    num_tokens = index_q.shape[0]
    token_ids = torch.arange(num_tokens, device=index_q.device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices].to(torch.int32)
    pool_base = pool_cu[seq_indices].to(torch.int32)
    local_positions = (token_ids - seq_token_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, KPOOL, rounding_mode="floor")
    shortcut = (local_positions + 1) <= INDEX_TOPK
    with torch.no_grad():
        pool_logits = indexer_fwd_interface(
            index_q,
            pooled,
            head_weights,
            pool_base.to(torch.int32),
            (pool_base + eligible_pools).to(torch.int32),
            clean_logits=True,
        )
    tokens = _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, KPOOL)(pool_logits, INDEX_TOPK)
    staged = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, KPOOL)
    assert torch.equal(fused, staged.unsqueeze(1))
    print("[replay-path] fused select == staged topk_fn + append_tail_and_pad")


def _time_ms(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def test_perf():
    rows = []
    for name in ("single_8192", "packed_mixed"):
        inputs = make_inputs(CU_CASES[name], realistic=True, seed=5)
        t_ref = _time_ms(lambda inputs=inputs: run_ref(*inputs))
        t_tri = _time_ms(lambda inputs=inputs: run_triton(*inputs))
        rows.append((name, t_ref, t_tri, t_ref / t_tri))
        print(f"[perf] {name}: torch {t_ref:.3f} ms, triton {t_tri:.3f} ms, speedup {t_ref / t_tri:.2f}x")
    assert all(r[3] >= 1.5 for r in rows), rows


if __name__ == "__main__":
    test_parity_and_selection_identity()
    test_tie_rate()
    test_determinism()
    test_replay_off_matches_topk_fn_path()
    test_perf()
    print("all kpool indexer tests passed")
