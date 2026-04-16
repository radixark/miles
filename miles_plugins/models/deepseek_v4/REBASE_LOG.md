# DeepSeek-V4 Operator Rebase Log

Tracking operator-level changes from V4's original implementation toward GLM-5 operators.
See `ARCHITECTURE.md` Section 5 for the full rebase analysis.

## Verification Method

- `run_megatron` with TP=1, seq_length=256, 5-layer model
- Source patcher: `scripts/debug/v4_rebase_patches.yaml` (28 dump points)
- Routing replay enabled (record baseline, replay on target)
- Comparator preset: `raw`, threshold: `1e-4`
- Baseline: original nn.Linear + RMSNorm implementation

## Changes

### 1. `nn.Linear` → `TELinear` for `wq_a` and `wkv`

**Date**: 2026-03-15
**Files**: `deepseek_v4.py`
**What**: Removed `_use_te_linear_for_wq_a_wkv` flag and `nn.Linear` fallback. `wq_a` and `wkv` now always use `TELinear(parallel_mode="duplicated")`. Forward simplified from conditional `[0]` to always `[0]`.
**Checkpoint**: Re-converted to `/node_public/models/DeepSeek-V4-285B-5layer_te_torch_dist`
**Result**: 145/147 passed (2 failed = `indexer_topk_idxs` tie-breaking, expected)

| Tensor | rel_diff |
|--------|----------|
| q_after_wq_a | ~1e-5 |
| kv_after_wkv | ~8e-6 |
| attn_out_pre_inv_rope | ~1e-5 |
| mlp_raw_output | ~2e-7 |
| hc_ffn_post_output (final) | ~4e-7 |

### 2. `RMSNorm` → `TENorm` for `q_norm` and `kv_norm`

**Date**: 2026-03-15
**Files**: `deepseek_v4.py`
**What**: Replaced custom `RMSNorm` (from `ref_model.py`) with Megatron's `TENorm(config_no_sp, ...)` for both `q_norm` and `kv_norm`. Uses `config_no_sp` since these norms operate on replicated data inside attention (after SP gather).
**Checkpoint**: Re-converted (same path, overwritten)
**Result**: 145/147 passed (2 failed = `indexer_topk_idxs`, expected)

| Tensor | rel_diff |
|--------|----------|
| q_after_norm | ~1.6e-5 |
| kv_vanilla_after_norm | ~9e-6 |
| attn_out_pre_inv_rope | ~9e-6 |
| mlp_raw_output | ~2e-7 |
| hc_ffn_post_output (final) | ~4e-7 |

### 3. Tilelang sparse MLA attention kernel (fork GLM-5, add `attn_sink`)

**Date**: 2026-03-13
**Files**: `ops/tilelang_sparse_mla_fwd.py`, `ops/tilelang_sparse_mla_bwd.py` (rewrote both)
**What**: Forked GLM-5's tilelang sparse MLA kernels and adapted for V4's architecture:
- Removed `kv_group` dimension and D/D_tail split (V4 uses single-head KV `[B, S_kv, D]`, no split)
- Removed `kv_group` from index shape (`[B, S, topk]` instead of `[B, S, kv_group, topk]`)
- Single Q@KV GEMM instead of two (D + D_tail)
- **Added `attn_sink`** to forward: `sumexp[h] += exp2(attn_sink[h]*log2e - m_i[h]*log2e)` after the main attention loop, modifying the softmax denominator
- **Added `dAttnSink`** gradient to backward: `dAttnSink[h] += -Delta[b,s,h] * exp2(attn_sink[h]*log2e - LSE[b,s,h])` via atomic adds (no sm_scale — attn_sink is a pre-scaled logit)
- `attention_core.py` (`DeepSeekV4SparseAttention` autograd wrapper) unchanged — already had matching interface
- **Topk padding**: Interface functions pad `topk_idxs` with `-1` to next multiple of block size (fwd: 64, bwd: 32) for non-divisible topk (e.g., V4's topk=130)
**Tests**: `tests/test_v4_tilelang_sparse_mla.py` — 11 forward configs, 5 backward configs, attn_sink mode tests, partial invalid indices. Dense torch reference (full Q@K^T with mask).
**Bugs fixed during debug**:
- `out_idx=[-2]` → `out_idx=[-3]` in bwd kernel (dQ is the auto-allocated output, not dKV)
- `attn_sink * sm_scale` → `attn_sink * log2(e)` in both fwd and bwd (attn_sink is pre-scaled, only needs ln→log2 conversion)
**Result**: Verified on Megatron 5-layer fwd+bwd (TP=8, SP, seq_length=256)

| Tensor (Layer 2 — DSA layer) | rel_diff | max_abs |
|-------------------------------|----------|---------|
| attn_out_pre_inv_rope | 1.22e-05 | 3.91e-02 |
| attn_after_wo_a | 2.87e-05 | 3.91e-02 |
| attn_after_wo_b | 3.15e-05 | 2.50e-01 |
| hc_attn_post_output | 6.07e-07 | 2.00e+00 |
| mlp_raw_output | 1.56e-06 | 1.00e+00 |

Loss: 13.0721 (dense) vs 13.0764 (tilelang), delta=0.03%

### 4. Tilelang `lighting_indexer` (replace `fused_qk_topk_naive`)

**Date**: 2026-03-13
**Files**: `ops/tilelang_indexer_fwd.py`, `ops/tilelang_indexer_bwd.py`, `ops/tilelang_indexer.py` (all new)
**What**: Adapted GLM-5's `lighting_indexer` tilelang kernel for V4's DSA indexer:
- Batch dimension handled by external loop (kernel operates per-sample, same as GLM-5)
- Causal masking via `cu_seqlens`: for query at position `p`, valid compressed KV range is `[0, (p+1)//compress_ratio)`, converted to `cu_seqlen_ks=0`, `cu_seqlen_ke=(p+1)//ratio`
- Core kernel logic identical to GLM-5: GEMM → ReLU → weight scaling → head reduction → topk
- `V4IndexerFunction` autograd wrapper handles SBHD↔SHD reshaping and cu_seqlens generation
- Supports both C4 (`compress_ratio=4`) and C128 (`compress_ratio=128`) layer types
- Note: Hadamard rotation and compressor are applied by the caller (outside the kernel), matching V4's `DSAIndexer.forward_before_topk` flow
- Replay manager and KL loss integration not yet wired — requires changes in `dsa.py` to call `v4_lighting_indexer` instead of `fused_qk_topk_naive`
**Tests**: `tests/test_v4_tilelang_indexer.py` — 14 forward configs, 8 backward configs, causal mask correctness, numerical stability, V4 real config (h=64, d=128, topk=512)
**Result**: Verified (see #5 for e2e)

### 5. `V4Indexer` — move indexer from Megatron dsa.py to miles_plugins

**Date**: 2026-03-16
**Files**: `ops/v4_indexer.py` (new), `deepseek_v4.py` (modified)
**What**: Created self-contained `V4Indexer` module in miles_plugins, replacing Megatron's `DSAIndexer`:
- Moved all indexer logic (linear_wq_b, linear_weights_proj, compressor, RoPE, hadamard, score+topk) from `dsa.py` into `ops/v4_indexer.py`
- Uses tilelang kernel directly for score computation (causal mask via cu_seqlens, no dense float mask)
- Parameter names preserved for checkpoint compatibility (`linear_wq_b`, `linear_weights_proj`, `compressor`, `freqs_cis`)
- Replay manager integration preserved
- **Removed KL loss support** — `FusedDSAIndexerLoss` and `DSAIndexerLossAutoScaler` no longer in the V4 code path
- Legacy `DSAIndexer` path kept behind env var for reproducibility
**Env vars**:
- `V4_INDEXER_IMPL=tilelang` (default) → `V4Indexer` in miles_plugins
- `V4_INDEXER_IMPL=legacy` → `DSAIndexer` in Megatron's dsa.py
**Tests**: `tests/test_v4_tilelang_indexer.py` — 45 tests (14 forward scores, 14 topk self-consistency, 8 backward, causal mask, edge cases)
**Bugs fixed during debug**:
- Backward kernel `block_I` assertion for small topk: pad topk_indices/grad_scores to 32-boundary with -1/0 (zero contribution, no accuracy impact)
- Weights dtype mismatch: added `.float()` cast in `indexer_fwd_interface` (kernel expects fp32 weights, DSAIndexer produces bf16)
**Result**: 672/672 passed, 0 failed (TP=8, SP, seq_length=256)

| Metric | Value |
|--------|-------|
| Loss (baseline dense) | 13.0753 |
| Loss (tilelang attn + V4Indexer) | 13.0770 |
| Loss delta | 0.01% |
| Activations passed | 672/672 |
| Activations failed | 0 |

### 6. Contiguous (allgather) CP — replace zigzag CP

**Date**: 2026-03-16
**Files**: `ops/cp_utils.py` (rewritten), `ops/compressor.py` (simplified), `deepseek_v4.py` (updated imports/calls), `ops/v4_indexer.py` (added CP k-gather)
**Test harness**: `run_megatron` — `batch.py` (added `--allgather-cp` flag), `worker/main.py`, `cli/commands/args.py`, `worker/script_args.py`, `worker/replay.py`
**What**: Replaced zigzag CP with contiguous (allgather) CP partitioning:
- `cp_utils.py`: Removed `zigzag_to_natural`, `natural_to_zigzag_slice`, `get_pos_emb_on_this_cp_rank`. New `all_gather_cp` (plain cat, no reorder). Position/freqs use contiguous slicing.
- `compressor.py`: `overlap_transform_with_cp` uses `all_gather_cp` + contiguous slice instead of zigzag reorder.
- `deepseek_v4.py`: KV all-gather uses `all_gather_cp` instead of `all_gather_cp_natural_order`.
- `v4_indexer.py`: Added CP all-gather on compressed keys `k` so each rank scores against global KV. `cu_seqlens` computed from global positions, sliced to local queries.
- `run_megatron`: `--allgather-cp` flag controls contiguous vs zigzag data slicing in `batch.py`. `replay.py` uses contiguous slice instead of `natural_to_zigzag_slice`.
- `v4_rebase_patches.yaml`: Added `[cp]` annotations to all sequence-dim tensors, `# cp:replicated` for post-gather tensors.
**Bugs fixed during debug**:
- V4Indexer only saw local compressed keys (64 instead of 128 with CP=2) → added `all_gather_cp(k, dim=0)` on indexer's compressed keys
- `replay.py` imported deleted `deepseek_v4_cp_utils` → replaced with inline contiguous slice
**Result**: Verified on Megatron 5-layer forward (TP=1 CP=1 vs TP=1 CP=2, and TP=8 CP=1 vs TP=4 CP=2)

| Config | Activations | Failed | Errored |
|--------|-------------|--------|---------|
| TP=1 CP=1 vs TP=1 CP=2 | 54 passed | 0 | 0 (70 dims annotation errors for `tp:replicated` on TP=1) |
| TP=8 CP=1 vs TP=4 CP=2 | 117 passed | 3 | 5 |
| TP=1 CP=1 vs TP=4 CP=2 (routing replay) | 123 passed | 2 | 0 |

Failed: `indexer_topk_idxs` (topk tie-breaking, expected), `mlp_raw_output` at 1 layer (rel=1.81e-03, different MoE routing without replay)
Errored: `kv_assembled` dims annotation `cp:replicated` invalid when baseline has no CP axis

## Remaining (per ARCHITECTURE.md Section 5.3)

| Operator | Status | Risk |
|----------|--------|------|
| TELinear for wq_a, wkv | Done | Low |
| TENorm for q_norm, kv_norm | Done | Low |
| Tilelang attention kernel (fork GLM-5, add attn_sink) | Done | Low |
| lighting_indexer (replace fused_qk_topk_naive) | Done | Low |
| V4Indexer (move indexer to miles_plugins) | Done | Low |
| Contiguous (allgather) CP | Done | Low |
| RoPE | Keep V4's complex-number format | N/A |
| Data format | Keep BSHD | N/A |
