# DeepSeek-V4 Architecture

## Overview

DeepSeek-V4 (285B) is a 43-layer Transformer with three key architectural innovations over V3:
1. **Compressed Attention** — a variant of DSA (DeepSeek Sparse Attention) that adds learned compressors to reduce KV length
2. **Hyper-Connection (mHC)** — replaces standard residual connections with a multi-stream mixer
3. **Hash-based MoE routing** — deterministic token-ID-to-expert mapping for early layers

Model dimensions: `hidden_size=4096`, `num_heads=64`, `kv_lora_rank=512`, `qk_pos_emb_head_dim=64`, `vocab_size=129280`, `num_experts=256`, `topk=6`.

---

## 1. Compressed Attention

### 1.1 Layer Types

Each of the 43 layers has a per-layer `compress_ratio` from the config:

```
[0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 128, 4]
```

This defines three layer types:

| Type | `compress_ratio` | Layers | Components |
|------|-----------------|--------|------------|
| **C1** | 0 | 0, 1 | SWA only (no compression) |
| **C4** | 4 | 2, 4, 6, ..., 42 | SWA + Compressor(ratio=4) + Indexer |
| **C128** | 128 | 3, 5, 7, ..., 41 | SWA + Compressor(ratio=128), static indices |

All layers share the same attention mechanism — the difference is in how the KV set is constructed.

### 1.2 Query Path (shared across all layers)

```
x: [B, S, 4096]
    │
    ├─ wq_a: Linear(4096, 1024)          # down-project
    ├─ q_norm: RMSNorm(1024)             # normalize
    ├─ wq_b: ColumnParallelLinear(1024, 64*512)  # up-project, TP-sharded
    ├─ unflatten → [B, S, n_local_heads, 512]
    ├─ per-head RMSNorm: x * rsqrt(x².mean(-1) + eps)
    └─ apply_rotary_emb on last 64 dims  # RoPE on q[..., -64:]
    → q: [B, S, n_local_heads, 512]
```

The intermediate `qr` (after `q_norm`, before `wq_b`) is reused by the Indexer in C4 layers.

### 1.3 Vanilla KV Path (shared across all layers)

Every layer produces "vanilla" (uncompressed) KV tokens, one per input position:

```
x: [B, S, 4096]
    │
    ├─ wkv: Linear(4096, 512)   # single-head KV projection
    ├─ kv_norm: RMSNorm(512)
    └─ apply_rotary_emb on last 64 dims  # RoPE on kv[..., -64:]
    → kv_vanilla: [B, S, 512]
```

### 1.4 Compressor (C4 and C128 layers)

The compressor reduces `S` tokens into `S/ratio` compressed KV tokens via a learned gated pooling.

**Input**: `x: [B, S, 4096]` (the hidden states, same as attention input)

**Architecture** (`DeepSeekV4Compressor`):
```
x_fp32 = x.float()                        # all compressor math in FP32

kv   = wkv(x_fp32)                        # Linear(4096, coff*512), no bias
score = wgate(x_fp32)                      # Linear(4096, coff*512), no bias
                                           # coff = 2 for C4 (overlap), 1 for C128

# Reshape into groups of `ratio` consecutive tokens
kv    = kv.unflatten(1, (-1, ratio))       # [B, S/ratio, ratio, coff*512]
score = score.unflatten(1, (-1, ratio)) + ape   # ape: learnable [ratio, coff*512]
```

**C4 overlap transform** (`coff=2`, `overlap=True`):

For C4 layers, each group sees not only its own `ratio=4` tokens but also the previous group's tokens, forming a sliding window of `2*ratio=8` tokens per group:

```python
# overlap_transform: [B, G, ratio, 2*d] → [B, G, 2*ratio, d]
new[:, :, ratio:]  = tensor[:, :, :, d:]    # current group's second half
new[:, 1:, :ratio] = tensor[:, :-1, :, :d]  # previous group's first half
# First group's previous slot is zero-filled (kv) or -inf-filled (score)
```

**C128 no overlap** (`coff=1`, `overlap=False`):

Each group of 128 tokens is compressed independently with no overlap.

**Gated pooling and output**:
```
score_softmax = softmax(score, dim=2)      # softmax over tokens within each group
kv = (kv * score_softmax).sum(dim=2)       # weighted sum → [B, S/ratio, 512]
kv = RMSNorm(kv.to(bf16))                  # normalize
apply_rotary_emb(kv[..., -64:], freqs_cis) # RoPE at compressed positions
→ kv_compress: [B, S/ratio, 512]
```

The compressed positions for RoPE use stride `ratio` (i.e., position 0, 4, 8, ... for C4; position 0, 128, 256, ... for C128).

### 1.5 KV Assembly and Index Selection

The full KV tensor concatenates vanilla and compressed tokens:

```
kv = cat([kv_vanilla, kv_compress], dim=1)   # [B, S + S/ratio, 512]
```

The `topk_idxs` tensor tells each query which KV positions to attend to. It is built by concatenating:

1. **Window indices (all layers)**: For each query at position `p`, the local SWA window selects indices `[max(0, p-W+1), ..., p]` where `W=128`. Invalid positions are marked `-1`.

2. **Compressed indices**: Vary by layer type:
   - **C1**: No compressed indices.
   - **C4 (with Indexer)**: The DSA Indexer selects top-k compressed token indices (see Section 1.6).
   - **C128 (static)**: All valid compressed tokens `[0, ..., ⌊(p+1)/128⌋ - 1]` are included. Future groups (where group_idx >= `(p+1)//ratio`) are masked to `-1`.

The concatenated `topk_idxs: [B, S, window_size + n_compress_tokens]` maps into the combined KV tensor. Compressed indices are offset by `S` (i.e., `compress_idx + seqlen_global`) to index into the concatenated KV.

### 1.6 DSA Indexer (C4 layers only)

The Indexer is a lightweight network that scores compressed token groups and selects the top-k most relevant ones for each query. It lives **inside** the C4 attention layer (not the compressor).

**Architecture** (`DSAIndexer`):
```
qr: [S, B, 1024]  (low-rank query, reused from main attention's q after q_norm)
x:  [S, B, 4096]  (hidden states)

# Query path
q = linear_wq_b(qr)               # TELinear(1024, 64*128), duplicated
q = reshape → [S, B, 64, 128]     # 64 indexer heads, 128-dim each
apply_rotary_emb(q[..., -64:])     # RoPE on last 64 dims
q = hadamard_transform(q)          # rotate activation

# Key path (uses its own compressor with ratio=4, rotate=True)
k = compressor(x)                  # DeepSeekV4Compressor(head_dim=128, ratio=4, rotate=True)
                                   # → [S/4, B, 128], with Hadamard rotation applied

# Score computation
weights = linear_weights_proj(x)   # TELinear(4096, 64) → [S, B, 64]
weights = weights * (64^-0.5) * (128^-0.5)

# Index scores via fused_qk_topk_naive:
#   scores[b,q,k] = sum_h(weights[b,q,h] * dot(q[b,q,h], k[b,k])) + mask
# → select top-512 compressed groups per query position
```

The indexer's compressor is separate from the main attention's compressor — it has `head_dim=128` (vs 512 for main) and applies Hadamard rotation (`rotate=True`) for FP8-friendly quantization.

A causal mask ensures queries cannot attend to future compressed groups: group `g` is valid for query at position `p` only if `g < (p+1) // 4`.

### 1.7 Sparse Attention Kernel

The attention uses **sparse gather-based** computation:

```
# q: [B, S, n_local_heads, 512]
# kv: [B, S + S/ratio, 512]       (shared across heads, single-head KV)
# topk_idxs: [B, S, topk]         (per-query selected KV positions)
# attn_sink: [n_local_heads]       (learnable per-head scalar, FP32)

kv_gathered = kv[:, topk_idxs]             # [B, S, topk, 512]
scores = einsum(q, kv_gathered) * scale    # [B, S, H, topk]
scores[topk_idxs == -1] = -inf             # mask invalid

# Online softmax with attention sink:
exp_scores = exp(scores - max)
numerator = einsum(exp_scores, kv_gathered)
denominator = sum(exp_scores) + exp(attn_sink - max)   # sink acts as a virtual token

o = numerator / denominator                # [B, S, H, 512]
```

The `attn_sink` is a learnable per-head bias that acts as a virtual "always-attend" token. It participates in the softmax denominator but contributes zero to the numerator, effectively allowing attention heads to "dump" probability mass when no real token is relevant.

### 1.8 Output Projection (Grouped LoRA)

The output uses a **grouped low-rank** projection:

```
# o: [B, S, n_local_heads, 512]
# Reshape to groups: [B, S, n_local_groups, heads_per_group * 512]
#   n_groups=8, so heads_per_group = 64/8 = 8

wo_a: [n_local_groups, o_lora_rank, heads_per_group * 512]  # ColumnParallel
o = einsum("bsgd,grd->bsgr", o, wo_a)     # → [B, S, n_local_groups, 1024]

wo_b: RowParallelLinear(8*1024, 4096)
output = wo_b(o.flatten(2))                # → [B, S, 4096]
```

### 1.9 RoPE Design

DeepSeek-V4 uses **YaRN** (Yet another RoPE eNtension) with different base frequencies for vanilla vs. compressed tokens:

| Component | RoPE base | Applied to dims |
|-----------|-----------|-----------------|
| Vanilla KV & Query | `10000` | Last 64 of 512 |
| Compressed KV (compressor output) | `160000` | Last 64 of 512 |
| Indexer Q & K | `160000` | Last 64 of 128 |

YaRN parameters: `factor=4`, `original_max_position_embeddings=65536`, `beta_fast=32`, `beta_slow=1`.

Compressed tokens receive RoPE at their group's representative position (stride by `ratio`), using the higher base frequency `160000` to encode coarser positional information.

An **inverse RoPE** is applied to the attention output `o[..., -64:]` after the sparse attention, undoing the positional encoding before the output projection.

---

## 2. Hyper-Connection (mHC)

DeepSeek-V4 replaces standard residual connections (`x + sublayer(x)`) with a **multi-stream Hyper-Connection** structure with `hc_mult=4` streams.

### 2.1 Structure

The hidden state has shape `[S, B, hc_mult, D]` = `[S, B, 4, 4096]` throughout the transformer block (expanded from `[S, B, D]` at the embedding layer).

Each sub-layer (attention, MLP) is wrapped with:

```
# PRE: mix 4 streams → 1 input for sublayer
y, post, comb = hc_pre(hidden_states, hc_fn, hc_scale, hc_base)
    # hidden_states: [B, S, 4, 4096]
    # 1. RMSNorm-style: rsqrt = 1/sqrt(mean(x²) + eps)
    # 2. Linear mixing: mixes = F.linear(x_flat, hc_fn) * rsqrt
    # 3. Sinkhorn split → pre[4], post[4], comb[4,4]
    #    (pre: input mixing weights, post: output routing, comb: residual mixing)
    # y = sum(pre_i * stream_i)  → [B, S, 4096]

# SUBLAYER (attention or MLP)
output = sublayer(y)

# POST: route output back to 4 streams + residual
hidden_states = hc_post(output, residual, post, comb)
    # stream_i = post_i * output + sum_j(comb_ij * residual_j)
    # → [B, S, 4, 4096]
```

### 2.2 Sinkhorn Normalization

The `comb` matrix (4x4) is made doubly-stochastic via Sinkhorn iterations (20 iterations, eps=1e-6):
1. Exponentiate: `comb = exp(comb - row_max)`
2. Alternating row/column normalization for `sinkhorn_iters` rounds
3. Result: each row and column approximately sums to 1

This ensures balanced information flow across streams.

### 2.3 Block Head

At the final layer, the 4 streams are collapsed back to 1 via `block_head`:
```
y = sum(sigmoid(scale * mix + base) * stream_i)  → [B, S, 4096]
```

### 2.4 Mixer Parameters

Per-layer HC parameters (`HCHeadParams`):
- `hc_head_fn`: [4, 4*4096] — mixing projection (FP32)
- `hc_head_base`: [4] — bias per stream (FP32)
- `hc_head_scale`: [1] — global scale (FP32)

The mixer computation runs in **no-grad mode** — only the actual sublayer (attention/MLP) receives gradients. The mixing weights are treated as a fixed (but learned) routing.

---

## 3. MoE with Hash-Based Routing

All 43 layers are MoE layers (`moe_layer_freq=[1,1,...,1]`).

### 3.1 Configuration

- 256 routed experts + 1 shared expert
- `topk=6` experts per token
- Score function: `sqrtsoftplus`
- Expert bias enabled (learned correction bias)
- Token dispatcher: AllToAll
- Scaling factor: 1.5

### 3.2 Hash Routing (first 3 layers)

The first `dsv4_n_hash_layers=3` layers use **hash-based deterministic routing**:

```python
# tid2eid: Parameter([vocab_size=129280, topk=6], dtype=int32)
# Lookup: O(1), no gradient, no learned gate
top_indices = tid2eid[input_ids]   # direct index by token ID
```

- `tid2eid` is a pre-computed lookup table mapping each token ID to its 6 expert indices
- No routing computation needed — purely deterministic
- Weights are still computed via the score function for the selected experts
- Bypasses routing replay (deterministic, so no need for replay during RL)

### 3.3 Learned Routing (layers 3–42)

Remaining layers use standard learned routing:
```
scores = score_fn(gate(x))         # sqrtsoftplus scoring
scores_for_routing = scores + expert_bias
top_indices = topk(scores_for_routing, k=6)
```

---

## 4. Other Details

### 4.1 Normalization

- **RMSNorm** throughout (eps=1e-6)
- Query uses per-head RMSNorm after `wq_b`: `q * rsqrt(q².mean(-1) + eps)` (no learnable weight)
- KV uses standard RMSNorm with learnable weight after `wkv`
- Compressor output uses RMSNorm with learnable weight

### 4.2 Activation

- **SwiGLU** for MLP
- FFN hidden size: 2048 (both routed and shared experts)

### 4.3 Attention Sink

Each attention layer has a learnable `attn_sink: [n_local_heads]` parameter (FP32). This acts as a virtual token in the softmax denominator — when no real KV token is relevant, attention probability flows to the sink rather than being forced onto irrelevant tokens.

### 4.4 Parallelism

- **TP** (Tensor Parallel): Shards heads across `wq_b`, `wo_a`, `wo_b`. KV is single-head (duplicated), then broadcast via `copy_to_tensor_model_parallel_region`.
- **CP** (Context Parallel): Zigzag partitioning of sequences. Vanilla KV and compressed KV are all-gathered across CP ranks before attention. Compressor overlap transform handles cross-CP-rank boundaries.
- **EP** (Expert Parallel): Via AllToAll token dispatcher for MoE.
- **SP** (Sequence Parallel): Gather before attention, scatter after.

---

## 5. Rebase Analysis: DeepSeek-V4 → GLM-5 Operators

### 5.1 Structural Comparison

| Aspect | DeepSeek-V4 (current) | GLM-5 |
|--------|----------------------|-------|
| **Data layout** | `[S, B, D]` / `[B, S, H, D]` (BSHD) | `[T, H, D]` (THD, packed varlen via `cu_seqlens`) |
| **Batch handling** | Explicit batch dim | `packed_seq_params.cu_seqlens_q/kv` |
| **KV structure** | Single-head `[B, S, 512]` | `[T, kv_group, 576]` (split D=512 + D_tail=64) |
| **RoPE format** | Complex-number based (`view_as_complex`, last 64 dims) | Interleaved (`x[...,0::2]/x[...,1::2]`), apex `fused_apply_rotary_pos_emb_thd` |
| **Attention kernel** | Sparse gather with `attn_sink` | Sparse gather, **no** `attn_sink` |
| **Output proj** | Grouped LoRA (`wo_a`+`wo_b`, 8 groups) + inverse RoPE | Absorbed MLA (`einsum thm,hdm→thd` + `linear_proj`) |
| **Indexer** | `DSAIndexer` in Megatron (compressor + Hadamard + `fused_qk_topk_naive`) | Inline (`wq_b`, `wk`, `k_norm`, `weights_proj` + `lighting_indexer` tilelang kernel) |
| **CP strategy** | Zigzag partitioning, `all_gather_cp_natural_order` + reorder | Standard `gather_from_sequence_parallel_region` on CP group |
| **Tilelang kernel shapes** | `q:[B,S,H,D]`, `kv:[B,S_kv,D]`, `idx:[B,S,topk]` | `q:[B,S,H,D+D_tail]`, `kv:[B,S_kv,G,D+D_tail]`, `idx:[B,S,G,topk]` |

**Key code locations**:
- GLM-5 attention: `miles_plugins/models/glm5/glm5.py` (`DSAMLASelfAttention`)
- GLM-5 tilelang fwd: `miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py`
- GLM-5 tilelang bwd: `miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py`
- GLM-5 indexer: `miles_plugins/models/glm5/ops/indexer.py` (`lighting_indexer` → `IndexerFunction`)
- GLM-5 sparse MLA wrapper: `miles_plugins/models/glm5/ops/sparse_mla.py` (`SparseMLA`)

### 5.2 Per-Component Rebase Analysis

#### 5.2.1 Data Format: BSHD → THD

**Feasibility**: Doable but invasive.

GLM-5 uses packed variable-length sequences (`cu_seqlens`) — fundamentally different from V4's padded `[B, S, ...]` format. V4's attention currently does `einops.rearrange(hidden_states, "s b d -> b s d")` at entry.

**Compressor conflict**: The compressor's `overlap_transform` assumes contiguous sequences within a batch — it reshapes `[B, S, ...]` into `[B, S/ratio, ratio, ...]` and shifts groups. In THD packed format, group boundaries could cross sequence boundaries. Handling per-sequence boundaries in the overlap is non-trivial and error-prone.

**CP conflict**: V4 uses zigzag CP with `all_gather_cp_natural_order` + `natural_to_zigzag_slice` (custom reordering). GLM-5 uses standard `gather_from_sequence_parallel_region`. These are fundamentally different CP strategies. The compressor's `overlap_transform_with_cp` explicitly handles zigzag reordering — switching CP methods requires rewriting all compressor CP logic.

**Recommendation**: Since the compressor stays as-is (Section 5.2.3), the data format around the compressor must also stay BSHD. Converting THD → BSHD before compressor and BSHD → THD after adds overhead and requires knowing `cu_seqlens` to unpack/repack. Not worth it unless there is a compelling perf reason.

#### 5.2.2 Tilelang Attention Kernels

**attn_sink gap (critical)**: GLM-5's kernel has no `attn_sink` support. V4's `attn_sink` modifies the softmax denominator:
```python
# V4 forward kernel:
sumexp[h] += exp2(attn_sink[h] * log2e - m_i[h] * sm_scale)
# V4 backward kernel:
dAttnSink computed via atomic_add
```
This changes softmax normalization and requires backward gradient flow. Must add:
1. `attn_sink` tensor to GLM-5 kernel signature
2. Modified online softmax accumulation in forward
3. `dAttnSink` gradient computation in backward (atomic adds)

**D/D_tail split**: GLM-5 splits Q/KV into `[D=512, D_tail=64]` with separate GEMMs for main and tail (RoPE) dimensions. V4 uses a single `D=512` with RoPE on the last 64 dims inline. Conceptually similar — V4 could adopt the split format since the last 64 dims are already the RoPE portion. The split allows the kernel to handle RoPE dims separately, potentially enabling different precision or fusion.

**Index shape**: GLM-5 uses `[B, S, kv_group, topk]` vs V4's `[B, S, topk]`. V4 has single-head KV (no `kv_group`). Trivial to adapt by setting `kv_group=1`.

**Variable topk**: V4's `topk = window_size + n_compress_tokens` varies by layer type (C1: 128 window only, C4: 128 + 512 indexer-selected, C128: 128 + S/128 static). GLM-5's kernel uses a fixed `topk`. The kernel itself doesn't care about the semantic meaning — just needs `topk % block_I == 0` — but the interface must handle varying topk per layer.

**Recommendation**: Fork GLM-5's kernel. Add `attn_sink` support (forward + backward). Keep `kv_group=1`. This gives the D/D_tail optimization and GLM-5's more mature kernel infrastructure.

#### 5.2.3 Compressor — Keep Current Implementation

The compressor (`DeepSeekV4Compressor`) is V4-specific: FP32 gated pooling with learned APE, overlap transform for C4, no equivalent in GLM-5. No rebase needed.

The compressor's CP handling (`overlap_transform_with_cp`) is tightly coupled to zigzag CP. If CP strategy changes (Section 5.2.1), this must be redesigned — but since we recommend keeping BSHD, this stays as-is.

#### 5.2.4 Indexer → GLM-5 `lighting_indexer`

**What GLM-5 offers**: Fused tilelang kernel (`lighting_indexer`) that computes QK scores, applies per-head weights, and extracts top-k in a single kernel — more efficient than V4's separate `fused_qk_topk_naive`. Also has tilelang backward kernels (`tilelang_indexer_fwd.py`, `tilelang_indexer_bwd.py`).

**Issues to fix when adopting**:

1. **Compressor in key path**: GLM-5's indexer uses `wk` (linear projection) for keys, producing full-length `index_k`. V4's indexer uses a compressor (`DeepSeekV4Compressor` with `head_dim=128`, `ratio=4`, `rotate=True`) producing `S/4` compressed keys. Must feed compressor output as `index_k` to `lighting_indexer`, not `wk(x)`. The kernel interface accepts arbitrary `seq_len_kv`, so `S/4` works.

2. **Hadamard rotation**: V4 applies `hadamard_transform` (from `fast_hadamard_transform`) to both q and k in the indexer. GLM-5's `lighting_indexer` doesn't include this. Must apply Hadamard rotation to q and k before calling the kernel.

3. **RoPE format mismatch**: GLM-5's indexer uses `fuse_rope` (interleaved, apex). V4's indexer uses `apply_rotary_emb` (complex-number, last 64 dims). See Section 5.2.5 for the general RoPE concern. Both use the same base frequency (`160000` for compressed), but format must be consistent with the main attention's RoPE.

4. **Variable-length masking**: GLM-5 uses `cu_seqlen_ks/ke` for causal masking. V4 uses a dense float mask (`_compute_indexer_mask` returning `[B, S, S/4]`). GLM-5's approach is more efficient but assumes THD packed format. If staying BSHD, you'd need to generate `cu_seqlens` from the batch or adapt the masking.

5. **Replay manager**: GLM-5's indexer doesn't have `indexer_replay_manager` integration. Must re-add for RL training determinism (registered via `indexer_replay_manager.register_to_module`). The replay manager overrides the topk function to produce deterministic routing during eval.

6. **KL loss**: V4's `DSAIndexer.forward_with_scores()` supports KL divergence loss between indexer scores and true attention scores. GLM-5's `lighting_indexer` returns `(index_score, topk_indices)` which can support this, but the loss computation pipeline must be preserved.

**Recommendation**: Use GLM-5's `lighting_indexer` kernel for better efficiency. Pre-process: compressor → Hadamard rotation → (adapt RoPE if needed). Post-process: re-add replay manager hooks and KL loss support.

#### 5.2.5 RoPE Implementation

**Format difference**: V4 uses complex-number RoPE (`view_as_complex(x.unflatten(-1, (-1, 2)))` → multiply by `freqs_cis` → `view_as_real`). GLM-5 uses interleaved RoPE (`x[..., 0::2]`, `x[..., 1::2]` → concatenate → `fused_apply_rotary_pos_emb_thd`).

These two formats produce different results unless the weight layout is matched. The model weights were trained with V4's complex-number RoPE — switching format requires either:
1. Verifying numerical equivalence with a reference forward pass
2. Or converting weights at load time

**Inverse RoPE constraint**: V4 applies inverse RoPE on the attention output (`o[..., -64:]`) before the output projection. This is tied to V4's non-absorbed grouped LoRA output design. GLM-5 doesn't need inverse RoPE because it uses absorbed attention (separate `q_no_pe`/`q_pos_emb` paths). If V4's grouped LoRA output is kept, inverse RoPE must be kept.

**Recommendation**: Risky to change. Keep V4's complex-number RoPE unless numerical equivalence is verified. RoPE format is deeply coupled to weight layout and affects all layers.

#### 5.2.6 Output Projection

V4 uses grouped LoRA: `einsum("bsgd,grd->bsgr", o, wo_a)` + `RowParallelLinear(wo_b)` with inverse RoPE before projection. GLM-5 uses absorbed MLA: `einsum("thm,hdm->thd", o, wv)` + `linear_proj`. These are architecturally different — V4's output structure is model-specific (8 groups, rank 1024), not an operator that can be swapped.

#### 5.2.7 Other Operators Worth Rebasing

**TELinear for all linear layers**: V4 uses `nn.Linear` for `wq_a` and `wkv` (with optional `TELinear` behind env var `MEGATRON_HACK_ENABLE_SEVERAL_TE_LINEAR`). GLM-5 consistently uses `TELinear` (via `build_module` with TE spec). Switching to `TELinear` everywhere enables FP8 training and better TE kernel fusion. Low risk, worth doing.

**TENorm for normalization**: V4 uses custom `RMSNorm` (from `ref_model.py`) for `q_norm` and `kv_norm`. GLM-5 uses Megatron's `TENorm`. Could switch for consistency and potential fused kernels. Low impact, low risk.

### 5.3 Recommendation: Module-Level vs Operator-Level Rebase

**Recommendation: Keep V4's attention module structure, replace individual operators.**

**Against full rebase to GLM-5's module**:

1. **Different MLA design**: GLM-5 uses absorbed attention (`q_no_pe @ w_kc → absorbed_q`), V4 does not. The output paths are completely different (absorbed `einsum thm,hdm→thd` vs grouped LoRA). This is a different algorithm, not an operator swap.

2. **Compressor creates asymmetry**: The compressor is V4-specific and tightly coupled to the attention flow (KV concatenation, index offset by `seqlen_global`, per-layer compress ratios C1/C4/C128). GLM-5's module has no concept of compressed tokens.

3. **Hyper-Connection**: V4's hidden states are `[S, B, 4, D]` due to mHC. The attention module entry/exit must handle HC streams. GLM-5's module interface doesn't support this.

4. **attn_sink**: Fundamental to V4's attention, absent in GLM-5. Requires kernel modification regardless of which module structure is used.

**Recommended operator-level replacements** (in priority order):

| Operator | Action | Risk | Benefit |
|----------|--------|------|---------|
| Tilelang attention kernel | Fork GLM-5's kernel, add `attn_sink`, adapt shapes | Medium — kernel changes need careful testing | Better D/D_tail split, more mature kernel |
| Tilelang indexer kernel | Use GLM-5's `lighting_indexer`, pre-process with compressor + Hadamard | Medium — must preserve replay + KL loss | Fused QK+topk, tilelang backward |
| `nn.Linear` → `TELinear` | Replace `wq_a`, `wkv` with `TELinear` | Low | FP8 support, TE fusion |
| `RMSNorm` → `TENorm` | Replace `q_norm`, `kv_norm` | Low | Consistency, potential fusion |
| RoPE | Keep V4's implementation | N/A | Avoids weight-format risk |
| CP strategy | Keep V4's zigzag CP | N/A | Avoids compressor redesign |
| Data format | Keep BSHD | N/A | Avoids compressor + HC incompatibility |

### 5.4 Risks and Mitigations

1. **Numerical drift**: Any operator swap (kernel, RoPE, norm) can cause subtle numerical differences that compound over training. **Mitigation**: Run a layer-by-layer forward-pass comparison (dump activations with the dumper, compare before/after rebase) on a short sequence before any training.

2. **Compressor + THD incompatibility**: If data format changes to THD, the overlap transform's group reshaping (`unflatten(1, (-1, ratio))`) would break at sequence boundaries in packed format. Off-by-one errors here are silent and catastrophic. **Mitigation**: Keep BSHD around the compressor, or add exhaustive unit tests with multiple sequences of varying lengths.

3. **Indexer topk divergence**: If the new `lighting_indexer` kernel produces even slightly different topk selections (due to precision, tiebreaking, or RoPE differences), training dynamics can shift since it determines which KV tokens are attended to. **Mitigation**: Compare indexer outputs on identical inputs before/after rebase. Verify topk overlap rate is >99%.

4. **RL replay incompatibility**: The replay manager records indexer topk decisions for deterministic replay during RL training. If `lighting_indexer`'s topk selection order differs from `fused_qk_topk_naive`, existing replay data becomes invalid. **Mitigation**: Regenerate replay data after rebase, or verify exact topk equivalence.

5. **Kernel correctness under attn_sink**: Adding `attn_sink` to GLM-5's kernel modifies the softmax denominator and backward gradient flow. A bug here would silently corrupt attention outputs. **Mitigation**: Unit test the modified kernel against V4's reference `sparse_attn_torch` implementation on random inputs, checking both forward outputs and all gradients (dq, dkv, dAttnSink).