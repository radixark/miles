# MiniMax-M3 (MSA) miles plugin

Training support for **MiniMax-M3** in miles, built the same way as the GLM-5
plugin (`miles_plugins/models/glm5/`): a custom attention module + sparse-index
kernels in `ops/`, wired through Megatron's `--spec` mechanism.

```
--spec "miles_plugins.models.minimax_m3.minimax_m3" "get_minimax_m3_spec"
```

## Why M3 needs its own plugin (it is *not* M2, and *not* GLM-5)

| | M2 (full attn) | GLM-5 (DSA) | **M3 (MSA)** |
|---|---|---|---|
| attention backbone | GQA, dense | **MLA** (latent KV) | **GQA** (real KV) |
| sparse selection | none | per-**token**, topk=2048 | per-**block**, 16×128 |
| index sharing | — | one index, all q-heads | one index per GQA group (4 heads) |
| head reduction | — | sum (learned `weights_proj`) | **max** (`amax`, no gating weight) |
| block-forced keep | — | — | init block 0 + local block |
| QK-norm | full-dim | — | **per-head** |

So M3 reuses neither the M2 bridge (dense GQA) nor GLM-5's MLA path. The novel
pieces are the **block-level lightning indexer** and **block-sparse GQA**.

## Architecture (from `MiniMaxAI/MiniMax-M3` config.json, arXiv:2606.13392)

- 60 layers, hidden 6144, 64 q-heads / 4 kv-heads, head_dim 128.
- partial RoPE on first `rotary_dim=64`, θ = 5e6.
- **layers 0–2**: dense FFN (intermediate 12288), **full causal attention**.
- **layers 3–59**: MoE (128 experts, top-4, 1 shared, sigmoid + routing bias,
  `routed_scaling_factor=2.0`, expert FFN 3072) + **MSA sparse attention**.
- indexer: `sparse_index_dim=128`, `sparse_num_index_heads=4`.
- selection: `sparse_block_size=128`, `sparse_topk_blocks=16`,
  `sparse_score_type="max"`, `sparse_init_block=0`, `sparse_local_block=1`.

## MSA algorithm (what the code implements)

Per query token `i` on a sparse layer:

1. `index_q = q_norm(Wq_idx·x)` → `[N, 4, 128]`, `index_k = k_norm(Wk_idx·x)` → `[N, 128]`
   (one shared key head per GQA group); partial RoPE on the first 64 dims.
2. token logits `S[h,i,j] = <index_q[i,h], index_k[j]>/√128`, causal `j ≤ i`.
3. block max-pool `M[i,b] = max_h max_{j∈block b} S[h,i,j]`  (`score_type=max`).
4. force-keep init block(s) (sink) + the local/current block.
5. `TopK_b(M[i,·], 16)` → selected blocks; main GQA attends only to those
   (≤ 16·128 = 2048 keys), causally masked inside each block.

Files:
- `ops/msa_indexer.py` — steps 1–5. `block_topk` computes token logits
  chunk-by-chunk (bounded memory at 1M ctx), then block-pool + top-k.
  **Status:** it currently runs the pure-torch reduction; GLM-5's tilelang
  `tilelang_indexer_fwd` is imported only as an availability gate and is **not yet
  wired in** (TODO). `block_topk_reference` is the unchunked pure-torch oracle.
- `ops/block_sparse_attn.py` — block-sparse GQA over the selected blocks.
  `BlockSparseGQA` is the autograd hook (same call shape as GLM-5's `SparseMLA`).
- `minimax_m3.py` — `MSASelfAttention` (GQA + indexer) and `get_minimax_m3_spec`
  (dense layers keep full attention; sparse layers swap in MSA).

## Indexer training (important)

The top-k is non-differentiable, so the indexer gets **no** LM gradient — it is
detached in `_compute_block_selection`. M3 trains it with a **KL distillation**
loss: the indexer's softmax over selected tokens matches the main attention's
per-group averaged distribution (`D_KL(stopgrad(P_main) ‖ P_index)`), with a
two-stage warmup (full attention first, then switch to sparse). Add this auxiliary
loss in the training loop to fine-tune the indexer; for pure SFT on top of the
released weights the detached indexer is used as-is.

## Weight loading (the bridge)

Training on the Megatron backend needs two pieces: the **compute graph** (this
`--spec`) and the **weight bridge** (HF→Megatron param mapping). The bridge is
[`miles_plugins/megatron_bridge/minimax_m3.py`](../../megatron_bridge/minimax_m3.py),
registered for `MiniMaxM3VLForConditionalGeneration`, subclassing
`MiniMaxM2Bridge` (same MiniMax MoE lineage) and grounded in the real HF keys
(`language_model.model.*`, `block_sparse_moe`, `index_{q,k}_{proj,norm}`,
`shared_experts`, `w1/w2/w3`). Only the text backbone is mapped; VL checkpoint
vision weights are simply not requested for a text-only `GPTModel`.

## Status / what's verified vs TODO

**Complete & CPU-verified:** indexer projections (q+k norm), block max-pool,
forced init/local blocks, top-k selection, `layer_types`-driven dense/sparse
split, MoE/router config, per-head QK-norm, partial RoPE, spec wiring, weight
bridge, launch script. The MSA reference is unit-tested (invariants hold;
all-blocks-selected == dense causal GQA, err 0).

**Needs on-GPU validation:** exact Megatron-path ↔ HF-key matching in the bridge
(`bridge.load_hf_weights`), and `MSASelfAttention`'s integration with the TE
QKV/RoPE helpers (written to spec, unrun).

**Reference (correct, not yet fused):** `block_sparse_attention_reference`
materialises a dense block mask — O(N²) memory, fine for ≤ ~32k smoke tests and
as the numerical oracle. The **only** kernel TODO is a fused varlen block-sparse
GQA fwd/bwd (flash-attn block-mask or a tilelang kernel mirroring GLM-5's
`sparse_mla`); drop it into `BlockSparseGQA.forward/backward` with no other
changes.

## VL (multimodal) — the "less-efficient e2e" path

Enable with `--minimax-m3-vl` (plus the text `--spec`). Rather than reimplement
M3's Qwen-VL-family tower in parallel Megatron (Conv3d, 3D-RoPE, dynamic res,
2×2 merge, video — *not* in Megatron core, which only has CLIP/RADIO 2D ViT),
this follows Megatron-Bridge's **Kimi-K2.5-VL template**: run the **HF-native
vision tower + projector** (replicated across TP, frozen-friendly) and feed its
embeddings into the Megatron M3 text decoder.

Pieces (all in this plugin + 3 small gated miles-core hooks):
- [`vl_model.py`](vl_model.py) — `MiniMaxM3VLModel` composite. forward:
  `embed(text)` → `projector(vision_tower(pixel_values))` → scatter at
  image/video placeholders → `language_model(decoder_input=merged)`. The LM is
  the GPTModel built by `get_minimax_m3_spec` (MSA+MoE). `build_minimax_m3_vl()`
  loads the vision tower/projector from the HF checkpoint.
- [`mm_data.py`](mm_data.py) — expands one media placeholder (ids 200025/200026)
  into `(H/2)*(W/2)` slots so placeholder count == vision-token count.
- bridge ([`../../megatron_bridge/minimax_m3.py`](../../megatron_bridge/minimax_m3.py))
  switches LM param prefix to `language_model.` and adds vision
  `ReplicatedMapping` when `MINIMAX_M3_VL=1` (set by `build_minimax_m3_vl`).
- miles-core hooks (gated on `--minimax-m3-vl`, default off → text path
  untouched): `model_provider.py` wraps the GPTModel; `data.py` runs token
  expansion before lengths are read; `utils/arguments.py` declares the flag.
  The forward seam (`model.py` unpacking `multimodal_train_inputs`) already
  exists in miles.

**Verified (CPU):** `merge_vision_into_text` places vision embeds exactly at
placeholder rows (and guards count mismatch); `expand_media_tokens` expands +
re-masks correctly. **Needs GPU:** the HF vision tower load + full forward, and
the bridge param-path match (esp. the `language_model.` prefix + vision
ReplicatedMapping). Vision is replicated (less efficient) — swap in a parallel
tower later behind the same `vision_tower` interface. For RL, weight-sync of
vision/projector back to the rollout engine (a `megatron_to_hf` converter, à la
miles' `kimi_vl.py`) is still TODO; SFT does not need it. CP of the sparse path
is still CP=1 (TP/EP unaffected).

## Quick correctness check

```python
import torch
from miles_plugins.models.minimax_m3.ops.msa_indexer import (
    block_topk_reference, block_topk)
from miles_plugins.models.minimax_m3.ops.block_sparse_attn import (
    block_sparse_attention_reference)

N, Hidx, d = 600, 4, 128
cu = torch.tensor([0, 256, 600], dtype=torch.int32, device="cuda")
iq = torch.randn(N, Hidx, d, device="cuda", dtype=torch.bfloat16)
ik = torch.randn(N, d, device="cuda", dtype=torch.bfloat16)
blk = block_topk_reference(iq, ik, cu, block_size=128, topk_blocks=16)
assert blk.shape == (N, 16)
# block 0 (sink not forced here, init=0) and the local block must be present:
q = torch.randn(N, 64, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(N, 4, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(N, 4, 128, device="cuda", dtype=torch.bfloat16)
o = block_sparse_attention_reference(q, k, v, blk, cu, block_size=128)
assert o.shape == (N, 64, 128)
print("ok")
```

## Known limitations (from code review 2026-06; honest status)

The text path (MSA spec, bridges, HF converters) is exercised and the HF↔Megatron
conversions are bit-exact. The **VL/vision path is NOT yet correct for real vision
flow** — prior "e2e VLM" SFT runs trained the LM on image-placeholder *tokens* with
the vision tower loaded but never exercised (see below). Open items:

1. **VL merge is not sequence-parallel-correct.** Under `--sequence-parallel`
   (every VL launch script sets it) the LM embedding returns sequence-scattered
   `[t/TP, b, h]`, but `merge_vision_into_text` builds a full-length mask. With real
   `pixel_values` this now raises a clear error (added guard); the correct fix is to
   gather the embedding to full length, merge, then re-scatter before the decoder.
   Until then, run vision without SP (small models) or implement the gather/scatter.
2. **Vision/projector are not actually trained.** They are native HF modules,
   replicated, and are NOT placed in Megatron's grad buffer / distributed optimizer,
   so they receive no optimizer update (verified via `M3_VL_PROBE`: weight norms are
   constant across steps). `--minimax-m3-train-vision` therefore does not (yet) work;
   there is also no TP grad all-reduce for these replicated params.
3. **Checkpoint save drops vision/projector.** `sharded_state_dict` delegates only to
   the LM, so any trained projector/tower is not saved; `merge_m3_vision_into_hf.py`
   re-attaches the *original* (untrained) vision weights. Fine while (2) holds (nothing
   trains), but must be fixed alongside (2)/(1) for a real trained-VL checkpoint.
4. **Indexer is frozen.** It is detached (no LM grad) and the KL-distillation loss
   described above is NOT implemented in this plugin — index_* params get no gradient.
   Correct for SFT on released weights; do not expect indexer adaptation.
5. **`sparse_init_block` is a count, not an index.** The force-keep loop is
   `range(min(init_blocks, nb))` with default `init_blocks=0`, so with the released
   config value the attention-sink block is not force-kept. Harmless for ≤16-block
   smoke seqs (all blocks selected); verify against the HF MSA reference + the real
   config value before long-context training.
6. **`_load_vision_only` is lenient** (`strict=False`, warn-only on missing keys):
   a bad checkpoint→module rename yields a partially random-init tower with only a
   warning. Consider failing hard once the rename is GPU-validated.
