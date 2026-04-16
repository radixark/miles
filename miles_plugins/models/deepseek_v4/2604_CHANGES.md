# DeepSeek-V4 2604 Checkpoint Changes (vs 2601)

This document describes the architectural and weight format differences between the old (2601, January 2026) and new (2604, April 9 2026) DeepSeek-V4 checkpoints.

## Summary

The 2604 checkpoint is not a new architecture — it is the same DeepSeek-V4 model with:
1. More aggressive quantization (FP4 experts, FP8 scales → ~149G vs ~276G)
2. RoPE factor increased 4x for longer context (1M)
3. All layers are MoE (no dense replacement layers)
4. `wo_a` changed from BF16 to FP8
5. No tool calling / agent support — user dialogue and reasoning only

---

## 1. Config Changes

Configs compared: `sunrise/reference_implementation_updated_old/config_285B.json` (2601) vs `sunrise/official_code_0409/code/config.json` (2604).

| Parameter | 2601 | 2604 | Notes |
|-----------|------|------|-------|
| `rope_factor` | 4 | 16 | 4x increase for 1M context |
| `expert_dtype` | *(absent)* | `"fp4"` | New field — routed experts use FP4 |
| `max_seq_len` | *(absent)* | 61000 | New field — explicit max seq len |
| `compress_ratios` | 43 entries (ends with `4`) | 44 entries (ends with `4, 0`) | Extra trailing 0 for final layer |

Everything else is identical: 43 layers, 256 routed experts + 1 shared, topk=6, head_dim=512, etc.

---

## 2. Weight Format Changes

Tensor shape files: `sunrise/debug_info/tensor_shapes_output_{2601,2604}.txt`.

### 2.1 Checkpoint Size

| | 2601 | 2604 |
|---|---|---|
| Total tensors | 69,145 | 69,187 (+42) |
| Total size | 275.81 GB | 148.65 GB (**-46%**) |

### 2.2 Expert Weights (routed, per-expert)

| Tensor | 2601 | 2604 |
|--------|------|------|
| `w1.weight` | `float8_e4m3fn [2048, 4096]` | `int8 [2048, 2048]` (FP4 packed) |
| `w1.scale` | `float32 [16, 32]` | `float8_e8m0fnu [2048, 128]` |
| `w2.weight` | `float8_e4m3fn [4096, 2048]` | `int8 [4096, 1024]` (FP4 packed) |
| `w2.scale` | `float32 [32, 16]` | `float8_e8m0fnu [4096, 64]` |
| `w3.weight` | `float8_e4m3fn [2048, 4096]` | `int8 [2048, 2048]` (FP4 packed) |
| `w3.scale` | `float32 [16, 32]` | `float8_e8m0fnu [2048, 128]` |

The FP4 weights are stored as `int8` (two FP4 values packed per byte). Scale dtype changed from `float32` to `float8_e8m0fnu` (block-wise microscaling exponent).

### 2.3 Attention Output Projection (`wo_a`)

| Tensor | 2601 | 2604 |
|--------|------|------|
| `wo_a.weight` | `bfloat16 [8192, 4096]` (64 MB) | `float8_e4m3fn [8192, 4096]` (32 MB) |
| `wo_a.scale` | *(absent)* | `float8_e8m0fnu [64, 32]` (per 128x128 block) |

At load time in SGLang, 2604's FP8 `wo_a` is dequantized back to BF16 (see `_dequant_fp8_wo_a` in `deepseek_v4.py:2646`).

### 2.4 Other Attention Scales

| Tensor | 2601 | 2604 |
|--------|------|------|
| `wkv.scale` | `float32` | `float8_e8m0fnu` |
| `wo_b.scale` | `float32` | `float8_e8m0fnu` |

Shared expert weights remain FP8 (unchanged).

---

## 3. Code Branching (`SGLANG_DSV4_MODE`)

All runtime differences are gated by `SGLANG_DSV4_MODE` env var (`"2601"` or `"2604"`). This env var **must** be set — the server crashes with `NotImplementedError` otherwise.

Key branches in `NightFall/python/sglang/srt/models/deepseek_v4.py`:

### 3.1 Attention Dimensions (lines 1178–1200)

```python
# 2604: compute qk_nope_head_dim from head_dim
# 2601: read qk_nope_head_dim from config directly
if SGLANG_DSV4_MODE == "2604":
    qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim  # 512 - 64 = 448
    assert head_dim == config.head_dim
else:
    qk_nope_head_dim = config.qk_nope_head_dim
    assert head_dim == config.v_head_dim
```

### 3.2 RoPE Factor (lines 1241–1246)

```python
if SGLANG_DSV4_MODE == "2604":
    assert rope_scaling["factor"] == 16
elif SGLANG_DSV4_MODE == "2601":
    assert rope_scaling["factor"] == 4
```

### 3.3 MoE Layer Sparsity (lines 1680–1685)

```python
# 2604: all 43 layers are MoE (no dense layers)
# 2601: some early layers may be dense, controlled by config
if SGLANG_DSV4_MODE == "2604":
    first_k_dense_replace = 0
    moe_layer_freq = 1
else:
    first_k_dense_replace = config.first_k_dense_replace
    moe_layer_freq = config.moe_layer_freq
```

### 3.4 APE Hotfix (line 246)

The compressor's Absolute Positional Embedding parameter reordering is only applied to 2601. The 2604 checkpoint does not need this hotfix.

### 3.5 Shared Expert Fusion (lines 2097–2100)

2604 with FP4 experts cannot use shared expert fusion, because routed experts are FP4 while shared experts remain FP8. Fusing them would incorrectly apply FP4 dequant to shared experts.

### 3.6 MTP HC Hidden (lines 2014–2017, 2133–2134)

2604 has special handling for MTP (Multi-Token Prediction) Hyper-Connection hidden states, gated by `SGLANG_FIX_MTP_HC_HIDDEN` (should be enabled for 2604).

### 3.7 Weight Loading (lines 2253–2291)

- 2604 enforces config file validation against reference (`_debug_assert_model_path_configs`)
- 2604 FP8 `wo_a` weights are dequantized to BF16 at load time (unless `SGLANG_OPT_FP8_WO_A_GEMM` is enabled)
- 2604 with FP4 experts: dequant FP8 `wo_a`; without FP4: drop stale `wo_a.scale`

---

## 4. Reference Implementation Differences

Comparing `sunrise/reference_implementation_updated_old/model.py` (2601) vs `sunrise/official_code_0409/code/model.py` (2604):

| Aspect | 2601 | 2604 |
|--------|------|------|
| `ModelArgs.expert_dtype` | not present | `Literal[None, "fp4"] = None` |
| `ModelArgs.n_mtp_layers` | present (default 0) | present (default 1) |
| `scale_dtype` global | separate `scale_dtype` variable | uses `Linear.scale_fmt` attribute |
| `Transformer.head` | `ParallelHead` (norm + HC + linear) | `ColumnParallelLinear` + separate `hc_head()` method |
| `Transformer.mtp` | `ModuleList` of MTPBlocks | *(absent from reference code)* |
| `Attention.forward` | `forward(x, start_pos)` | `forward(x, start_pos, debug_return_kv=False, debug_return_topk=False)` |
| FP4 GEMM support | not present | `fp4_gemm()` function added |
| Dumper integration | basic `dumper.dump()` | adds `dumper.override_enable()` for selective layer logging |

---

## 5. Hardware Constraints

- **H200 (SM90)**: Does not support native FP4 GEMM. Must convert FP4 weights to FP8 first (see Section 6).
  Use `SGLANG_HACK_SKIP_FP4_FP8_GEMM=1` for smoke testing only (produces garbled output).

- **B200/GB300 (SM100)**: Supports native FP4 GEMM. Can run the FP4 checkpoint directly with `SGLANG_DSV4_FP4_EXPERTS=1`.

---

## 6. FP4 to FP8 Conversion

The FP8 converted checkpoint is needed for:
- **H200 inference** (no native FP4 GEMM support)
- **Miles RL training pipeline** (training backend operates on FP8)

### 6.1 Conversion Command

```bash
python sunrise/convert_fp4_to_fp8.py \
  --input-dir /data/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP4 \
  --output-dir /data/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP8-cvt
```

Pure CPU computation (no GPU required). Processes 46 safetensors files sequentially. Requires `torch`, `safetensors`, `tqdm`.

### 6.2 What the Conversion Does

| Component | FP4 (input) | FP8 (output) |
|-----------|-------------|--------------|
| Routed expert weights | `int8 [N, K/2]` (packed FP4) + `e8m0fnu [N, K/32]` scale | `float8_e4m3fn [N, K]` + `float32 [N/128, K/128]` scale |
| `wo_a` weights | `float8_e4m3fn` + `e8m0fnu` scale | `bfloat16` (dequantized, scale dropped) |
| Other e8m0fnu scales | `float8_e8m0fnu` | `float32` (same shape) |
| Everything else | unchanged | unchanged |

Output checkpoint is ~276G (vs 149G FP4) — format is compatible with the 2601 loading path.

### 6.3 Checkpoint Versions

Multiple config versions exist (`v1`, `v2`, `v3`, `v4`). The safetensors weights are identical across all versions — only config/metadata files differ. `v4` is the latest, adding `n_group` and `topk_group` fields to config.json.

The SGLang code validates configs via `_debug_assert_model_path_configs()` against reference files in `sunrise/assembled_hf_config_0409/{version}/`. Set `SGLANG_HACK_ASSERT_CKPT_VERSION=v4` if using v4 checkpoint (default is `v1`).

---

## 7. Launching SGLang Server

### 7.1 FP4 Checkpoint (B200/GB300 — native FP4 GEMM)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_DSV4_MODE=2604 \
SGLANG_DSV4_FP4_EXPERTS=1 \
SGLANG_FIX_MTP_HC_HIDDEN=1 \
SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
SGLANG_ENABLE_THINKING=1 \
python3 -m sglang.launch_server \
  --model-path /cluster_public/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP4/ \
  --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention \
  --moe-a2a-backend deepep \
  --max-prefill-tokens 1024 --chunked-prefill-size 1024 \
  --cuda-graph-max-bs 32 --mem-fraction-static 0.7 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

B200 IB workaround (most HCA cables unplugged — only GPU 4-7 work without this):
```bash
NVSHMEM_ENABLE_NIC_PE_MAPPING=1 NVSHMEM_HCA_LIST=mlx5_7,mlx5_8,mlx5_9,mlx5_10
```

### 7.2 FP8 Converted Checkpoint (H200 / Miles RL training)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_DSV4_MODE=2604 \
SGLANG_FIX_MTP_HC_HIDDEN=1 \
SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT=1 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
SGLANG_ENABLE_THINKING=1 \
python3 -m sglang.launch_server \
  --model-path /cluster_public/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP8-cvt/ \
  --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention \
  --moe-a2a-backend deepep \
  --max-prefill-tokens 1024 --chunked-prefill-size 1024 \
  --mem-fraction-static 0.7 \
  --skip-server-warmup \
  --host 0.0.0.0 --port 30000
```

Key differences from FP4 launch:
- **No** `SGLANG_DSV4_FP4_EXPERTS` (experts are already FP8)
- **No** `SGLANG_HACK_SKIP_FP4_FP8_GEMM` (real FP8 GEMM runs)
- `wo_a` is already BF16 — SGLang's `load_weights` drops stale `wo_a.scale` if present

### 7.3 Quick Smoke Test

```bash
curl -s --max-time 60 http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 20}'
```

Non-garbled output confirms the checkpoint loaded correctly.

### 7.4 MTP (EAGLE Speculative Decoding)

Add to the server launch command:
```bash
--speculative-algo EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

Note: with EAGLE MTP + DeepEP, `num_max_dispatch_tokens_per_rank` is hard-limited to 1024. Use `--cuda-graph-max-bs 256 --max-running-requests 256` to avoid assertion failures.

---

## 8. Required Environment Variables

```bash
# Must set — selects code path
SGLANG_DSV4_MODE=2604

# Performance (recommended, pending full verification)
SGLANG_OPT_DEEPGEMM_SCALE_CONVERT_AT_INIT=1
SGLANG_FIX_MTP_HC_HIDDEN=1

# B200/GB300 with native FP4 (omit for FP8 converted checkpoint)
SGLANG_DSV4_FP4_EXPERTS=1

# B200 DeepEP IB workaround (most HCA cables unplugged)
NVSHMEM_ENABLE_NIC_PE_MAPPING=1
NVSHMEM_HCA_LIST=mlx5_7,mlx5_8,mlx5_9,mlx5_10
```

---

## 9. Weight Paths

| Machine type | 2601 (old) | 2604 FP4 (native) | 2604 FP8 (converted) |
|-------------|------------|-------------------|---------------------|
| Baremetal | `/data/weights/hello2026` | `/data/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP4/` | `/data/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP8-cvt` |
| K8s (gcp-gb300) | `/cluster_public/weights/hello2026` | `/cluster_public/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP4/` | `/cluster_public/weights/ckpt20260409_v4/DeepSeek-V4-HF-FP8-cvt` |
