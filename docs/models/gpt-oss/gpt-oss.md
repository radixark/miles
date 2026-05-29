---
title: GPT-OSS 20B
description: Two launchers — Megatron BF16 (8 GPU, mbridge) and FSDP (4 GPU, dequantizes MXFP4 → BF16 first).
---

# GPT-OSS 20B

## 1. Model Introduction

[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) is OpenAI's open-weight language model, designed for reasoning, agentic tasks, and developer use cases. miles supports the 20 B variant.

**Key highlights:**

- **Configurable reasoning effort**: low / medium / high reasoning effort selectable per request.
- **Full chain-of-thought**: the reasoning trace is exposed and trainable.
- **MXFP4 native weights**: the HF checkpoint ships in MXFP4 (post-trained) — the FSDP launcher dequantizes to BF16 first; the BF16 launcher uses mbridge to load HF directly.
- **Sink attention**: requires `--qkv-format bshd` on the Megatron path, which precludes dynamic batch sizing.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| gpt-oss-20b | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
# Megatron BF16 path — stage everything under /root/shared (script's hardcoded BASE_DIR)
hf download openai/gpt-oss-20b --local-dir /root/shared/gpt-oss-20b
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/shared/dapo-math-17k

# FSDP path — script downloads + dequantizes automatically; only the dataset is needed
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

### 3.2 HF → Megatron `torch_dist` conversion

Neither launcher needs a `convert_hf_to_torch_dist.py` step. The Megatron BF16 launcher loads the HF checkpoint directly via `--megatron-to-hf-mode bridge` (mbridge); the FSDP launcher loads HF directly.

The FSDP launcher additionally runs an inline `convert_model.py` snippet that downloads `openai/gpt-oss-20b` and dequantizes its MXFP4 weights to BF16, saving to `/root/models/gpt-oss-20b-bf16`.

## 4. Launch

### 4.1 Quick start

```bash
# Megatron BF16 (1 node × 8 GPU)
cd /root/miles
bash scripts/run-gpt-oss-20b-bf16.sh

# FSDP (1 node × 4 GPU; restricts to GPUs 4-7 via CUDA_VISIBLE_DEVICES)
bash scripts/run-gptoss-20b-fsdp.sh
```

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | `micro-batch-size` | GPUs |
|---|---|---|---|---|---|---|---|
| `run-gpt-oss-20b-bf16.sh` (Megatron) | 8 | 1 | 1 | 8 | 1 | 1 | 8 (1 × 8) |
| `run-gptoss-20b-fsdp.sh` (FSDP) | 1 | – | – | – | – | – | 4 (1 × 4) |

`--use-dynamic-batch-size` is **not** used on the Megatron BF16 path — the script's comment explains: `--qkv-format bshd` (required for sink attention with TE) is incompatible with dynamic batch size. Only `--micro-batch-size 1` is set. `--sequence-parallel` is on (required for TP + EP).

The FSDP variant passes `--train-backend fsdp --bf16 --attn-implementation eager` to `train.py`.

### 5.2 Algorithm

Both scripts use GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00`. **`--use-kl-loss` is commented out in both scripts** (the BF16 script's comment notes "need gpt oss ckpt conversion" before KL can be enabled).

Note that `--rm-type` differs: `math` for the Megatron BF16 path, `deepscaler` for FSDP.

### 5.3 Rollout & SGLang

`run-gpt-oss-20b-bf16.sh`:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
   --sglang-mem-fraction-static 0.70
)
```

`run-gptoss-20b-fsdp.sh`:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-tensor-parallel-size 1
   --sglang-dtype bfloat16
   --sglang-decode-log-interval 1000
)
```

### 5.4 Optimizer

The Megatron BF16 launcher enables CPU Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`); the FSDP launcher does not.

### 5.5 Notable quirks

`MISC_ARGS` in the BF16 script:

```bash
--attention-dropout 0.0
--hidden-dropout 0.0
--qkv-format bshd        # required for TE sink attention (SWA + learnable softmax offset)
--attention-backend fused
```

`--qkv-format bshd` is mandated by the sink-attention pattern; in turn it precludes `--use-dynamic-batch-size`. Don't toggle either flag without the other.

Neither launcher writes `--save`/`--load`/`--save-interval`.

## 6. Pairs Well With

- [Backends Beyond Megatron](../../advanced/architecture-support.md) — the FSDP variant.
- [Low Precision RL](../../advanced/fp8-low-precision.md)
