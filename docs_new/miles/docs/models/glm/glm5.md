---
title: GLM5
description: Launch recipe for GLM-5 (744B-A40B) — Python launcher, 16+ node config.
---

# GLM5 / GLM5.1

## 1. Model Introduction

[GLM-5](https://huggingface.co/zai-org/GLM-5) is the most powerful language model in Zhipu AI's GLM series, scaling to 744 B parameters (40 B active) and integrating DeepSeek Sparse Attention (DSA) for long-context efficiency. [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) is the next-generation model for agentic engineering on top of GLM-5, sharing the same model architectures,

**Key highlights:**

- **Sparse MoE at frontier scale**: 744 B total / 40 B active per token, 256 routed experts top-8 + 1 shared.
- **MLA + DSA attention**: Multi-head Latent Attention (q-LoRA 2048 / kv-LoRA 512) combined with DeepSeek Sparse Attention to keep KV-cache cost low at long context.
- **Speculative decoding**: EAGLE/MTP rollout supported via `--enable-mtp`.
- **PD disaggregation**: prefill/decode disaggregation enabled by default for ≥1 node.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-5.1 | 40 B / 744 B | [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) |
| GLM-5 | 40 B / 744 B | [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) |

## 3. Environment Setup

### 3.1 Download model + datasets

The Python launcher's `prepare` subcommand handles download + dataset staging:

```bash
python scripts/run_glm5_744b_a40b.py prepare --model-name GLM-5 --num-nodes 16
```

### 3.2 HF → Megatron `torch_dist` conversion

Also handled by `prepare`. The launcher patches `config.json` to set `model_type=deepseek_v32` (`_process_glm_checkpoint`) before conversion — GLM-5 is loaded through the DeepseekV32 architecture path. Run `prepare-cp` afterwards on every node to copy the converted checkpoint from shared NFS to local disk.

## 4. Launch

### 4.1 Quick start

```bash
python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5 --num-nodes 16
```

The Typer app exposes four subcommands:

```bash
# Full pipeline: download, convert, copy, train
python scripts/run_glm5_744b_a40b.py full-train --model-name GLM-5 --num-nodes <N>

# Just download model + datasets and convert to Megatron
python scripts/run_glm5_744b_a40b.py prepare    --model-name GLM-5 --num-nodes <N>

# Copy converted checkpoint from shared NFS to local disk (run on every node)
python scripts/run_glm5_744b_a40b.py prepare-cp --model-name GLM-5 --num-nodes <N>

# Train only (assumes prepare/prepare-cp done)
python scripts/run_glm5_744b_a40b.py train      --model-name GLM-5 --num-nodes <N>
```

The launcher's docstring says it's tested on **H200 / B200 / GB300**; the dataclass restricts `--hardware` to `{H200, B200, GB300}`.

## 5. Recipe Configuration

### 5.1 Parallelism

Verbatim from `_execute_train`, `--num-nodes ≥ 16` branch:

| TP | PP | CP | EP | expert-TP | `decoder-last-pipeline-num-layers` | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 2 | 32 | 1 | 18 | 16384 | ≥ 128 (≥ 16 × 8) |

Plus `--use-dynamic-batch-size`, `--data-pad-size-multiplier 4096`, `--log-probs-chunk-size 1024`, `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`. R3 (`--use-rollout-routing-replay`) is **not** enabled by default.

### 5.3 Rollout & SGLang

Always-on flags:

```bash
--sglang-mem-fraction-static 0.70
--sglang-enable-dp-attention
--sglang-ep-size <world_size>
--sglang-dp-size <world_size>
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

# DSA / NSA attention
--sglang-page-size 64
--sglang-nsa-decode-backend flashmla_sparse
--sglang-nsa-prefill-backend flashmla_sparse
--sglang-attention-backend nsa

--sglang-max-running-requests 512
--sglang-watchdog-timeout 3600
```

### 5.4 Optimizer

`--enable-optimizer-offload` adds `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` (opt-in).

### 5.5 Notable quirks

The launcher exposes these as flags:

- `--fp8-rollout` — runs `tools/convert_hf_to_fp8.py --strategy block --block-size 128 128` and feeds the FP8 directory to SGLang (Megatron stays BF16).
- `--enable-mtp` — adds SGLang EAGLE speculative decoding (`--sglang-speculative-{algorithm,num-steps,eagle-topk,num-draft-tokens}`).
- `--enable-pd` (default `True` for ≥1 node) — enables prefill/decode disaggregation; with PD the launcher uses larger SGLang world sizes (16 for `<16` nodes, 64 for `≥16` nodes).
- `--use-deepep` (default `True`) — enables Megatron-side DeepEP (`--moe-enable-deepep --moe-token-dispatcher-type flex`); falls back to `alltoall`. Forced off on GB300.

## 6. Pairs Well With

- [PD Disaggregation](../../advanced/pd-disaggregation.md) — on by default for `num_nodes ≥ 1`.
- [FP8 & Low Precision](../../advanced/fp8-low-precision.md) — opt-in via `--fp8-rollout`.
- [Speculative Decoding](../../advanced/speculative-decoding.md) — opt-in via `--enable-mtp`.
