---
title: Gemma-4
description: Launch recipes for Gemma-4 26B-A4B-it (MoE) and 31B-it (dense) via the HF to Megatron bridge.
---
## 1. Model Introduction

[Gemma-4](https://huggingface.co/google) is Google's multimodal model line.
miles trains both released instruction-tuned sizes as language models, on the
base VLM checkpoint directly.

Both go through the HF to Megatron bridge (`--megatron-to-hf-mode bridge`), and
on the rollout side sglang runs `Gemma4ForConditionalGeneration`, which loads
Gemma-4's hybrid `head_dim` weights correctly. There is no offline `torch_dist`
conversion and no LLM-view rewrite of the checkpoint.

**Key highlights:**

- **Two shapes, one recipe family**: 26B-A4B is MoE (128 experts, top-8), 31B is
  dense. They differ mainly in expert parallelism and the token budget.
- **Bridge-mode load** straight from the VLM checkpoint.
- **Tied embeddings**: neither config passes
  `--untie-embeddings-and-output-weights`.
- **Single node**: both recipes target 8 × H200.

## 2. Supported Variants

| Model | Class | Active / Total | Layers | Hidden | HF ID |
|---|---|---|---|---|---|
| Gemma-4 26B-A4B-it | MoE, 128 experts top-8 | 4 B / 26 B | 30 | 2816 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Gemma-4 31B-it | Dense | 31 B | 60 | 5376 | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |

Both use GQA with `kv_channels=256`, RoPE base 1e6, and a 262144-token vocab.

The 31B recipe requires the `zhichen/gemma4-dense` branch of `radixark/Megatron-Bridge`.

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download google/gemma-4-26B-A4B-it --local-dir /root/models/gemma-4-26B-A4B-it
```

`--model-dir` and `--data-dir` default to `/root/models` and `/root/datasets`.
`prepare` performs these downloads for you.

### 3.2 No `torch_dist` conversion

The bridge reads the HF checkpoint directly, so `--hf-checkpoint` and
`--ref-load` both point at the download:

```bash
--hf-checkpoint <model-dir>/<model-name>
--ref-load      <model-dir>/<model-name>
--megatron-to-hf-mode bridge
```

## 4. Launch

```bash
cd /root/miles

# MoE, single node
python scripts/run_gemma_4_26b_a4b.py full-train --num-nodes 1

# dense, single node
python scripts/run_gemma_4_31b.py full-train --num-nodes 1
```

Passing `--num-nodes 1` puts the recipe into `debug_minimal` mode, which shortens
`--rollout-max-response-len` to 256 for a quick smoke test. Multi-node runs use
the full 8192.

## 5. Recipe Configuration

### 5.1 Parallelism

| Model | TP | PP | CP | EP | ETP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|---|
| 26B-A4B (MoE) | 4 | 1 | 1 | 8 | 1 | 1024 | 8 (1 × 8) |
| 31B (dense) | 4 | 1 | 1 | — | — | 512 | 8 (1 × 8) |

Activation checkpointing is on for the MoE recipe
(`--recompute-granularity full --recompute-method uniform
--recompute-num-layers 1`). The dense 31B runs a smaller token budget because
its 60 dense layers at hidden 5376 cost more activation memory per token than
the MoE's 30 layers at 2816.

### 5.2 Algorithm

GRPO. The MoE recipe adds low-variance KL; the dense one runs without it:

```bash
--advantage-estimator grpo
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
--rm-type gemma_math
--balance-data

# 26B-A4B only
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
```

Rollout batch 32 at 8 samples per prompt, global batch 256, `--lr 1e-6`. AIME
evaluation every 20 steps is available behind `--enable-eval` and is off by
default.

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 4
--sglang-mem-fraction-static 0.55   # 0.5 for the dense 31B
```

The MoE recipe pins sglang to conservative kernels:

```bash
--sglang-attention-backend triton
--sglang-moe-runner-backend triton
--sglang-disable-custom-all-reduce
--sglang-disable-cuda-graph
--sglang-disable-overlap-schedule
--sglang-disable-radix-cache
--use-rollout-routing-replay
```

`--use-rollout-routing-replay` replays the rollout's expert routing during the
training forward pass, so train log-probs match rollout log-probs. Every
sigmoid- or softmax-routed MoE recipe in miles needs this; the dense 31B does
not.

### 5.4 Notable quirks

- **Trained on the VLM checkpoint.** miles does not strip the vision tower; the
  bridge and sglang both handle the multimodal config, and the RL recipe simply
  trains the language stack.
- `--attention-backend unfused` on the training side for the MoE recipe.
- Routing is softmax with `seq_aux_loss` balancing and the bias update rate at 0
  (`--moe-router-bias-update-rate 0 --moe-aux-loss-coeff 0`), plus
  `--moe-grouped-gemm` and `--moe-router-dtype fp32`.
- The 31B recipe needs the `gemma4-dense` branch of `radixark/Megatron-Bridge`,
  because the dense config is driven straight through `Gemma4VLBridge`.

## 6. Pairs Well With

- [Backends Beyond Megatron](/advanced/architecture-support)
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
