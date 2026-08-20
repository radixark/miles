---
title: GLM4.7 Flash
description: Launch recipes for GLM-4.7-Flash — compact MLA + MoE with R3 enabled by default.
---
## 1. Model Introduction

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) is a lightweight, high-speed MoE model in the GLM-4.7 series from Zhipu AI, designed for single-GPU-node deployment.

**Key highlights:**

- **Compact MoE architecture**: 30 B total / 3 B active, sparse activation for efficient inference.
- **MLA attention**: Multi-head Latent Attention with q-LoRA rank 768 and kv-LoRA rank 512.
- **MTP head + EAGLE speculative**: built-in `--mtp-num-layers 1` and EAGLE rollout enabled by default.
- **R3 on by default**: the miles launcher enables `--use-rollout-routing-replay` out of the box.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| GLM-4.7-Flash | 3 B / 30 B | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download zai-org/GLM-4.7-Flash --local-dir /root/models/GLM-4.7-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

The launcher does all three downloads itself into `--model-dir` (default `/root/models`) and `--data-dir` (default `/root/datasets`), so this step is optional.

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py glm4.7-flash)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/GLM-4.7-Flash \
   --save          /root/models/GLM-4.7-Flash_torch_dist
```

The launcher does the conversion automatically.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
python scripts/run_glm47_flash.py --rollout-num-gpus-per-engine 4

# pass --hardware B200 on B200
```

Defaults (see `ScriptArgs`): `model_org=zai-org`, `model_name=GLM-4.7-Flash`, `num_gpus_per_node=8`, `hardware=H200`, `sglang_attention_backend=None`, `data_dir=/root/datasets`, `model_dir=/root/models`, `output_dir=/root/shared_data`. The `hardware` CLI also accepts `B200`.

## 5. Recipe Configuration

### 5.1 Parallelism

| TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |

Left to itself the launcher picks `--rollout-num-gpus-per-engine 2` on B200 and 1 on H200; those and the 4 above all divide the model's 20 attention heads. The recipe passes no `--sglang-enable-dp-attention` / `--sglang-dp-size` — the in-source comment notes that DP-attention requires `tp_size % dp_size == 0`.

### 5.2 Algorithm

GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --use-kl-loss --kl-loss-coef 0.00`.

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 4
--sglang-mem-fraction-static 0.7

# EAGLE speculative decoding (MTP)
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 2
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 3

# R3 — on by default in this recipe
--use-rollout-routing-replay
```

### 5.4 Optimizer

CPU Adam on:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

### 5.5 Notable quirks

- Megatron-side DeepEP / `flex` dispatcher are not enabled by this recipe.
- R3 (`--use-rollout-routing-replay`) is enabled by default — atypical for the rest of the model lineup.

## 6. Pairs Well With

- [Rollout Routing Replay (R3)](/advanced/miles-router) — already on by default.
- [Low Precision RL](/advanced/low-precision)
