---
title: Qwen3
description: Launch recipes for dense Qwen3 models (0.6 B – 32 B).
---
## 1. Model Introduction

[Qwen3](https://github.com/QwenLM/Qwen3) is the latest generation of Alibaba's Qwen language model series, available in dense and MoE variants with both Instruct and reasoning-enhanced Thinking editions.

**Key highlights:**

- **Stronger general intelligence**: significant improvements in instruction following, logical reasoning, mathematics, science, coding, and tool usage over Qwen2.5.
- **Extended context length**: trained for 256 K-token contexts, useful for long-document reasoning and agentic workflows.
- **Flexible deployment options**: dense sizes from 0.6 B up to 32 B; this page covers the dense recipes (MoE recipes live in [qwen3-moe](/models/qwen/qwen3-moe)).
- **Stronger agent interaction**: improved tool-use and search-based agent performance.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Qwen3-4B-Instruct-2507 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-14B | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| Qwen3-32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024
```

### 3.2 HF → Megatron `torch_dist` conversion

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-4B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-4B \
   --save          /root/models/Qwen3-4B_torch_dist
```

The converter auto-derives PP from `WORLD_SIZE`; for larger sizes drive it with `torchrun --nproc-per-node 8`. The FSDP launcher (`scripts/run_qwen3_0_6b_fsdp.py`) loads the HF checkpoint directly and skips this step.

## 4. Launch

### 4.1 Quick start

```bash
cd /root/miles
python scripts/run_qwen3_dense.py --model-name Qwen3-4B
```

One launcher covers the whole dense line — pick the variant with `--model-name` (`Qwen3-32B`, and the Qwen3.5 / Qwen3.6 sizes), and it selects the matching `qwen3-XB.py` model config. To run on a slice of a node, add `--num-gpus-per-node 4 --cuda-visible-devices 4,5,6,7`.

Checkpoints are read from `--model-dir` (default `/root/models`), datasets from `--data-dir` (default `/root/datasets`), and `--save`/`--load` point under `--output-dir` (default `/root/shared_data`).

The Qwen3-4B-Instruct-2507 config (`scripts/models/qwen3-4B-Instruct-2507.py`) just calls `qwen3-4B` with `rotary_base=5000000` (`MODEL_ARGS_ROTARY_BASE` still works as an environment override) — load it when converting / launching the Instruct-2507 checkpoint.

## 5. Recipe Configuration

### 5.1 Parallelism

`scripts/run_qwen3_dense.py` holds one recipe per `--model-name`:

| `--model-name` | TP | PP | CP | EP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|---|
| `Qwen3-4B` | 2 | 1 | 1 | 1 | 9216 | 8 (1 × 8) |
| `Qwen3-32B` | 8 | 1 | 1 | 1 | 20480 | 8 (1 × 8) |

`Qwen3-4B` also runs on a 4-GPU slice with `--num-gpus-per-node 4`; the parallelism is unchanged.

`--sequence-parallel` is on whenever TP > 1.

### 5.2 Algorithm

GRPO baseline across all dense recipes:

```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
```

Rollout uses `--rm-type deepscaler` against `dapo-math-17k`. The SFT recipe (`python scripts/run_qwen3_sft.py --model-name Qwen3-4B-Base`) trains on `/root/datasets/openhermes2_5.parquet`.

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 2
--sglang-mem-fraction-static 0.7
```

`Qwen3-32B` additionally pins `--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)`. The FSDP variant uses `--attn-implementation flash_attention_3`, SGLang attention backend `fa3`, and adds `--update-weight-buffer-size 536870912 --gradient-checkpointing`.

### 5.4 Optimizer

`Qwen3-32B` enables CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

The 4 B / 8 B / 14 B recipes leave Adam on GPU.

### 5.5 Notable quirks

- **BF16 train + FP8 inference**: download `Qwen/Qwen3-4B-FP8` and pass `--extra-args "--hf-checkpoint /root/models/Qwen3-4B-FP8"` to swap rollout to FP8 while keeping BF16 training. See [Low Precision RL](/advanced/low-precision).
- **FSDP backend**: `python3 scripts/run_qwen3_0_6b_fsdp.py` runs a Qwen3-0.6B recipe with `--train-backend fsdp` (downloads model + datasets itself); no Megatron `torch_dist` conversion needed.
- **AMD ROCm**: `python scripts/amd/run_qwen3_4b.py` mirrors the recipe, with the GPU count per node resolved from `--hardware` (`MI350X` / `MI355X`).

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
- [Backends Beyond Megatron](/advanced/architecture-support) — for the FSDP variant.
