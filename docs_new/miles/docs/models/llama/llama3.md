---
title: Llama 3
description: Launch recipes for Llama 3.1 and 3.2 on NVIDIA and AMD.
---

# Llama 3

Miles supports Llama 3.1-8B and Llama 3.2-3B on both NVIDIA (H100 / H200 / B-series) and AMD MI300X. The launch script is created by copying the Qwen3 template and swapping the `source` line — `scripts/run-qwen3-4B.sh` on NVIDIA, `scripts/amd/run-qwen3-4B-amd.sh` on AMD.

## Variants

| Model | Params | HF ID | Model config |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 3 B | `meta-llama/Llama-3.2-3B-Instruct` | `scripts/models/llama3.2-3B-Instruct.sh` |
| Llama-3.1-8B-Instruct | 8 B | `meta-llama/Llama-3.1-8B-Instruct` | `scripts/models/llama3.1-8B-Instruct.sh` |

## Quick start (Llama-3.2-3B-Instruct, 8× H100)

```bash
cd /root/miles

# 1. Download weights
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /root/Llama-3.2-3B-Instruct

# 2. Convert HF → Megatron torch_dist
source scripts/models/llama3.2-3B-Instruct.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Llama-3.2-3B-Instruct \
   --save          /root/Llama-3.2-3B-Instruct_torch_dist

# 3. Launch (copy + adapt the Qwen3 script, swapping the source line)
cp scripts/run-qwen3-4B.sh scripts/run-llama3.2-3B-Instruct.sh
sed -i 's|scripts/models/qwen3-4B.sh|scripts/models/llama3.2-3B-Instruct.sh|' \
   scripts/run-llama3.2-3B-Instruct.sh
bash scripts/run-llama3.2-3B-Instruct.sh
```

On AMD MI300X, copy from the AMD Qwen3 template instead:

```bash
cp scripts/amd/run-qwen3-4B-amd.sh scripts/amd/run-llama3.2-3B-Instruct-amd.sh
sed -i 's|scripts/models/qwen3-4B.sh|scripts/models/llama3.2-3B-Instruct-amd.sh|' \
   scripts/amd/run-llama3.2-3B-Instruct-amd.sh
bash scripts/amd/run-llama3.2-3B-Instruct-amd.sh
```

## Expected signal

Standard `loss=… reward=…` trainer stdout. Llama is a straightforward dense recipe — if it's not converging, it's usually the dataset or chat template, not the model.

---

## Deep dive

### Launch scripts

Miles ships no pre-built Llama launch script — both NVIDIA and AMD use the Qwen3 / GLM4 script template. Copy `scripts/run-qwen3-4B.sh` (or `scripts/amd/run-qwen3-4B-amd.sh` on AMD) and swap the `source` line to your Llama model config.

### Parallelism

| Size | TP | PP | CP | `max_tokens_per_gpu` | GPUs |
|---|---|---|---|---|---|
| 3 B | 2 | 1 | 1 | 12 288 | 8 |
| 8 B | 2 | 1 | 1 | 8192 | 8 |

### AMD path

Use the ROCm image and the AMD copy-from-template snippet shown in the Quick start above. See [Platforms: AMD MI300X](../../platforms/amd.md) for the container setup.
