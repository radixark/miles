---
title: Qwen3-Next 80B-A3B
description: Launch recipes for Qwen3-Next-80B-A3B-Thinking on Megatron and FSDP backends.
---

# Qwen3-Next 80B-A3B

Qwen3-Next swaps classical attention for Gated-Delta-Net (GDN). Miles runs it through the HuggingFace-wrapped Megatron backend, which loads the `Qwen/Qwen3-Next-80B-A3B` HF module as a Megatron stage without re-implementing GDN from scratch.

## Variant

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Qwen3-Next-80B-A3B-Thinking | 3 B / 80 B | `Qwen/Qwen3-Next-80B-A3B-Thinking` | `scripts/models/qwen3-next-80B-A3B.sh` |

## Required env

All three launch scripts (`run-qwen3-next-80B-A3B.sh`, `run-qwen3-next-80B-A3B-8gpus.sh`, `run-qwen3-next-80B-A3B-fsdp.sh`) hard-fail unless these are set:

```bash
export BASE_FOLDER=<shared FS path, must contain the staged checkpoint + datasets>
export MASTER_ADDR=<head node IP>
```

Each script expects the following layout under `$BASE_FOLDER`:

```
Qwen3-Next-80B-A3B-Thinking/                     --hf-checkpoint
Qwen3-Next-80B-A3B-Thinking_torch_dist/          --ref-load 
Qwen3-Next-80B-A3B-Thinking_miles/               --load / --save
dapo-math-17k/dapo-math-17k.jsonl                training prompts
aime-2024/aime-2024.jsonl                        eval prompts
```


## Quick start

```bash
cd /root/miles
export BASE_FOLDER=...; export MASTER_ADDR=...

# Convert HF → Megatron torch_dist (drive on the appropriate number of GPUs)
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_FOLDER/Qwen3-Next-80B-A3B-Thinking \
   --save          $BASE_FOLDER/Qwen3-Next-80B-A3B-Thinking_torch_dist

# Launch — ssh fan-out happens inside the script
bash scripts/run-qwen3-next-80B-A3B.sh
```

## Launch scripts

All values straight from `scripts/`:

| Script | Backend | GPUs | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` |
|---|---|---|---|---|---|---|---|---|
| `scripts/run-qwen3-next-80B-A3B.sh` | Megatron | 32 | 2 | 4 | 2 | 8 | 1 | 8192 |



## SGLang flags

Canonical (`run-qwen3-next-80B-A3B.sh`) enables EAGLE speculative rollout:

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.8
--sglang-ep-size 8
--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

# mtp / EAGLE
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 2
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 3
--sglang-enable-draft-weights-cpu-backup
--sglang-max-running-requests 512
```

The 6-GPU and FSDP variants ship the EAGLE block commented out and use `--rollout-num-gpus-per-engine 2 --rollout-num-gpus 2 --sglang-mem-fraction-static 0.8 --sglang-ep-size 1`.

## GSPO + CPU Adam

All three scripts use GSPO (`--advantage-estimator gspo --eps-clip 4e-4`, `--use-kl-loss` commented out). The Megatron variants also enable `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`.

## Pairs well with

- [Backends Beyond Megatron](../../advanced/architecture-support.md)
- [Miles Router (R3)](../../advanced/miles-router.md)
