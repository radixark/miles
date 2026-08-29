---
title: Inkling-Small
description: Launch recipe for Inkling-Small (276 B), the compact sibling of Inkling — same architecture, 4-node H200 footprint.
---

## 1. Model Introduction

[Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) is the compact member of Thinking Machines Lab's Inkling family: a 276 B-total / 12 B-active-parameter, 42-layer multimodal MoE (256 routed + 2 shared experts, top-6 sigmoid routing) that matches — and on some benchmarks exceeds — the flagship 975 B model (e.g. SWEBench Verified 80.2 vs 77.6) at a fraction of the deployment footprint. The architecture is the same as [Inkling](/models/thinkingmachines/inkling) — ShortConv, local/global relative attention, and the shared-expert-sink MoE — so everything on the Inkling page (architecture summary, attention backends, R3 routing replay, LoRA schema, multimodal RL) applies unchanged. This page only covers what differs: the model registry entry and the validated small-cluster launch profile.

## 2. Supported Variants

| Model | Active / Total | Layers | HF ID | Recipe |
|---|---|---|---|---|
| Inkling-Small | 12 B / 276 B | 42 | [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) | this page |

## 3. Quick start

Validated on 4 nodes × 8 H200 (TP4 SP PP8 EP4, DP1):

```bash
cd /root/miles

# Full-parameter GRPO. 276 B fits with the CPU-offloaded optimizer -
# no NVMe streaming needed (unlike the 975 B recipe).
python scripts/run_inkling.py train \
   --model-name Inkling-Small --train-mode full --task dapo_math \
   --num-nodes 4 --num-gpus-per-node 8 \
   --lr 5e-5 --rollout-batch-size 64 --global-batch-size 128 \
   --sglang-context-length 4096 --rollout-max-response-len 2048 \
   --extra-args "--offload-train-target cpu --sglang-mem-fraction-static 0.65 \
      --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer"

# LoRA GRPO (rank 32, all-linear), same cluster; adapter-only weight
# sync swaps in ~4 s per rollout.
python scripts/run_inkling.py train \
   --model-name Inkling-Small --train-mode lora --task dapo_math \
   --num-nodes 4 --num-gpus-per-node 8 \
   --lr 2e-4 --rollout-batch-size 64 --global-batch-size 128 \
   --sglang-context-length 4096 --rollout-max-response-len 2048
```

The model definition lives in `scripts/models/inkling-small.py` (`MODEL_ARGS_NUM_LAYERS` overrides the layer count for sliced smoke/parity checkpoints). HF → `torch_dist` conversion uses the same tool as Inkling with this recipe file — a single 8-GPU node (TP8 EP8) converts it in one pass.

## 4. Validated parallelism

| Hardware | GPUs | TP | SP | PP | EP | expert-TP | Notes |
|---|---|---|---|---|---|---|---|
| H200 | 32 | 4 | on | 8 | 4 | 1 | `--decoder-last-pipeline-num-layers 7` (42 = 7×5 + 7) |
| H200 | 56 | 4 | on | 7 | 8 | 1 | 42 = 7×6, no uneven split needed; use `--rollout-num-gpus-per-engine 8` |
| H200 | 64 | 4 | on | 8 | 8 | 1 | 8 GPUs per node; `--decoder-last-pipeline-num-layers 7` (PP7 cannot divide 64) |

Batch shape is configurable from the launcher (`--rollout-batch-size`, `--global-batch-size`; defaults 32/64). The validated Small runs used 64/128: full with `--lr 5e-5`, LoRA with `--lr 2e-4` — both produce a steadily rising dapo-math reward curve. At the launcher's conservative LoRA default (5e-6) the zero-initialised B factors take hundreds of rollouts to accumulate a visible delta-W, which reads as "not learning".
