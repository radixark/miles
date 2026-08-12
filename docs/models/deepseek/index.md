---
title: DeepSeek
sidebarTitle: Overview
description: Miles recipes for the DeepSeek family — V4 Flash, V4 Pro, and V3.2.
---
Miles ships recipes for the DeepSeek family across two generations. **DeepSeek-V4** pairs sparse multi-head latent attention with a learned indexer, KV compressors, and hyper-connection routing. **DeepSeek-V3.2** keeps the V3 MoE and MLA shapes and adds DeepSeek Sparse Attention (DSA), the same attention implementation the GLM-5 recipes use. **DeepSeek-V3** itself remains available through `scripts/run_deepseek.py`.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| DeepSeek-V4-Pro | 49 B / 1.6 T | TBA | [deepseek-v4-pro](/models/deepseek/deepseek-v4-pro) |
| DeepSeek-V4-Flash | 13 B / 284 B | `sgl-project/DeepSeek-V4-Flash-FP8` | [deepseek-v4-flash](/models/deepseek/deepseek-v4-flash) |
| DeepSeek-V3.2 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3.2` | [deepseek-v3-2](/models/deepseek/deepseek-v3-2) |
| DeepSeek-V3 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3` | [deepseek](/models/deepseek/deepseek) |

A validated DeepSeek-V4-Pro recipe is not yet available — see [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046) for tracking.

## Fastest path to train

DeepSeek-V4-Flash needs 8 nodes of 8× H200 and the `radixark/miles:latest` image:

```bash
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 8 --num-gpus-per-node 8
```

DeepSeek-V3.2 needs 8 training nodes of 8 GPUs plus separate rollout GPUs:

```bash
cd /root/miles
python scripts/run_deepseek_v32.py full-train \
   --actor-num-nodes 8 --rollout-num-gpus 8
```

See the [DeepSeek-V4 Flash](/models/deepseek/deepseek-v4-flash) page for the V4 architecture summary, parallelism layouts, and known workarounds. See the [DeepSeek-V3.2](/models/deepseek/deepseek-v3-2) page for the V3.2 flow — FP8 → BF16 conversion, the TP2 / PP4 / EP16 training layout, the NSA rollout settings, and the FP8 / MXFP8 options; see the [DeepSeek V3](/models/deepseek/deepseek) page for the V3 recipe.

## Pairs well with

- [Low Precision RL](/advanced/low-precision) — both generations ship FP8 checkpoints and optional low-precision rollout.
- [P2P Weight Transfer](/advanced/p2p-weight-transfer) — amortize weight sync across ranks.
- [Fault Tolerance](/advanced/fault-tolerance) — node failures are inevitable at 8-node scale and above.
