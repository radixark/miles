---
title: DeepSeek
description: Miles recipes for the DeepSeek family — DeepSeek-V4 Flash (sparse-MLA + DSA indexer), V3, and R1.
---

# DeepSeek family

Miles ships recipes for the DeepSeek family across two generations: **DeepSeek-V4 Flash** introduces sparse multi-head latent attention with a learned indexer and KV compressors (8-node H200), while **V3 / R1** remain the canonical 16-node 671 B-parameter recipes (BF16 train + 128×128 block-wise FP8 rollout, DeepEP, DAPO-style dynamic sampling).

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| DeepSeek-V4-Pro | 49 B / 1.6 T | TBA | [deepseek-v4-pro](deepseek-v4-pro.md) |
| DeepSeek-V4-Flash | 13 B / 284 B | `sgl-project/DeepSeek-V4-Flash-FP8` | [deepseek-v4-flash](deepseek-v4-flash.md) |
| DeepSeek-V3 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3` | [deepseek](deepseek.md) |
| DeepSeek-R1 | 37 B / 671 B | `deepseek-ai/DeepSeek-R1` | [deepseek](deepseek.md) |

A validated DeepSeek-V4-Pro recipe is not yet available — see [`radixark/miles#1046`](https://github.com/radixark/miles/issues/1046) for tracking.

## Fastest path to train

DeepSeek-V4-Flash needs 8 nodes of 8× H200 and the `radixark/miles:dev` image:

```bash
cd /root/miles
python scripts/run_deepseek_v4.py full-train \
   --model-name DeepSeek-V4-Flash-FP8 \
   --num-nodes 8 --num-gpus-per-node 8
```

DeepSeek-R1 needs 16 nodes of 8× H100:

```bash
cd /root/miles
bash scripts/run-deepseek-r1.sh              # full 16-node run
```

See the [DeepSeek-V4 Flash](deepseek-v4-flash.md) page for the V4 architecture summary, parallelism layouts, and known workarounds; see the [DeepSeek R1 / V3](deepseek.md) page for the V3 flow — FP8 → BF16 conversion, Megatron parallelism layout (TP8 / PP4 / EP32 / CP4), per-arg walkthrough, and the alternate Python launcher (`scripts/run_deepseek.py`).

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md) — 671 B is where PD really earns its keep.
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md) — amortize weight sync across ranks.
- [Fault Tolerance](../../advanced/fault-tolerance.md) — node failures are inevitable at 16-node scale.
