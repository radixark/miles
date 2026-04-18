---
title: Kimi K2
description: Launch recipes for Kimi K2-Instruct and K2-Thinking (1 T total, 32 B active).
---

# Kimi K2

Kimi K2 is a 1 T-parameter MoE with 32 B activation per token. Miles supports both
variants — Instruct and Thinking.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Kimi-K2-Instruct | 32 B / 1 T | `moonshotai/Kimi-K2-Instruct` | `scripts/models/kimi-k2.sh` |
| Kimi-K2-Thinking | 32 B / 1 T | `moonshotai/Kimi-K2-Thinking` | `scripts/models/kimi-k2-thinking.sh` |

## Architecture

| Property | Value |
|---|---|
| Layers | 61 |
| First K dense layers | 1 |
| Hidden size | 7168 |
| FFN hidden size | 18 432 |

## Launch scripts

| Script | Target |
|---|---|
| `scripts/run-kimi-k2-Instruct.sh` | K2-Instruct |
| `scripts/run-kimi-k2-Thinking.sh` | K2-Thinking |

## Prerequisites

Kimi K2 ships with `model_type: "kimi_k2"` in `config.json`, but Miles's conversion
pipeline treats it as a DeepSeek-V3 architecture. Edit the file:

```bash
sed -i 's/"model_type": "kimi_k2"/"model_type": "deepseek_v3"/' \
   /data/Kimi-K2-Thinking/config.json
```

## Convert weights (4 nodes)

```bash
cd /root/miles
source scripts/models/kimi-k2-thinking.sh

PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/Kimi-K2-Thinking \
   --save          $BASE_DIR/Kimi-K2-Thinking_torch_dist
```

## Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | Nodes |
|---|---|---|---|---|---|
| 8 | 32 | 4 | 4 | 16 384 | 16 |

## Launch

```bash
bash scripts/run-kimi-k2-Thinking.sh
```

## INT4 QAT

Kimi K2-Thinking is the canonical target for INT4 QAT — it's where RadixArk's single-node
deployment recipe really pays off. See [INT4 Quant-Aware Training](../../advanced/int4-qat.md).

## Related

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Miles Router (R3)](../../advanced/miles-router.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md) — mandatory at this scale
