---
title: Kimi K2
description: Launch recipes for Kimi K2-Instruct and K2-Thinking (1 T total, 32 B active).
---

# Kimi K2

Kimi K2 is Moonshot's 1 T-parameter MoE — 32 B active per token, 61 layers, Instruct and Thinking variants. Miles treats it as a DeepSeek-V3-shaped architecture (one `sed` away), converts on 4 nodes, and trains on 16. K2-Thinking is also the canonical target for INT4 QAT.

## Variants

| Model | Active / Total | HF ID | Model config |
|---|---|---|---|
| Kimi-K2-Instruct | 32 B / 1 T | `moonshotai/Kimi-K2-Instruct` | `scripts/models/kimi-k2.sh` |
| Kimi-K2-Thinking | 32 B / 1 T | `moonshotai/Kimi-K2-Thinking` | `scripts/models/kimi-k2-thinking.sh` |

## Quick start (K2-Thinking, 16 nodes × 8 H100)

```bash
cd /root/miles

# 1. Env (shared FS required)
export BASE_DIR=/shared/kimi
export MASTER_ADDR=<head node IP>

# 2. Download weights
hf download moonshotai/Kimi-K2-Thinking --local-dir $BASE_DIR/Kimi-K2-Thinking

# 3. Patch config — Miles treats kimi_k2 as deepseek_v3
sed -i 's/"model_type": "kimi_k2"/"model_type": "deepseek_v3"/' \
   $BASE_DIR/Kimi-K2-Thinking/config.json

# 4. Convert HF → Megatron torch_dist (4 nodes, NODE_RANK=0..3)
source scripts/models/kimi-k2-thinking.sh
PYTHONPATH=/root/Megatron-LM torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/Kimi-K2-Thinking \
   --save          $BASE_DIR/Kimi-K2-Thinking_torch_dist

# 5. Launch GRPO (node 0; attach other nodes via Ray, see DeepSeek recipe)
bash scripts/run-kimi-k2-Thinking.sh
```

## Expected signal

Standard `loss=… reward=…` trainer stdout. Signals at 1 T scale:

- The `sed` patch is easy to forget — if conversion fails with "unknown model_type", that's it.
- `router/replay_hit_rate` should climb to ≈ 1.0 with R3 enabled. Kimi K2 has 384 experts so imbalance can get expensive fast.
- Fault tolerance is effectively mandatory at 16-node scale — see the [Fault Tolerance](../../advanced/fault-tolerance.md) page.

---

## Deep dive

### Architecture

| Property | Value |
|---|---|
| Layers | 61 |
| First K dense layers | 1 |
| Hidden size | 7168 |
| FFN hidden size | 18 432 |

### Launch scripts

| Script | Target |
|---|---|
| `scripts/run-kimi-k2-Instruct.sh` | K2-Instruct |
| `scripts/run-kimi-k2-Thinking.sh` | K2-Thinking |

### Parallelism

| TP | EP | PP | CP | `max_tokens_per_gpu` | Nodes |
|---|---|---|---|---|---|
| 8 | 32 | 4 | 4 | 16 384 | 16 |

### INT4 QAT

K2-Thinking is the canonical target for INT4 QAT — it's where the single-node deployment recipe really pays off. See [INT4 Quant-Aware Training](../../advanced/int4-qat.md).

### Related

- [FP8 & Low Precision](../../advanced/fp8-low-precision.md)
- [Miles Router (R3)](../../advanced/miles-router.md)
- [Fault Tolerance](../../advanced/fault-tolerance.md) — mandatory at this scale
