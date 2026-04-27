---
title: Kimi
description: Miles recipes for the Moonshot family — Kimi K2 / K2-Thinking (1 T / 32 B-A) and Moonlight 16B-A3B.
---

# Kimi family

Miles supports both ends of Moonshot's MoE line: the 1 T-parameter Kimi K2 (Instruct and Thinking variants) at 32 B active per token, and the compact Moonlight 16B-A3B that fits on a single 8× H100 node — handy as a single-node test target before scaling K2 across 16 nodes. K2-Thinking is also the canonical target for INT4 QAT.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| Kimi-K2-Instruct | 32 B / 1 T | `moonshotai/Kimi-K2-Instruct` | [kimi-k2](kimi-k2.md) |
| Kimi-K2-Thinking | 32 B / 1 T | `moonshotai/Kimi-K2-Thinking` | [kimi-k2](kimi-k2.md) |
| Moonlight-16B-A3B | 3 B / 16 B | `moonshotai/Moonlight-16B-A3B` | [moonlight](moonlight.md) |

## Fastest path to train

Moonlight on a single 8× H100 node — the smallest Moonshot recipe and a good MoE smoke test:

```bash
cd /root/miles
hf download moonshotai/Moonlight-16B-A3B --local-dir /root/Moonlight-16B-A3B
bash scripts/run-moonlight-16B-A3B.sh
```

See the [Moonlight](moonlight.md) page for the full walkthrough, or [Kimi K2](kimi-k2.md) for the 16-node K2-Thinking recipe (including the one-line `model_type` patch that lets Miles treat K2 as a DeepSeek-V3-shaped architecture).

## Which variant do I pick?

- **Single-node MoE smoke test** → Moonlight-16B-A3B ([moonlight](moonlight.md)).
- **Frontier-scale instruction-tuned MoE** → Kimi-K2-Instruct ([kimi-k2](kimi-k2.md)).
- **Reasoning-style training, INT4 QAT target** → Kimi-K2-Thinking ([kimi-k2](kimi-k2.md)).
