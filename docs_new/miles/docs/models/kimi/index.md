---
title: Kimi
description: Miles recipes for the Moonshot family — Kimi K2.6 / K2.5 (multimodal, 1 T / 32 B-A), Kimi K2 / K2-Thinking, and Moonlight 16B-A3B.
---
Miles supports Moonshot's MoE line from top to bottom. The latest Kimi K2.6 and K2.5 are natively multimodal agentic models at 1 T total / 32 B active per token; the text-only Kimi K2 (Instruct and Thinking variants) runs at the same 1 T / 32 B scale; and the compact Moonlight 16B-A3B fits on a single 8× H100 node, a handy single-node test target before scaling K2 across many nodes. K2-Thinking is the canonical INT4 QAT target, and the K2.5 / K2.6 recipe trains an INT4 actor under the same QAT path.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| Kimi-K2.6 | 32 B / 1 T | `moonshotai/Kimi-K2.6` | [kimi-k2.5](/miles/docs/models/kimi/kimi-k2.5) |
| Kimi-K2.5 | 32 B / 1 T | `moonshotai/Kimi-K2.5` | [kimi-k2.5](/miles/docs/models/kimi/kimi-k2.5) |
| Kimi-K2-Instruct | 32 B / 1 T | `moonshotai/Kimi-K2-Instruct` | [kimi-k2](/miles/docs/models/kimi/kimi-k2) |
| Kimi-K2-Thinking | 32 B / 1 T | `moonshotai/Kimi-K2-Thinking` | [kimi-k2](/miles/docs/models/kimi/kimi-k2) |
| Moonlight-16B-A3B | 3 B / 16 B | `moonshotai/Moonlight-16B-A3B` | [moonlight](/miles/docs/models/kimi/moonlight) |

## Fastest path to train

Moonlight on a single 8× H100 node — the smallest Moonshot recipe and a good MoE smoke test:

```bash
cd /root/miles
hf download moonshotai/Moonlight-16B-A3B --local-dir /root/Moonlight-16B-A3B
bash scripts/run-moonlight-16B-A3B.sh
```

See the [Moonlight](/miles/docs/models/kimi/moonlight) page for the full walkthrough, or [Kimi K2](/miles/docs/models/kimi/kimi-k2) for the 16-node K2-Thinking recipe (including the one-line `model_type` patch that lets Miles treat K2 as a DeepSeek-V3-shaped architecture).

## Which variant do I pick?

- **Latest multimodal agentic model** → Kimi-K2.6 or Kimi-K2.5 ([kimi-k2.5](/miles/docs/models/kimi/kimi-k2.5)).
- **Single-node MoE smoke test** → Moonlight-16B-A3B ([moonlight](/miles/docs/models/kimi/moonlight)).
- **Frontier-scale instruction-tuned MoE** → Kimi-K2-Instruct ([kimi-k2](/miles/docs/models/kimi/kimi-k2)).
- **Reasoning-style training, INT4 QAT target** → Kimi-K2-Thinking ([kimi-k2](/miles/docs/models/kimi/kimi-k2)).
