---
title: Kimi
sidebarTitle: Overview
description: Miles recipes for the Moonshot family — Kimi K2.6 / K2.5 (multimodal, 1 T / 32 B-A) and Kimi K2 / K2-Thinking.
---
Miles supports Moonshot's MoE line from top to bottom. The latest Kimi K2.6 and K2.5 are natively multimodal agentic models at 1 T total / 32 B active per token, and the text-only Kimi K2 (Instruct and Thinking variants) runs at the same 1 T / 32 B scale. K2-Thinking is the canonical INT4 QAT target, and the K2.5 / K2.6 recipe trains an INT4 actor under the same QAT path.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| Kimi-K2.6 | 32 B / 1 T | `moonshotai/Kimi-K2.6` | [kimi-k2.5](/models/kimi/kimi-k2.5) |
| Kimi-K2.5 | 32 B / 1 T | `moonshotai/Kimi-K2.5` | [kimi-k2.5](/models/kimi/kimi-k2.5) |
| Kimi-K2-Instruct | 32 B / 1 T | `moonshotai/Kimi-K2-Instruct` | [kimi-k2](/models/kimi/kimi-k2) |
| Kimi-K2-Thinking | 32 B / 1 T | `moonshotai/Kimi-K2-Thinking` | [kimi-k2](/models/kimi/kimi-k2) |

## Fastest path to train

The single-node Kimi-K2.5 2-layer smoke test, before scaling the full model across many nodes:

```bash
python scripts/run_kimi_k25.py full-train --model-name Kimi-K2.5-2layer --num-nodes 1
```

See the [Kimi K2.5](/models/kimi/kimi-k2.5) page for the full multi-node recipe, or [Kimi K2](/models/kimi/kimi-k2) for the 16-node K2-Thinking recipe (including the one-line `model_type` patch that lets Miles treat K2 as a DeepSeek-V3-shaped architecture).

## Which variant do I pick?

- **Latest multimodal agentic model** → Kimi-K2.6 or Kimi-K2.5 ([kimi-k2.5](/models/kimi/kimi-k2.5)).
- **Frontier-scale instruction-tuned MoE** → Kimi-K2-Instruct ([kimi-k2](/models/kimi/kimi-k2)).
- **Reasoning-style training, INT4 QAT target** → Kimi-K2-Thinking ([kimi-k2](/models/kimi/kimi-k2)).
