---
title: Qwen
description: Miles recipes for the full Qwen3, Qwen3.5, and Qwen3-Next line — dense and MoE.
---
Miles ships ready-to-run RL recipes for every generation of the Qwen line: the dense Qwen3 series (0.6 B → 32 B), the Qwen3.5 family with its gated-attention architecture, the Qwen3 and Qwen3.5 MoE variants, and the Gated-Delta-Net Qwen3-Next-80B-A3B.

## Variants

| Family | Class | Sizes | Recipe |
|---|---|---|---|
| Qwen3 | Dense | 0.6 B · 1.7 B · 4 B · 8 B · 14 B · 32 B | [qwen3](/models/qwen/qwen3) |
| Qwen3 | MoE | 3 B / 30 B · 22 B / 235 B | [qwen3-moe](/models/qwen/qwen3-moe) |
| Qwen3.5 | Dense | 4 B · 9 B · 27 B | [qwen3-5](/models/qwen/qwen3-5) |
| Qwen3.5 | MoE | 3 B / 35 B | [qwen3-5-moe](/models/qwen/qwen3-5-moe) |
| Qwen3-Next | MoE (GDN) | 3 B / 80 B | [qwen3-next](/models/qwen/qwen3-next) |

## Fastest path to train

Qwen3-4B on a single 8× H100 node — the canonical starter recipe:

```bash
cd /root/miles
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
bash scripts/run-qwen3-4B.sh
```

Dataset is [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17K) at `/root/dapo-math-17k/dapo-math-17k.jsonl`. See the [Qwen3 Dense](/models/qwen/qwen3) page for the full walkthrough, weight conversion, and variants.

## Which variant do I pick?

- **Learning Miles for the first time** → Qwen3-4B ([qwen3](/models/qwen/qwen3)). Fits one H100 node, fast loop.
- **Need MoE on a single node** → Qwen3-30B-A3B ([qwen3-moe](/models/qwen/qwen3-moe)).
- **Scaling to multi-node** → Qwen3-235B-A22B ([qwen3-moe](/models/qwen/qwen3-moe)).
- **Latest dense architecture (gated attention, A\_log FP32)** → Qwen3.5-4B ([qwen3-5](/models/qwen/qwen3-5)).
- **Hybrid MTP / speculative decoding experiments** → Qwen3.5-35B-A3B ([qwen3-5-moe](/models/qwen/qwen3-5-moe)).
- **Gated-Delta-Net (fla backend, real-CP)** → Qwen3-Next-80B-A3B ([qwen3-next](/models/qwen/qwen3-next)).
