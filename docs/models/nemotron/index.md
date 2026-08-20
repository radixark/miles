---
title: Nemotron
sidebarTitle: Overview
description: Miles recipes for NVIDIA's Nemotron-3 family — Mamba+Attention(+MoE) hybrids loaded via Megatron AutoBridge.
---
Miles supports NVIDIA's Nemotron-3 line: a Mamba + Attention hybrid that, in the Super tier, adds MoE and ships natively in FP8. All three variants load via the Megatron AutoBridge path, so there is no offline HF → `torch_dist` conversion step.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| Nemotron-3-Nano | 4 B / 4 B (dense) | `nvidia/Nemotron-3-Nano-4B` | [nemotron-3-nano](/models/nemotron/nemotron-3-nano) |
| Nemotron-3-Nano MoE | 3 B / 30 B | `nvidia/Nemotron-3-Nano-30B-A3B` | [nemotron-3-nano-moe](/models/nemotron/nemotron-3-nano-moe) |
| Nemotron-3-Super | 12 B / 120 B (FP8) | `nvidia/Nemotron-3-Super-120B-A12B-FP8` | [nemotron-3-super](/models/nemotron/nemotron-3-super) |
| Nemotron-3-Ultra | 55 B / 550 B | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | [nemotron-3-ultra](/models/nemotron/nemotron-3-ultra) |

## Fastest path to train

Nemotron-3-Nano (dense, 4 B) is the smallest and runs on a single 8-GPU node:

```bash
python scripts/run_nemotron_3_nano.py --model-name NVIDIA-Nemotron-3-Nano-4B-BF16
```

See the [Nemotron-3-Nano](/models/nemotron/nemotron-3-nano) page for the dense walkthrough, [Nemotron-3-Nano MoE](/models/nemotron/nemotron-3-nano-moe) for the 30 B MoE variant, and [Nemotron-3-Super](/models/nemotron/nemotron-3-super) for the FP8-native 120 B-A12B recipe.

## Which variant do I pick?

- **Smallest, single-node smoke test** → Nemotron-3-Nano ([nemotron-3-nano](/models/nemotron/nemotron-3-nano)).
- **Mid-scale hybrid MoE** → Nemotron-3-Nano MoE ([nemotron-3-nano-moe](/models/nemotron/nemotron-3-nano-moe)).
- **Frontier-scale FP8-native MoE** → Nemotron-3-Super ([nemotron-3-super](/models/nemotron/nemotron-3-super)).
- **Largest, latent MoE across 16 nodes** → Nemotron-3-Ultra ([nemotron-3-ultra](/models/nemotron/nemotron-3-ultra)).

## Pairs well with

- [Backends Beyond Megatron](/advanced/architecture-support) — the AutoBridge path Nemotron rides on.
- [Low Precision RL](/advanced/low-precision) — Super ships natively in FP8.
