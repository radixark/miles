---
title: Gemma
sidebarTitle: Overview
description: Miles recipes for Google's Gemma-4 line, trained on the base VLM checkpoint through the HF to Megatron bridge.
---
Miles supports Google's Gemma-4 in both released instruction-tuned sizes. Both train as
language models on the base VLM checkpoint, through the HF to Megatron bridge
(`--megatron-to-hf-mode bridge`), so there is no offline `torch_dist` conversion.

## Variants

| Model | Class | Active / Total | HF ID | Recipe |
|---|---|---|---|---|
| Gemma-4 26B-A4B-it | MoE, 128 experts top-8 | 4 B / 26 B | `google/gemma-4-26B-A4B-it` | [gemma-4](/models/gemma/gemma-4) |
| Gemma-4 31B-it | Dense | 31 B | `google/gemma-4-31B-it` | [gemma-4](/models/gemma/gemma-4) |

## Fastest path to train

Both recipes run on a single 8-GPU node:

```bash
cd /root/miles
python scripts/run_gemma_4_26b_a4b.py full-train --num-nodes 1
```

`--num-nodes 1` shortens the response length for a smoke test. See
[Gemma-4](/models/gemma/gemma-4) for the full walkthrough.

## Which variant do I pick?

- **Cheaper to train, sparse** → 26B-A4B. Four billion active parameters, and expert
  parallelism carries the width.
- **Dense, no routing to reason about** → 31B. It needs the `gemma4-dense` branch of
  `radixark/Megatron-Bridge`, and runs a smaller per-GPU token budget because its 60 dense
  layers cost more activation memory per token.

## Pairs well with

- [Backends Beyond Megatron](/advanced/architecture-support), the bridge path Gemma rides on.
- [P2P Weight Transfer](/advanced/p2p-weight-transfer)
