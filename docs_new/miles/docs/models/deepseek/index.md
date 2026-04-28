---
title: DeepSeek
description: Miles recipes for DeepSeek-V3 and DeepSeek-R1 — 671 B total / 37 B active, BF16 training with FP8 rollout.
---

# DeepSeek family

Miles ships the canonical 16-node recipe for the largest models it currently trains — DeepSeek-V3 and DeepSeek-R1, both 671 B total parameters with 37 B active per token. Training runs in BF16, rollout in 128×128 block-wise FP8, with DeepEP and DAPO-style dynamic sampling.

## Variants

| Model | Active / Total | HF ID | Recipe |
|---|---|---|---|
| DeepSeek-V3 | 37 B / 671 B | `deepseek-ai/DeepSeek-V3` | [deepseek](deepseek.md) |
| DeepSeek-R1 | 37 B / 671 B | `deepseek-ai/DeepSeek-R1` | [deepseek](deepseek.md) |

## Fastest path to train

DeepSeek-R1 needs 16 nodes of 8× H100:

```bash
cd /root/miles
bash scripts/run-deepseek-r1.sh              # full 16-node run
```

See the [DeepSeek R1 / V3](deepseek.md) page for the full flow — FP8 → BF16 conversion, Megatron parallelism layout (TP8 / PP4 / EP32 / CP4), per-arg walkthrough, and the alternate Python launcher (`scripts/run_deepseek.py`).

## Pairs well with

- [PD Disaggregation](../../advanced/pd-disaggregation.md) — 671 B is where PD really earns its keep.
- [P2P Weight Transfer](../../advanced/p2p-weight-transfer.md) — amortise weight sync across ranks.
- [Fault Tolerance](../../advanced/fault-tolerance.md) — node failures are inevitable at 16-node scale.
