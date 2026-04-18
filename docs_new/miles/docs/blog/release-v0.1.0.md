---
title: Release v0.1.0
description: INT4 QAT, R3 routing, unified FP8 — Miles's first stable release.
date: 2026-02-15
---

# Release v0.1.0

*February 15, 2026 — RadixArk team*

After three months of public beta, **Miles v0.1.0** is now the recommended release for
production training jobs. This is the first version with stable API, documented flags,
and a supported upgrade path.

## Highlights

### INT4 W4A16 QAT pipeline

Quant-aware training in 4-bit weights with 16-bit activations. Fits 1 T-class models on
a single H200 node and doubles rollout efficiency by eliminating cross-node sync.

→ [INT4 QAT recipe](../advanced/int4-qat.md)

### Complete argument reference

The full CLI reference is now documented, organised by subsystem — cluster, model,
rollout, eval, performance, RL, optimiser, SGLang passthrough.

→ [Server arguments](../user-guide/cli-reference.md)

### Unified FP8 — generally available

Promoted out of experimental. End-to-end FP8 with bit-identical quantisation between
rollout and training; no more MoE collapse from precision drift.

→ [FP8 & Low Precision](../advanced/fp8-low-precision.md)

### Rollout Routing Replay (R3) — generally available

Expert routing captured during inference, replayed during training. Eliminates the
train/inference routing mismatch.

→ [Miles Router (R3)](../advanced/miles-router.md)

### Multi-agent co-evolution

First-class multi-turn rollout for two-or-more-agent systems.

→ [Multi-agent example](../examples/multi-agent.md)

### Speculative decoding with online MTP-SFT

Online training of MTP layers during RL keeps draft acceptance high as the target
drifts.

→ [Speculative decoding](../advanced/speculative-decoding.md)

### Fully async rollout

Continuous background generation; queue-based consumption. Up to 2× end-to-end speedup
on rollout-bound jobs.

→ [Fully async example](../examples/fully-async.md)

## Breaking changes

| Change | Action |
|---|---|
| `--rm-path` removed | Use `--custom-rm-path` |
| `--num-rollout` is now required | Set explicitly |
| Sync → async train loop | See [migration guide](../developer/migration.md) |

## Known issues

- Some AMD MI300X MoE configurations need additional tuning of the Triton FA backend.
- Reproducibility on AMD is best-effort due to MIOpen workspace specifics.
- Speculative decoding for external (non-MTP) draft models is work-in-progress.

## Compatibility

| Package | Version |
|---|---|
| Python | 3.10+ |
| CUDA | 12.4+ |
| ROCm | 6.3+ |
| SGLang | 0.4.4+ |
| Megatron-LM | mcore-r0.10 |

## Get it

```bash
docker pull radixark/miles:latest
# or
pip install -U miles
```

Full release notes on [GitHub](https://github.com/radixark/miles/releases/tag/v0.1.0).

## What's next

**v0.2** is in development:

- FSDP-only trainer path (Megatron-free).
- Chrome-trace profiler integrated into the trainer.
- Adaptive per-source data weighting.
- VLM multi-turn polish.
