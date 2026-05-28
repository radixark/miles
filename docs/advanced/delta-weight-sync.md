---
title: Delta Weight Sync
description: Sparse non-colocated actor-to-rollout weight synchronization.
---

# Delta Weight Sync

Delta weight sync is a non-colocated weight update mode that sends only changed
weight positions and values instead of broadcasting every parameter every step.
It is intended for training/inference disaggregation where full model broadcasts
are too expensive.

## Quick Start

Disk transport writes per-flush safetensors files to a filesystem shared by the
trainer and rollout engines:

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd
--update-weight-delta-dir /shared/fs/delta-updates
```

NCCL transport sends the same delta payloads directly over the model update
process group:

```bash
--update-weight-mode delta
--update-weight-transport nccl
--update-weight-encoding indices
```

Receiver-side SGLang tuning:

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
--sglang-update-weight-delta-read-workers 4
```

## How It Works

The first delta update seeds a pinned CPU snapshot from the current trainer
weights and does not contact rollout engines. Later updates byte-compare current
HF-formatted weights against that snapshot, encode changed positions plus values,
and apply the payload on SGLang with a NaN-masked overwrite path.

Both transports use the same wire layout:

- `__positions__`: packed position bytes.
- `__values__`: changed values, concatenated across params.
- `DeltaSpec`: encoding, per-parameter offsets, shapes, dtypes, and checksum.

`indices` stores int32 absolute positions. `deltas` stores uint16 gap deltas
with uint32 fallback per parameter. `deltas_zstd` uses the delta encoding and
compresses disk files with zstd.

## Constraints

Delta sync is rejected with `--colocate` because colocated sync uses CUDA IPC and
does not benefit from sparse wire encoding. It is also separate from Miles' P2P
full-weight transfer path; choose `--update-weight-transport nccl` or `disk`
when `--update-weight-mode delta` is enabled.
