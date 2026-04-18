---
title: P2P Weight Transfer
description: Sub-30-second weight sync from actor to rollout via NCCL P2P / RDMA.
---

# P2P Weight Transfer

After every training step, the trainer's freshly-updated weights have to make their way
to the SGLang engines so the next rollout uses the on-policy model. At GLM4.5 / DeepSeek
scale that's hundreds of GB of state. The default broadcast path works, but it wastes
bandwidth — every rollout rank receives redundant copies. **P2P transfer** delivers each
shard directly to the rank that needs it.

## Enable it

```bash
--update-weight-transfer-mode p2p
```

That's the entire user-facing API. Reference design discussion: [issue #755](https://github.com/radixark/miles/issues/755).

## What changes

| | Broadcast (default) | P2P |
|---|---|---|
| Per-rank receives | All weights, then drops what it doesn't need | Only its shards |
| Bandwidth | NIC-bound × ranks | NIC-bound × shards |
| Implementation | NCCL collective | RDMA write |
| Latency at GLM4.5 scale | ~25–40 s | ~3–8 s |

## How it works

```mermaid
flowchart LR
    subgraph Trainer
        T1[Megatron rank 0]
        T2[Megatron rank 1]
        T3[Megatron rank ...]
    end
    subgraph Rollout
        R1[SGLang rank 0]
        R2[SGLang rank 1]
        R3[SGLang rank ...]
    end
    T1 -- RDMA write --> R1
    T1 -. " " .-> R2
    T2 -- RDMA write --> R2
    T3 -- RDMA write --> R3
```

In detail:

1. **Initialisation** — At startup, each trainer rank:
    * Builds a transfer plan that maps trainer ranks to rollout ranks based on TP/EP/PP
      and GPU counts.
    * Queries each rollout engine for its memory registration info (RDMA addresses +
      sizes).
    * Constructs a local CPU model replica that mirrors the target's sharding so weights
      can be reshaped before transfer.
2. **Weight gather** — Megatron TP/EP shards are all-gathered and converted to HF format
   (same as the broadcast path).
3. **P2P write** — Each trainer rank writes its bucketed tensors directly into the
   destination rollout rank's memory using RDMA. No collective; pure write.
4. **Sync** — When all writes ACK, rollout engines bump their weight version and resume
   generation.

## Hardware requirements

| Item | Required for P2P? |
|---|---|
| NVLink intra-node | Yes |
| InfiniBand or RoCEv2 inter-node | Yes (this is what RDMA writes ride) |
| NCCL with `NCCL_P2P_LEVEL=NVL` | Recommended |
| `NVSHMEM` | Optional, helps |

If you're on a vanilla TCP backplane, P2P falls back to broadcast — nothing breaks but
you lose the speedup.

## Tunable knobs

```bash
--p2p-bucket-size-mb 64        # bucket size for RDMA writes
--p2p-num-streams 4            # parallel CUDA streams for transfer
--p2p-weight-sync-retries 3    # retries on failure (then falls back to broadcast)
```

## What you should see

```text
weight_sync/mode               = p2p
weight_sync/bytes              = 670_512_341_504
weight_sync/duration_s         = 6.4
weight_sync/effective_bw_gbps  = 105
```

If `mode` says `broadcast`, P2P initialisation failed silently — usually a missing IB
device. Check `--debug-weight-sync 1` for the per-rank error.

## When NOT to use P2P

* Single-node jobs — NVLink intra-node already maxes out broadcast.
* Models < 30 B — overhead isn't worth it.
* You don't have RDMA-capable interconnect.

## Pairs nicely with

* [Fault tolerance](fault-tolerance.md) — P2P retries gracefully fall back to broadcast.
* [INT4 QAT](int4-qat.md) — 4× smaller weights, 4× faster sync.
