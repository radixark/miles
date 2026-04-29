---
title: P2P Weight Transfer
description: Direct rank-to-rank weight sync from actor to rollout via NCCL P2P.
---

# P2P Weight Transfer

After every training step, the trainer's updated weights have to reach the
SGLang engines so the next rollout is on-policy. The default broadcast path
sends each weight tensor to every rollout rank, which wastes bandwidth at
large model scales. P2P transfer delivers each shard directly to the rank that
needs it.

## Enable

```bash
--update-weight-transfer-mode p2p
```

`--update-weight-transfer-mode` accepts `broadcast` (default) or `p2p`.

## What changes

| | Broadcast | P2P |
|---|---|---|
| Per-rank receives | All weights, then drops what it doesn't need | Only its own shards |
| Implementation | NCCL collective | Direct rank-to-rank NCCL send/recv |
| Best-fit scale | Small to mid models | Large models with many rollout ranks |

## How it works

Steps performed by the actor (`miles/backends/megatron_utils/actor.py`):

1. Build a transfer plan that maps trainer ranks to rollout ranks based on
   TP/EP/PP sharding and GPU counts.
2. Megatron TP/EP shards are gathered and converted to HF parameter layout
   (the same conversion used by the broadcast path).
3. Each trainer rank sends its bucketed tensors to the destination rollout
   rank. There is no global collective.
4. When transfers complete, rollout engines update their weight version and
   resume generation.

## Hardware

P2P uses NCCL on top of NVLink (intra-node) and InfiniBand or RoCEv2
(inter-node). On a vanilla TCP backplane the speedup disappears.

## Tunable knobs

```bash
--p2p-transfer-num-workers 4           # thread-pool workers for P2P writes (default 4)
--p2p-transfer-timeout 30              # per-transfer timeout in seconds (default 30)
--update-weight-buffer-size 536870912  # bytes per update-weight buffer (default 512 MiB)
--update-weights-interval 1            # rollouts between weight syncs (default 1)
```

## Diagnostics

If P2P fails to initialise, Miles falls back to broadcast (usually because of
a missing IB device). Re-run with `NCCL_DEBUG=INFO` to see per-rank
diagnostics.

## When P2P helps

* Large models where the broadcast path is bandwidth-bound.
* Many rollout ranks: the broadcast tax scales with `num_rollout_ranks`.

## When broadcast is enough

* Single-node jobs where intra-node NVLink saturates the broadcast path.
* Smaller models.
* No RDMA-capable interconnect.

## Pairs with

* [Fault tolerance](fault-tolerance.md). P2P transfers are bounded by
  `--p2p-transfer-timeout` and fall back to broadcast on timeout.
* [INT4 QAT](int4-qat.md). Smaller weights mean less data per sync.
