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

## When P2P helps

* Large models where the broadcast path is bandwidth-bound.
* Many rollout ranks: the broadcast tax scales with `num_rollout_ranks`.

## When broadcast is enough

* Single-node jobs where intra-node NVLink saturates the broadcast path.
* Smaller models.
* No RDMA-capable interconnect.

## Pairs with

* [Fault tolerance](fault-tolerance.md).
* [INT4 QAT](int4-qat.md): smaller weights, shorter sync.
