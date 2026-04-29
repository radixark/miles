---
title: P2P Weight Transfer
description: Direct rank-to-rank weight sync from actor to rollout via RDMA writes.
---

# P2P Weight Transfer

After every training step, the trainer's updated weights have to reach the
SGLang engines so the next rollout is on-policy. The default broadcast path
sends each weight tensor to every rollout rank, which wastes bandwidth at
large model scales. P2P transfer delivers each shard directly to the rank
that needs it, using RDMA writes via a transfer engine
(`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/`).

## Enable

```bash
--update-weight-transfer-mode p2p
```

`--update-weight-transfer-mode` accepts `broadcast` (default) or `p2p`
(`miles/utils/arguments.py:507`).

## What changes

| | Broadcast | P2P |
|---|---|---|
| Per-rank receives | All weights, then drops what it doesn't need | Only its own shards |
| Implementation | NCCL collective (`broadcast.py`) | RDMA writes via mooncake transfer engine + `ThreadPoolExecutor` (`p2p_transfer_utils.py`) |
| Best-fit scale | Small to mid models | Large models with many rollout ranks |

## How it works

The P2P path lives in
`miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`
(class `UpdateWeightP2P`):

1. **Plan.** `RemoteTransferPlan(args, model)` maps trainer ranks to rollout
   ranks based on TP/EP/PP sharding and GPU counts.
2. **Gather and convert.** Megatron TP/EP shards are gathered and converted
   to HF parameter layout (`DistBucketedWeightUpdateMixin`, shared with the
   broadcast path).
3. **Register.** CPU pinned buffers are registered with the transfer engine
   (`register_cpu_memory`) and the engine is initialised over RDMA
   (`transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")` in
   `p2p_transfer_utils.py:211`).
4. **Submit and wait.** Per-rank tensors are submitted to a `P2PTransferManager`
   thread pool (default 4 workers); each worker performs an RDMA write to
   the destination rollout rank's pre-registered memory. `wait_transfers()`
   waits on each future with `transfer_timeout`.
5. **Bump version.** When all transfers complete, the rollout engine's
   weight version is incremented and generation resumes.

## Hardware

P2P uses RDMA, so the transport requires InfiniBand or RoCEv2 between nodes.
On a TCP-only backplane the transfer engine cannot initialise the RDMA
session.

## Tunable knobs

```bash
--p2p-transfer-num-workers 4           # ThreadPoolExecutor workers for RDMA writes (default 4)
--p2p-transfer-timeout 30              # per-future timeout in seconds (default 30.0)
--update-weight-buffer-size 536870912  # bytes per update-weight buffer (default 512 MiB)
--update-weights-interval 1            # rollouts between weight syncs (default 1)
```

All four are defined in `miles/utils/arguments.py` and consumed in
`p2p.py:73` and the bucketed update-weight path.

## Diagnostics

When a P2P transfer's future raises (timeout, RDMA error), the manager logs
`[P2P] Transfer future failed: ...` and continues with the remaining
futures (`p2p_transfer_utils.py`'s `wait()` method). There is no automatic
broadcast-mode fallback in the source today; mode is decided once at startup
by `--update-weight-transfer-mode`.

For per-rank diagnostics, run with `NCCL_DEBUG=INFO`.

## When P2P helps

* Large models where the broadcast path is bandwidth-bound.
* Many rollout ranks: the broadcast tax scales with `num_rollout_ranks`.

## When broadcast is enough

* Single-node jobs where intra-node NVLink saturates the broadcast path.
* Smaller models.
* No RDMA-capable interconnect.

## Pairs with

* [Fault tolerance](fault-tolerance.md). Failed P2P transfers are surfaced
  through `--p2p-transfer-timeout` and engine recovery happens at the next
  weight update (`actor.py:500`).
* [INT4 QAT](int4-qat.md). Smaller weights mean less data per sync.
