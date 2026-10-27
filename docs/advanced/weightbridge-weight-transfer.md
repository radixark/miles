---
title: WeightBridge Weight Transfer
description: Shard-routed, deduplicated trainer-to-rollout weight transfer that WeightBridge drives end to end.
---
`--update-weight-transfer-mode weightbridge` moves updated weights with
[WeightBridge](https://github.com/harvard-cns/weightbridge) (`pip install wbridge`). It needs
disaggregated placement (no `--colocate`), the Megatron backend, and SGLang engines that support
`--enable-weightbridge-receiver`, which miles turns on for you.

## Usage

```bash
pip install 'wbridge[cuda]'   # plus a Mooncake transfer engine build matching your CUDA version

python train_async.py ... \
  --update-weight-transfer-mode weightbridge \
  --weightbridge-protocol rdma
```

| Flag | Default | Notes |
|---|---|---|
| `--weightbridge-protocol` | `efa` | Data-plane transport: `efa` (Mooncake on AWS EFA), `rdma` (Mooncake RC-QP over InfiniBand/RoCE), `tcp` (Mooncake over TCP, for fabric-less tests), `monarch` (needs Monarch-actor workers). |
| `--weightbridge-async-delivery` | off | Return from an update once the weights are packed; engines apply it whenever it lands. By default the update returns only after every engine rank has loaded it, so the next rollout runs on the new weights. |
| `--pause-generation-mode` | `retract` | How engines clear in-flight requests when an update lands: `retract` requeues them to recompute under the new weights, `abort` aborts them. `in_place` is rejected. |

To validate a transfer path, add `--check-weight-update-equal --debug-exit-after-first-weight-update`:
the engines randomize their weights, receive the first update, compare every tensor with the
checkpoint byte for byte, and the run exits.

## How it works

- **No redundant transfer.** Every trainer rank sends only its own live Megatron shards, with no
  trainer-side gather, directly to the rollout ranks that need them, and each replicated tensor crosses
  the network only once.
- **Balanced transfer at high rollout DP.** Each rollout replica receives only a share of the weights
  from the trainer, and the rollout workers exchange the rest among themselves (over NVLink within a
  node), so trainer egress does not grow with the number of rollout replicas.
- **No per-model mapping code.** WeightBridge infers each side's weight layout by probing the
  framework's own `load_weights` (SGLang's model loader on the rollout side, Megatron-Bridge's
  HF→Megatron import on the trainer side) and verifies it byte for byte when the adapters are built.

The update itself runs without miles' engine session:

1. At startup, each SGLang engine builds a WeightBridge receiver on every worker; engine rank 0 also
   spawns the engine's coordinator, at the engine's HTTP port + 1717. On the first update, every
   trainer rank builds a sender and connects once through the coordinators.
2. Each update is `send_weights(metadata={"weight_version": v})` on every trainer rank. The trainer
   sends no HTTP or Ray request to the engines.
3. Every engine scheduler polls its receiver once per loop pass. When the data arrives, engine rank 0
   relays the decision to all TP/PP ranks, which pause (finish the in-flight batch, retract or abort
   every request, synchronize the device), load, flush the KV cache, set the weight version, and
   resume. Under pipeline parallelism the engine first finishes the micro-batches already in flight.

## Supported layouts

Rollout engines may use TP, PP, attention DP, and span several nodes (miles allocates the control-hub
port and passes `--weightbridge-hub-addr`). Not supported: data-parallel replicas inside one engine,
PD disaggregation, speculative decoding with PP, `--colocate`, LoRA, and `--fully-async`.
