---
title: Disk Offload
description: Spill the paused training actor to node-local disk when host RAM cannot hold it.
corresponding author: Zhichen Zeng (Zhichenzzz)
---

Colocated RL keeps a training actor and a rollout engine on the same GPUs, so the actor
must get out of the way while the engine generates. miles offloads the actor during that
window, and by default the backup lives in pinned host memory. For large models that copy
does not fit, and disk offload is the alternative.

## Usage

```bash
--offload-train --offload-train-target disk \
--offload-train-disk-dir /scratch/miles_offload \
--offload-train-disk-chunk-mb 256
```

Instead of a pinned host copy, the paused actor is streamed to per-rank files through a
fixed-size pinned staging buffer, so host memory stays bounded by
`--offload-train-disk-chunk-mb` regardless of how much is offloaded. Each rank writes to
its own directory under `--offload-train-disk-dir` (defaults to
`$SCRATCH/miles_train_offload_<uid>`), the files are overwritten in place every step, and
they are removed when the actor exits.

Point the directory at real node-local NVMe. A tmpfs mount (including `/tmp` on many
systems) keeps the backup in RAM and defeats the purpose.

## How it works

This runs on [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver), which
hooks the allocator, so it does not care what the memory holds — weights, gradient
buffers and optimizer state all move as one block when the actor is paused, and come back
on resume. miles' part is choosing the per-rank directory, launching each actor with the
matching `TMS_DISK_BACKUP_*` environment, and reclaiming the files at startup and exit.

Because pause and resume happen at phase boundaries, everything is resident again by the
time the optimizer step runs. If the binding constraint is instead that the optimizer
state does not fit the GPU *during* the step, actor offload cannot help; that case is what
`--stream-optimizer-state-to-disk` addresses, and the two compose.

## Streaming the optimizer state

```bash
--offload-train --offload-train-target disk \
--stream-optimizer-state-to-disk \
--offload-train-disk-dir /scratch/miles_offload
```

The two together are what this is for. Actor offload gets the paused actor out of HBM for
the whole rollout window; streaming keeps the largest part of it — 12 bytes per parameter
of fp32 main params and Adam moments — from being in HBM in the first place, including
while the step runs, which offload cannot do. Data parallelism divides the optimizer state,
so a run with GPUs to spare shards it small enough to stay resident and needs neither; the
runs that need both are the ones tight enough to sit at DP=1. GLM-5.2 744B on 8 GB300 nodes
is the shape of it: 279 GB of state per rank against 276.6 GB of HBM, on half the nodes the
run would otherwise need.

The fp32 main params and Adam moments live in per-bucket files instead of on the GPU.
Each step brings in one bucket, updates it, and writes it back, so peak residency is one
bucket rather than the whole state. Buckets are capped at 200M elements independently of
DDP's bucket sizes, which reach tens of GB at DP=1. Native-fp32 model params (a router's
`expert_bias`, a GDN/Mamba `A_log`) stay GPU-resident under a small separate Adam: they
are tiny, and unlike the bf16 path their optimizer shards alias the model params directly.

Initialization follows the same cumulative-state bound without a temporary state file. Megatron
creates each same-shaped CUDA fp32 main-param handle and immediately releases its storage. Once
the complete dense/expert optimizer chain exists, Miles reserves the final main-plus-moments files,
materializes one main-only runtime bucket, copies the still-live bf16 model shards into it, writes
those bytes directly to their final offsets, and releases the bucket before continuing. Each
initialized range is synchronized and advised `DONTNEED`, and the CUDA allocator cache is cleared
around initialization and model-weight reloads. The normal streamed step remains buffered I/O, so
reclaimable page cache is still charged to the trainer's host-memory cgroup.

The handle is allocated on CUDA before its storage can be released, so construction still needs
enough free HBM for the largest individual fp32 shard. Direct final-file initialization then needs
one main-only bucket (normally at most about 200M fp32 elements, except for an oversized entry).
It removes cumulative fp32-main residency; it does not make the largest shard allocation-free.

`fp32` storage is bit-identical to keeping the state on GPU, so turning this on does not
change results — it trades step time for memory. The step is I/O bound and the moments
tolerate less precision than the master copy, so they can be stored narrower:

```bash
--stream-optimizer-state-moment-dtype bf16
```

That cuts streaming volume by a third (12 bytes per param to 8). A checkpoint records the
dtypes it was written with and a resume verifies them, so bytes written as bf16 can never
be read back as fp32. The fp8 options work but are not recommended: `exp_avg_sq` needs
per-block scaling to survive 8-bit storage, which this does not implement.

Streaming requires `--ckpt-format torch_dist`; Megatron's legacy optimizer-checkpoint path reads
the master optimizer state that the NVMe store deliberately leaves empty. It supports BF16 model
training, not FP16, because the streamed optimizer binding does not checkpoint dynamic loss-scaler
state. It also requires `--accumulate-allreduce-grads-in-fp32`: otherwise Megatron creates separate
FP32 gradient tensors for all main parameters before the bucket loop and loses the GPU-memory bound.
Transformer Engine or Apex FusedAdam is required, and Megatron FSDP is not supported.

Other limits to know about. Resume requires the same topology, model definition, and parameter
order — the on-disk layout follows this rank's DP shard and does not reshard optimizer state.
Layout changes detected from the bucket and entry sizes fail explicitly.
A checkpoint written before streaming was enabled cannot be resumed with it: the streamed
state is the only optimizer state read, so miles refuses rather than silently restarting
Adam from zero — pass `--no-load-optim` to accept a fresh optimizer state. And the
optimizer state is copied to the checkpoint directory synchronously, outside
`--async-save`, so expect checkpoint saves to take noticeably longer.

The two also help each other. With the optimizer state already on disk there is that much
less to move when the actor is paused: on Qwen3-30B-A3B, sleep/wake went from 24s/8.9s to
5.2s/1.3s once the paused actor no longer carried the state.

## Choosing

`--offload-train` is the base mechanism, and it is not tied to colocation: the actor is
resident only during `train()` and sleeps for the whole rollout window either way. What
changes is who takes the freed HBM. Colocated, it is the engine on the same GPUs, so
offload is mandatory and defaults on. Disaggregated, it is whatever else you fit on the
training GPUs, which is the point when you are sizing engines against a fixed cluster.

- Paused actor fits in host RAM: keep `--offload-train-target=cpu` (default). Fastest, no
  scratch space.
- It does not fit: `--offload-train-target=disk`.
- On top of that, the optimizer state does not fit the GPU *during the step*, which offload
  cannot help with: add `--stream-optimizer-state-to-disk`, and consider
  `--stream-optimizer-state-moment-dtype bf16` to claw back some of the I/O cost.

Streaming on its own is the disaggregated case: nothing else wants the training GPUs during
rollout, so there is no actor to park and `--offload-train-target` is never read. Pass
`--stream-optimizer-state-to-disk` alone there. Under `--offload-train`, though, the two go
together — a run that cannot hold the optimizer state on the GPU for the step will not hold
a pinned host copy of the whole actor either — and miles asserts that pairing.

Both mechanisms share `--offload-train-disk-dir` and `--offload-train-disk-chunk-mb`. Point the
directory at real node-local NVMe: a tmpfs mount, which `/tmp` is on many systems, keeps
the data in RAM and defeats both. The chunk is a pinned host staging buffer and each
mechanism allocates its own, so enabling both costs 2x that per rank.
