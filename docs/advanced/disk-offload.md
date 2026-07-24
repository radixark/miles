---
title: Disk Offload
description: Spill the training actor, and optionally the optimizer state, to node-local disk when host RAM or GPU memory runs out.
corresponding author: Zhichen Zeng (Zhichenzzz)
---

Colocated RL keeps a training actor and a rollout engine on the same GPUs, so the actor
must get out of the way while the engine generates. miles offloads the actor during that
window, and by default the backup lives in pinned host memory. For large models that
copy does not fit — either because host RAM is exhausted, or because the optimizer state
alone exceeds what the GPU can hold even for a single step.

Two independent mechanisms cover those two different limits:

| | What moves | When | Enable with |
|---|---|---|---|
| Actor offload | The whole paused actor: weights, grad buffers, optimizer state | Between rollout and training, while the actor is idle | `--offload-train-target=disk` |
| Optimizer-state streaming | fp32 main params and Adam moments, one bucket at a time | Inside `optimizer.step()`, while it computes | `--optimizer-state-nvme-dir` |

They are orthogonal and compose. Pick by which limit you are hitting.

## Actor offload to disk

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

This runs on [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver), which
hooks the allocator, so it does not care what the memory holds — it moves the whole
paused actor as one block.

## Optimizer-state streaming

```bash
--optimizer-state-nvme-dir /scratch/miles_optimizer_state \
--optimizer-state-nvme-chunk-mb 256
```

Actor offload cannot help when the optimizer state does not fit the GPU *while the step
runs*: `pause`/`resume` happen at phase boundaries, so by the time the Adam kernel
launches, everything is resident again. Streaming solves that instead — the fp32 main
params and Adam moments live in per-bucket files, and each step brings in one bucket,
updates it, and writes it back, so peak residency is one bucket rather than the whole
state.

Buckets are capped at 200M elements independently of DDP's bucket sizes, which reach
tens of GB at DP=1. Native-fp32 model params (a router's `expert_bias`, a GDN/Mamba
`A_log`) stay GPU-resident under a small separate Adam: they are tiny, and unlike the
bf16 path their optimizer shards alias the model params directly.

The cost is real disk traffic on every step, so only enable this when the state genuinely
does not fit. It requires the default (non-precision-aware) optimizer and is mutually
exclusive with `--optimizer-cpu-offload` and `--offload-optimizer-states`.

### Narrower moments

The step is I/O bound, and Adam's moments tolerate less precision than the master copy,
so they can be stored narrower than the fp32 tensors the optimizer computes on:

```bash
--optimizer-state-nvme-moment-dtype bf16
```

`bf16` cuts streaming volume by a third (12 bytes per parameter to 8): on
Qwen3.5-35B-A3B it takes the state from 46.1 GB to 30.8 GB per rank, with rollout-vs-train
logprob agreement unchanged. Whether that shows up as a faster step depends entirely on
whether your disk is the bottleneck — with 8 ranks sharing one array we measured the same
configuration's step anywhere between 19s and 35s, so measure your own setup rather than
assuming the byte reduction translates.

`fp32` is the default and is bit-identical to keeping the state on GPU. The fp8 options
halve the moments again but are not recommended: `exp_avg_sq` needs per-block scaling to
survive 8-bit storage, which this does not implement, and a 3-rollout smoke test is far
too short to expose the drift that would cause.

A checkpoint records the dtypes it was written with and a resume verifies them, so
bytes written as bf16 can never be read back as fp32.

## Combining the two

```bash
--offload-train --offload-train-target disk \
--offload-train-disk-dir /scratch/miles_offload \
--optimizer-state-nvme-dir /scratch/miles_optimizer_state
```

The two reinforce each other: with the optimizer state already on disk, there is that
much less to move when the actor is paused, so `sleep`/`wake_up` also get faster.

Whether you need both depends on whether you are colocated.

Without `--colocate` the actor owns its GPUs for the whole run, so nothing has to get out
of the way and streaming stands on its own: `--optimizer-state-nvme-dir` alone is the
right configuration, and actor offload buys you nothing.

Under `--colocate` streaming alone is not enough. Without `--offload-train` the actor's
weights and grad buffers stay resident through the rollout, the engine has nowhere to put
its KV cache, and it fails to resume its memory. If you are colocated and reach for
streaming, keep actor offload on.

## Choosing

- Not colocated: you only ever need `--optimizer-state-nvme-dir`, and only if the
  optimizer state does not fit the GPU during the step.
- Colocated, host RAM holds the actor: the default `--offload-train-target=cpu` is
  fastest. Do not reach for disk.
- Colocated, host RAM does not hold the actor: add `--offload-train-target=disk`.
- The optimizer state does not fit the GPU during the step: add
  `--optimizer-state-nvme-dir`, and consider `--optimizer-state-nvme-moment-dtype bf16` to
  claw back some of the I/O cost.
