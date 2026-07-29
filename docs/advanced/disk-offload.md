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
Megatron's `--optimizer-state-nvme-dir` addresses (see
[radixark/Megatron-LM#63](https://github.com/radixark/Megatron-LM/pull/63)), and the two
compose.

## Choosing

- Host RAM holds the paused actor: keep the default `--offload-train-target=cpu`. It is
  faster and needs no scratch space.
- Host RAM does not hold it: switch to `--offload-train-target=disk`.
