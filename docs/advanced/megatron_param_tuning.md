---
title: Megatron Parameter Tuning
description: Quickly estimate model-state memory, host memory, and token capacity for Megatron training in Miles.
---

Use the largest trainer rank for every estimate. The formulas below are first-order planning numbers; the final configuration must pass a complete measured training step.

<Note>
This guide covers full-parameter fine-tuning only. LoRA and other partial-parameter fine-tuning methods are outside its calculation scope.
</Note>

## Quick estimate

### Inputs

| Symbol | What to use | How to get it |
|---|---|---|
| `N` | All resident model parameter elements on the heaviest rank | Take the largest Megatron startup `COUNT` |
| `Q` | Parameter elements in that rank's distributed-optimizer shard | For one shard group of size `S`, use `N / S`; for MoE, use `N_dense / S_dense + N_expert / S_expert` |

Under this full-parameter fine-tuning scope, every model parameter is trainable, so `N_t = N`. A recipe that deliberately freezes any parameters needs a separate trainable-parameter count and is outside the formulas below.

Megatron prints `N` at startup:

```text
> number of parameters on (tensor, pipeline) model parallel rank (T_RANK, P_RANK): COUNT
```

Use the largest `COUNT`. Before startup, `total_parameters / (TP × PP)` is only a rough dense-model estimate; it misses uneven stages and replicated modules. Never use a MoE model's advertised active-parameter count for weight memory. TODO: Add a dedicated parallel-group sizing guide and link it here.

### Static memory

| Configuration | HBM per trainer rank | CPU memory per trainer rank |
|---|---:|---:|
| Default BF16 + distributed GPU Adam | `6N + 12Q` bytes | — |
| BF16 + full CPU Adam offload | `6N` bytes | `16Q` bytes |

With one shard group, `Q = N/S`, so the default is `(6 + 12/S) × N`; full CPU offload uses `6N` HBM and `16N/S` CPU memory. Divide bytes by `2^30` for GiB.

The `16Q` CPU Adam total contains four FP32 buffers of `4Q` each: the main parameter, first moment, second moment, and persistent input-gradient buffer. The input-gradient buffer receives the optimizer-owned gradient shard before each Adam update; do not add it again under host-memory overhead.

| Planning limit | Initial target | Include in the total |
|---|---:|---|
| HBM | At most `80%` of physical HBM | Static state, activations, CUDA/NCCL buffers, workspaces, fragmentation, weight sync, and offload/onload transitions |
| Host memory | At most `80%` of pre-launch `MemAvailable` | Every local trainer and rollout process, CPU Adam, backups, Ray, pinned buffers, dataloaders, and checkpoint staging |

`80%` is a starting policy, not a guarantee. Use a lower target for long contexts, large token caps, new kernels, or colocated transitions, and do not rely on swap.

## Configuration choices

| Item | Default | Capacity option | Recommendation |
|---|---|---|---|
| Model parameters | BF16 in HBM, `2N` | Tested FP8 recipe | FP8 compute does not make every resident parameter `1 B`; measure the actual recipe |
| Gradients | FP32 in HBM, `4N` | `--grad-reduce-in-bf16` | Avoid for production and convergence CI; use only after numerical validation |
| Adam main parameter and moments | FP32 in HBM, `12Q` | CPU optimizer offload; or `--main-params-dtype fp16`, `--exp-avg-dtype fp16`, `--exp-avg-sq-dtype fp16` | Prefer CPU offload when HBM is tight; low-precision state requires `--use-precision-aware-optimizer` and numerical validation |
| QAT | Disabled | Model-specific fake-quant recipe | Enable only for a target quantized rollout or deployment format; it is not triggered automatically by model size or FP8 |

Recommended CPU Adam flags:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

CPU offload saves HBM but can reduce throughput. The current full CPU-offload path keeps FP32 Adam state on the host, so the FP16 optimizer-state flags do not reduce the `16Q` CPU estimate. See [Low Precision RL](advanced/fp8-low-precision) and [INT4 QAT](advanced/int4-qat) for format-specific setup.

## Additional host memory

| Source | Estimate | When to count it |
|---|---:|---|
| Other `torch_memory_saver` mirrors | Runtime-dependent | Count eligible allocations that are not explicitly excluded below when training offload is enabled |
| Flat gradient-buffer TMS backup | `0` in current Miles Megatron paths, avoiding about `4N` with default FP32 gradients | `offload_train=True` sets `disable_grad_buffers_cpu_backup=True`; without training offload, Miles does not enable TMS |
| Weight and parameter backups | About `2N` for the actor-sized copy, plus the tags below | Count the copy owner and every additional model-switching tag |
| Runtime overhead | Measure per node | Count Ray object store, pinned transfers, dataloaders, page cache, and checkpoint transients |

For default BF16 colocate with training offload, `--disable-weights-backuper` changes which subsystem owns the actor-sized CPU copy but does not remove that copy:

| Configuration | `TensorBackuper` `actor` copy | TMS `param_buffer` copy | Actor-sized CPU total |
|---|---:|---:|---:|
| Default | About `2N` | `0` | About `2N` |
| `--disable-weights-backuper` | `0` | About `2N` | About `2N` |

With explicit `--no-offload-train`, Miles does not enable TMS, so `--disable-weights-backuper` can remove the actor-sized CPU copy, but the trainer remains resident in HBM.

`MegatronTrainRayActor._enable_weight_backup` is true for colocate, reference-model KL, Megatron OPD teacher, or old-actor switching. With the default weight backuper enabled, each additional BF16 tag adds about `2N`:

| User configuration | Backup tag | Incremental CPU memory |
|---|---|---:|
| `--use-kl-loss` or nonzero `--kl-coef`; requires `--ref-load` | `ref` | `+2N` |
| `--use-opd --opd-type megatron`; requires `--opd-teacher-load` | `teacher` | `+2N` |
| `--keep-old-actor` | `old_actor` | `+2N` |
| `--keep-old-actor --update-weights-interval 1` | `rollout_actor` | `+2N` |

For combined conditions, count the union of created tags. These additional tags require the default weight backuper; do not use `--disable-weights-backuper` with model-switching features.

Sum these values across all local processes before comparing with `MemAvailable`.

## Tune `--max-tokens-per-gpu`

Activation memory is configuration- and data-dependent. Measure it with the exact production recipe instead of deriving it from parameter count.

| Step | Action |
|---:|---|
| 1 | Set the planning budget `B`, initially `0.8 × physical_HBM` |
| 2 | Run two safe token caps `K1 < K2`; include warmup, forward, backward, optimizer step, weight sync, and offload/onload transitions |
| 3 | Record `torch.cuda.max_memory_reserved()` on every rank; use separate runs or call `torch.cuda.reset_peak_memory_stats()` before each measurement |
| 4 | Estimate the slope and candidate cap with the formulas below |
| 5 | Round down, then validate the longest samples and representative packing patterns; use binary search if the peak is nonlinear |

$$
\alpha = \frac{M(K_2)-M(K_1)}{K_2-K_1}, \qquad K_{candidate} = K_1 + \frac{B-M(K_1)}{\alpha}.
$$

Miles packs a CP group against approximately `CP × max_tokens_per_gpu`, but the flag remains a per-GPU target. Always validate the maximum observed rank rather than assuming perfectly balanced tokens.
