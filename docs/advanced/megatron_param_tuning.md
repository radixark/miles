---
title: Megatron Parameter Tuning
description: Quickly estimate model-state memory, host memory, and token capacity for Megatron training in Miles.
---

Use this guide to answer two questions before launching a Megatron job: whether the trainer state fits, and how to choose `--max-tokens-per-gpu`. The formulas are first-order estimates; the final decision must use the maximum observed rank from a complete training step.

## 1. Get three numbers

Use these values for the most heavily loaded trainer rank:

| Symbol | Meaning |
|---|---|
| `N` | All resident model parameter elements on the rank |
| `N_t` | Trainable parameter elements on the rank; equal to `N` for full-parameter training |
| `Q` | Trainable parameter elements owned by the rank's distributed optimizer shard |

For full-parameter training with one optimizer shard group of size `S`, use `N_t = N` and `Q = N / S`. MoE models can have different dense and expert shard groups; in that case use `Q = N_dense / S_dense + N_expert / S_expert`.

Megatron prints the constructed parameter count for every TP/PP rank at startup:

```text
> number of parameters on (tensor, pipeline) model parallel rank (T_RANK, P_RANK): COUNT
```

Use the largest `COUNT` as `N`. A rough dense-model estimate is `total_parameters / (TP × PP)`, but it can miss embeddings, output heads, uneven pipeline stages, and replicated modules. Do not use a MoE model's advertised active-parameter count for weight memory.

This document does not derive parallel groups. Start from a tested recipe and see [Parallelism compatibility](/user-guide/usage#parallelism-compatibility) when changing TP, PP, CP, EP, or expert TP.

## 2. Estimate static memory

Miles defaults to BF16 model parameters, FP32 gradients, distributed Adam, and FP32 optimizer state. The steady-state HBM estimate is:

$$
M_{HBM} \approx 2N + 4N_t + 12Q \quad \text{bytes}.
$$

For full-parameter training with one shard group, this is `(6 + 12/S) × N` bytes. Divide by `2^30` for GiB.

| State | Approximate memory | Default placement |
|---|---:|---|
| BF16 model parameters | `2N` | HBM |
| FP32 gradients | `4N_t` | HBM |
| FP32 main parameters and two Adam moments | `12Q` | HBM |

This estimate excludes activations, CUDA/NCCL buffers, allocator fragmentation, kernel workspaces, padding, and temporary weight-sync or offload peaks.

### CPU optimizer offload

For large production and capacity-oriented CI jobs, use CPU Adam when optimizer state would consume too much HBM:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

With full CPU offload, estimate:

$$
M_{HBM} \approx 2N + 4N_t, \qquad M_{CPU,optimizer} \approx 16Q \quad \text{bytes}.
$$

The CPU estimate includes the FP32 main parameter, two Adam moments, and gradient staging. Sum it across every trainer rank on the node. CPU offload preserves HBM headroom but can reduce throughput, so benchmark GPU Adam when both configurations fit.

### Precision choices

| Choice | Recommendation |
|---|---|
| Default BF16 weights, FP32 gradients, FP32 Adam state | Use for production and convergence CI |
| `--grad-reduce-in-bf16` | Avoid for production and convergence CI; use only for validated fit-only jobs |
| `--main-params-dtype fp16`, `--exp-avg-dtype fp16`, `--exp-avg-sq-dtype fp16` | Require `--use-precision-aware-optimizer`; use only after numerical comparison |
| FP8 training | Use a tested hardware/model recipe; FP8 compute does not make every resident parameter `1 B` |
| QAT | Enable only for a target quantized rollout or deployment format; it is not triggered automatically by model size or FP8 |

The current full CPU-offload path keeps its Adam state in FP32 host memory, so the FP16 optimizer-state flags do not reduce the `16Q` CPU estimate. See [Low Precision RL](/advanced/fp8-low-precision) and [INT4 QAT](/advanced/int4-qat) for format-specific setup.

## 3. Add host-memory overhead

In addition to CPU Adam, include these consumers when enabled:

- `torch_memory_saver` mirrors eligible allocations when training is offloaded; Miles excludes the flat gradient buffer, but the remaining amount is runtime-dependent.

- Explicit weight backups can add approximately one local model shard per backup tag.

- Standard full-model colocated SGLang does not automatically keep another full CPU model copy; LoRA with `--lora-base-cpu-backup` is an explicit exception.

At the node level, add all trainer ranks, rollout processes, Ray object-store memory, pinned transfer buffers, dataloaders, and checkpoint staging. Use pre-launch `MemAvailable` from `/proc/meminfo`, not total installed RAM. Keeping planned usage below roughly `80%` of `MemAvailable` is a starting policy, not a universal safe limit, and the job should not rely on swap.

## 4. Calibrate activation memory

Activation memory depends on tokens, model shape, recomputation, attention kernels, pipeline scheduling, and MoE routing. It is more reliable to measure its slope than to derive it from parameter count.

1. Set the planning budget to `B = 0.8 × physical_HBM`; use a lower fraction for long contexts, large token caps, new kernels, or colocated transitions.

2. Run the exact recipe at two safe token caps `K1 < K2`. Include warmup, a full forward/backward/optimizer step, weight sync, and offload/onload transitions. Use separate runs or call `torch.cuda.reset_peak_memory_stats()` before each measurement.

3. Record `torch.cuda.max_memory_reserved()` on every rank and use the largest rank. Estimate:

$$
\alpha = \frac{M(K_2)-M(K_1)}{K_2-K_1}, \qquad K_{candidate} = K_1 + \frac{B-M(K_1)}{\alpha}.
$$

4. Round `K_candidate` down and validate it with representative longest samples and packing patterns. Use binary search when the measured peak is nonlinear.

Miles packs a CP group against approximately `CP × max_tokens_per_gpu`; the flag remains a per-GPU target. Always validate the actual maximum rank rather than assuming perfectly balanced tokens.

## Quick checklist

- Take the largest startup parameter count, not the average rank or advertised active parameters.

- Calculate default HBM with `2N + 4N_t + 12Q`.

- With CPU Adam, calculate HBM with `2N + 4N_t` and host optimizer memory with `16Q`.

- Add backup, Ray, rollout, checkpoint, activation, workspace, and transition peaks.

- Keep initial HBM and host plans below roughly `80%`, then confirm them with a complete measured step.
