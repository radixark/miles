---
title: Cake Training Kernels
description: Opt-in Cake training kernels for Blackwell and Rubin from the miles-ext-kernels submodule, one 'cake' backend choice per kernel family.
---
The miles-ext-kernels submodule (`3rdparty/miles-ext-kernels`, Python package `miles_ext_kernels`) carries Cake
training kernels for NVIDIA Blackwell (SM100a: B200 / GB200; SM103a: B300 / GB300) and Rubin (SM107a: R200). Every
kernel family is opt-in: its backend argument gains a `cake` choice, and every default stays as it was.

## Enable

```bash
git submodule update --init 3rdparty/miles-ext-kernels
# training arguments: the flag, then one or more family backends (table below)
--use-miles-ext-kernels --linear-attention-backend cake
```

- Every `cake` choice requires `--use-miles-ext-kernels`. The flag also registers the kernel package's own
  arguments (`--cake-<family>-...`), which the package validates at startup.
- The miles Ray launcher puts the submodule on the job's `PYTHONPATH` for runs that pass the flag, and the CUDA
  images other than `radixark/miles:dev-rubin` install it. A launch outside both needs `3rdparty/miles-ext-kernels`
  on `PYTHONPATH` on every node.

## Variants

A family that ships two variants takes `--cake-<family>-variant {det,fast}`:

- `det`: the bit-deterministic kernels. Repeated passes on the same inputs give the same output and gradient bits.
- `fast`: the faster sibling. Most `fast` kernels reduce in a run-dependent order.
- Unset, the variant is `fast`, or `det` under `--deterministic-mode`.
- An explicit `fast` under `--deterministic-mode` is a startup error for a family whose `fast` reduces in a
  run-dependent order.
- A call that `fast` does not serve (a shape, a layout, context parallelism) runs `det` and logs the reason once.

## Kernel builds

The kernels compile with nvcc on first use, so the first step of a run pays the build:

- tvm-ffi libraries go to `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/`;
- PyTorch extensions go to `TORCH_EXTENSIONS_DIR`.

SM107a needs a CUDA 13.4 or newer nvcc (`CUDA_HOME`, else `nvcc` on `PATH`). An older toolkit, such as the CUDA 13.0
one of the `radixark/miles:dev` image, builds the SM100a and SM103a kernels only; an SM107a build under it fails
with an error naming the toolkit it needs.

On multi-node runs, point both at node-local or per-job storage. Concurrent builds on one node share the cache
through file locks.

## Kernel families

| Family | Backend argument | Package arguments | Models | Limits |
|---|---|---|---|---|
| GDN chunked gated delta rule | `--linear-attention-backend cake` | `--cake-gdn-variant {det,fast}`, `--cake-gdn-recompute-state` | Qwen3.5 / Qwen3.6, Qwen3-Next, Qwen3.8-Next | K = V = 128. `fast` is fixed-order (bit-identical run to run, allowed under `--deterministic-mode`); context parallelism runs `det`. See [Qwen3.5](/models/qwen/qwen3-5). |
| KDA chunked delta attention | `--kda-backend cake` | none (`det` only) | Kimi-K3, GLM-5.3-Flash | Megatron backend, `--qkv-format thd`, no context parallelism, 1 to 64 KDA heads per rank (Kimi-K3's 96 heads need tensor parallelism of at least 2; GLM-5.3-Flash's KDA layer is not head-sharded). The kernels take a BF16 beta, so numerics differ slightly from fla's FP32 beta. |
| Short convolution (causal depthwise conv1d + SiLU) | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant {det,fast}`, `--cake-conv1d-silu-grad-preactivation {bf16,fp32}` (default `bf16`, fla's rounding point; `fp32` is more exact and slower) | the GDN and KDA layers above | Kernel size 4, channels divisible by 8, no context parallelism. Overrides the fla conv backend that `--linear-attention-backend cake` pins. |

## Tests

`tests/fast-gpu/kernels/test_miles_ext_kernels.py` runs the submodule's own GPU tests (`tests/<family>`), one pytest
process per family, and fails on any skip.
