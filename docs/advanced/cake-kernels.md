---
title: Cake Training Kernels
description: Opt-in Cake training kernels for Blackwell and Rubin from the miles-ext-kernels submodule, one 'cake' backend choice per kernel family.
---
The miles-ext-kernels submodule (`3rdparty/miles-ext-kernels`, Python package `miles_ext_kernels`) carries Cake
training kernels for NVIDIA Blackwell (SM100a: B200 / GB200; SM103a: B300 / GB300) and Rubin (SM107a: R200), one
module per kernel family (`miles_ext_kernels.cake.<op>`). Every kernel family is opt-in: its backend argument gains a
`cake` choice, and every default stays as it was.

## Enable

```bash
git submodule update --init 3rdparty/miles-ext-kernels
# training arguments: the flag, then one or more family backends (table below)
--use-miles-ext-kernels --linear-attention-backend cake
```

- Every `cake` choice requires `--use-miles-ext-kernels`. The flag also registers the kernel package's own
  arguments (`--cake-<family>-...` and `--cake-workspace-pool-bytes`), which the package validates at startup;
  miles hands their values to `miles_ext_kernels.configure` in every process that builds a `cake` layer.
- The miles Ray launcher puts the submodule on the job's `PYTHONPATH` for runs that pass the flag, and the CUDA
  images other than `radixark/miles:dev-rubin` install it. A launch outside both needs `3rdparty/miles-ext-kernels`
  on `PYTHONPATH` on every node.

## Variants

A family that ships two variants takes `--cake-<family>-variant {det,fast}`:

- `det`: the bit-deterministic kernels. Repeated passes on the same inputs give the same output and gradient bits.
- `fast`: the faster sibling. Most `fast` kernels reduce in a run-dependent order.
- Unset, the variant is `fast`, or `det` under `--deterministic-mode`.
- An explicit `fast` under `--deterministic-mode` is a startup error.
- A call that `fast` does not serve (a shape, a layout, context parallelism) runs `det` and logs the reason once.

## Kernel builds

The kernels compile with nvcc when miles builds a `cake` layer (`miles_ext_kernels.cake.prebuild`): every library the
layer may launch, for any token count, is built before the first step, so no training step compiles. The libraries
go to `${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/`; a later run finds them there.

SM107a needs a CUDA 13.4 or newer nvcc (`CUDA_HOME`, else `nvcc` on `PATH`). An older toolkit, such as the CUDA 13.0
one of the `radixark/miles:dev` image, builds the SM100a and SM103a kernels only; an SM107a build under it fails
with an error naming the toolkit it needs.

On multi-node runs, point `TVM_FFI_CACHE_DIR` at node-local or per-job storage. Concurrent builds on one node share
the cache through file locks.

## Memory

The kernels take most scratch buffers from one workspace pool per device, which keeps at most
`--cake-workspace-pool-bytes` between calls. The default is `0`: the pool keeps nothing, and every scratch buffer is a
new allocation from PyTorch's caching allocator (on B200 and GB300, a 1 GiB pool made no GDN training step faster and
kept up to 0.75 GiB per device). To reuse buffers across calls, raise the bound, for example
`--cake-workspace-pool-bytes 1073741824` (1 GiB per device); the pool then evicts the least recently used buffers
beyond it. A few kernel packages keep buffers of their own outside the pool, which the bound does not count. With
`--offload-train`, miles calls `miles_ext_kernels.release()` when the trainer sleeps, before it offloads: the pool,
the kernels' cached plans and the larger buffers the packages keep of their own are freed.

## Kernel families

| Family | Backend argument | Package arguments | Models | Limits |
|---|---|---|---|---|
| GDN chunked gated delta rule | `--linear-attention-backend cake` | `--cake-gdn-variant {det,fast}`, `--cake-gdn-recompute-state` | Qwen3.5 / Qwen3.6, Qwen3-Next, Qwen3.8-Next | K = V = 128. Context parallelism through fla's state passing runs `det`. See [Qwen3.5](/models/qwen/qwen3-5). |
| KDA chunked delta attention | `--kda-backend cake` | none (`det` only) | Kimi-K3, GLM-5.3-Flash | Megatron backend, `--qkv-format thd`, no context parallelism, 1 to 64 KDA heads per rank (Kimi-K3's 96 heads need tensor parallelism of at least 2; GLM-5.3-Flash's KDA layer is not head-sharded). The kernels take a BF16 beta, so numerics differ slightly from fla's FP32 beta. |
| Short convolution (causal depthwise conv1d + SiLU) | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant {det,fast}`, `--cake-conv1d-silu-grad-preactivation {bf16,fp32}` (default `bf16`, fla's rounding point; `fp32` is more exact and slower) | the GDN and KDA layers above | Kernel size 4, channels divisible by 8, no context parallelism. Overrides the fla conv backend that `--linear-attention-backend cake` pins. |
| SiLU-gated output RMSNorm (`RMSNorm(x) * w * SiLU(z)`) | `--linear-attention-output-norm-backend cake` | `--cake-gated-rmsnorm-variant {det,fast}` (`fast` adds the weight gradient with FP64 atomics; `dx` and `dz` are `det`'s bits) | the GDN layers above (SiLU gate) | Value head dimension 128, BF16 weight, no bias. A sigmoid-gated norm refuses it when the layer is built: the KDA layers (Kimi-K3, GLM-5.3-Flash) and a Qwen3.8-Next config whose output gate is a sigmoid. |
| Zero-centred input RMSNorm (`x * rstd * (1 + w)`) | `--linear-attention-input-norm cake` | `--cake-rmsnorm-variant {det,fast}` (`fast` adds the weight gradient with FP64 atomics in some token windows) | the Qwen3.5 / Qwen3-Next GDN layers | The hidden sizes the kernels ship: 2048, 4096 and 5120 among the Qwen3.5 widths, not Qwen3.5-4B's 2560 or 0.8B's 1024; the layer refuses others when it is built. Same single `weight` parameter as the `te` and `hf` norms, so checkpoints load unchanged. |

## Tests

`tests/fast-gpu/kernels/test_miles_ext_kernels.py` runs the submodule's own one-GPU tests (`tests/cake/_runtime` and
`tests/cake/<op>`), one pytest process each, and `tests/fast-gpu/kernels/test_miles_ext_kernels_multi_gpu.py` its
`multi_gpu` tests on four GPUs. Both fail on any skip but the hardware limits that
`tests/fast-gpu/kernels/miles_ext_kernels_skips.py` lists for the skipped test's op, such as a test that needs more
GPUs than the runner has.
