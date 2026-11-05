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
| DSA sparse attention and indexer scores | `--dsa-attention-backend cake` | `--cake-dsa-variant {det,fast}`, `--cake-dsa-bwd-workspace-bytes` (FP32 key-side partials of the `det` backward, default 2 GiB), `--cake-dsa-lazy-build` | GLM-5 / GLM-5.1 / GLM-5.2 / DeepSeek-V3.2 and GLM-5.3-Flash (`--dsa-impl miles`), DeepSeek-V4-Flash and DeepSeek-V4.1-Flash (`--dsv4-impl miles`) | `--megatron-to-hf-mode raw` without multi-LoRA (under `bridge` the flag's `tilelang` and `megatron` choices pick the Bridge kernels), Megatron backend, a 512-wide latent with a 0 or 64 RoPE tail, 8 / 16 / 32 / 64 indexer heads of dimension 128. `fast` serves 64 TP-local query heads without an attention sink and runs `det` for every other call (DeepSeek-V4 / V4.1 layers have a sink, so they run `det`); the indexer scores are always `det`. DeepSeek-V4 needs `V4_INDEXER_IMPL` unset. `--cake-dsa-lazy-build` gives up the build before the first step: each kernel builds on its first launch, inside a training step. |
| Batched per-head projection GEMM | `--batched-projection-gemm-backend cake` | none (deterministic only) | the miles DSA plugin layers: the MLA q-absorb and value projections of GLM-5 / GLM-5.1 / GLM-5.2 / DeepSeek-V3.2 and GLM-5.3-Flash, the grouped output projection of DeepSeek-V4-Flash / V4.1-Flash | Megatron backend, `--megatron-to-hf-mode raw` with `--dsa-impl` / `--dsv4-impl miles`, BF16 parameters, head dimensions that are multiples of 8, no CUDA graphs. Bit-identical call to call, not bitwise equal to cuBLAS. |
| MoE router gate GEMM (FP32) | `--moe-router-gemm-backend cake` | none (deterministic only) | every MoE model on the Megatron backend whose routers use Megatron's `Router.gating` | `--moe-router-dtype fp32`, `--disable-bias-linear`, BF16 parameters, no CUDA graphs, expert counts and hidden sizes that are multiples of 8; a router that overrides `Router.gating` refuses it when the model is built. FP32 logits and gradients in a fixed reduction order, not bitwise equal to Transformer Engine's GEMM. |
| MoE routed-expert grouped GEMM | `--moe-grouped-gemm-backend cake` | `--cake-moe-grouped-gemm-variant {det,fast}` (`fast` adds part of the weight-gradient tiles in a run-dependent order; the output and input gradient are `det`'s bits) | the routed experts (`w13`, `w2`) of the MoE models of `--train-backend torchtitan` | torchtitan backend, expert widths that are multiples of 256; quantized or adapted grouped linears and grouped linears outside the routed experts are refused when the model config is built. Same parameters and names, so checkpoints load unchanged; not bitwise equal to `torch._grouped_mm`. |
| KDA chunked delta attention | `--kda-backend cake` | none (`det` only) | Kimi-K3, GLM-5.3-Flash | Megatron backend, `--qkv-format thd`, no context parallelism, 1 to 64 KDA heads per rank (Kimi-K3's 96 heads need tensor parallelism of at least 2; GLM-5.3-Flash's KDA layer is not head-sharded). The kernels take a BF16 beta, so numerics differ slightly from fla's FP32 beta. |
| Packed causal GQA attention (head dim 256) | `--full-attention-backend cake` | `--cake-gqa-check-seqlens` (checks the packed sequence boundaries and the longest key segment at every call, one device synchronization each) | the full-attention layers (and the MTP layer) of Qwen3.5 / Qwen3.6 and Qwen3-Next | Megatron backend, `--megatron-to-hf-mode raw`, `--qkv-format thd`, no context parallelism, no attention dropout or `--window-size`, an even number of query heads per key/value head on each tensor-parallel rank. One variant: dQ is summed with FP32 atomic adds, so it is not bit-reproducible and `--deterministic-mode` rejects it; the output and dK / dV repeat bit for bit. Same state-dict keys as Transformer Engine's attention, so checkpoints load unchanged. |
| mHC hyper connection (four-stream prepare and finish, stream-mean head) | `--mhc-backend cake` | `--cake-mhc-variant {det,fast}` (`fast` sums the mapping-weight gradient with FP32 atomics) | GLM-5.3-Flash (`get_glm5_next_spec`, also its stream-mean head), DeepSeek-V4-Flash / V4-Pro (`get_dsv4_spec`, either `--dsv4-impl`) | Megatron backend, 4 streams, hidden size 4096 / 5120 / 7168 (SM107a: 4096 only), `--hidden-dropout 0`, no `mhc` in `--recompute-modules`, no CUDA graphs, no `--use-fused-mhc`, no `--fp32-residual-connection`, no `--overlap-moe-expert-parallel-comm`. Same parameters and names as Megatron's module, so checkpoints load unchanged; the coefficients stay FP32 where Megatron casts them to BF16. DeepSeek-V4.1's own hyper connection is not served. |
| Short convolution (causal depthwise conv1d + SiLU) | `--linear-attention-conv-backend cake` | `--cake-conv1d-variant {det,fast}`, `--cake-conv1d-silu-grad-preactivation {bf16,fp32}` (default `bf16`, fla's rounding point; `fp32` is more exact and slower) | the GDN and KDA layers above | Kernel size 4, channels divisible by 8, no context parallelism. Overrides the fla conv backend that `--linear-attention-backend cake` pins. |
| SiLU-gated output RMSNorm (`RMSNorm(x) * w * SiLU(z)`) | `--linear-attention-output-norm-backend cake` | `--cake-gated-rmsnorm-variant {det,fast}` (`fast` adds the weight gradient with FP64 atomics; `dx` and `dz` are `det`'s bits) | the GDN layers above (SiLU gate) | Value head dimension 128, BF16 weight, no bias. A sigmoid-gated norm refuses it when the layer is built: the KDA layers (Kimi-K3, GLM-5.3-Flash) and a Qwen3.8-Next config whose output gate is a sigmoid. |
| Zero-centred input RMSNorm (`x * rstd * (1 + w)`) | `--linear-attention-input-norm cake` | `--cake-rmsnorm-variant {det,fast}` (`fast` adds the weight gradient with FP64 atomics in some token windows) | the Qwen3.5 / Qwen3-Next GDN layers | The hidden sizes the kernels ship: 2048, 4096 and 5120 among the Qwen3.5 widths, not Qwen3.5-4B's 2560 or 0.8B's 1024; the layer refuses others when it is built. On SM107a, hidden size 4096 needs a CUDA 13.4 or newer runtime in the process (a `cu130` PyTorch wheel carries 13.0) and a driver that allows its oversized shared memory; otherwise the layer refuses it when it is built. Same single `weight` parameter as the `te` and `hf` norms, so checkpoints load unchanged. |

## Tests

`tests/fast-gpu/kernels/test_miles_ext_kernels.py` runs the submodule's own one-GPU tests (`tests/cake/_runtime` and
`tests/cake/<op>`), one pytest process each, and `tests/fast-gpu/kernels/test_miles_ext_kernels_multi_gpu.py` its
`multi_gpu` tests on four GPUs. Both fail on any skip but the hardware limits that
`tests/fast-gpu/kernels/miles_ext_kernels_skips.py` lists for the skipped test's op, such as a test that needs more
GPUs than the runner has. Each family's miles layer has its own GPU test in
`tests/fast-gpu/` (`test_cake_*.py`, `test_dsa_cake_backend.py`). End to end on Blackwell:

| Test | Families |
|---|---|
| `tests/e2e/megatron/test_qwen3_5_4B_gdn_cake.py` | GDN |
| `tests/e2e/megatron/test_qwen3_5_4B_cake_kernels.py` | GDN, short convolution, gated output norm, GQA |
| `tests/e2e/megatron/model_scripts/test_kimi_k3_4layer_cake_ci.py` | KDA, short convolution |
| `tests/e2e/megatron/model_scripts/test_deepseek_v4_flash_4layer_cake_ci.py` | mHC |
| `tests/e2e/megatron/model_scripts/test_glm5_2_5layer_cake_ci.py` | batched projection, MoE router gate |
| `tests/e2e/torchtitan/test_qwen3_30B_A3B_cake_grouped_gemm.py` | MoE grouped GEMM |

The input norm and DSA have no end-to-end test yet. The Qwen3.5 tests above use Qwen3.5-4B, whose hidden size 2560
the input-norm kernels do not serve; Qwen3.5-35B-A3B's 2048 is served, but no end-to-end test runs it with them. The
DSA `det` kernels build through torch's C++ extensions, which CI does not cache across jobs.
