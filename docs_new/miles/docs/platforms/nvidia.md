---
title: NVIDIA H / B Series
description: H100, H200, B100, B200 — Miles's primary target.
---

# NVIDIA H / B Series

Hopper (H100/H200) and Blackwell (B100/B200) are Miles's first-class targets. Every CI
pipeline runs on H-series; B-series uses identical flags and is validated on every
release. A100 works with FP8 features disabled.

## Recommended setup

```bash
docker pull radixark/miles:latest

docker run --rm \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  -it radixark/miles:latest /bin/bash
```

The image bundles:

| Component | Why pinned |
|---|---|
| CUDA 12.4+ | Required for FP8 GEMM via cuBLASLt |
| FlashAttention-3 (default), flashinfer (det) | Best-in-class attention kernels |
| DeepGEMM | Kernel for grouped GEMM (MoE) |
| NCCL 2.20+ | NVLink SHARP, IB-aware collectives |
| TransformerEngine | FP8 forward/backward |

## Per-GPU notes

### H100 / H200

* Default target — every flag in this site assumes Hopper.
* H200's 141 GB lets you fit larger models with smaller TP. Drop TP from 8 to 4 where
  possible.
* SHARP requires `NCCL_NVLS_ENABLE=1` (already on in our containers).

### B100 / B200

* Same flags as H-series. Native FP8 throughput is roughly 2× H100.
* Kernel pre-compile times can be longer first run — bump
  `--rollout-health-check-first-wait` to 600s+.

### A100

* No FP8 GEMM. Set `--no-fp8` in `MODEL_ARGS`; SGLang auto-detects and falls back to
  BF16.
* Tested as a target; not a CI target.

## Multi-node networking

* **InfiniBand HDR/NDR**: ~200/400 Gbps per port. Default in most H100 deployments.
* **RoCEv2**: works, configure `NCCL_IB_HCA` to your physical NICs.
* **Slingshot 11**: requires `NCCL_NET_PLUGIN=cassini`.

Confirm bandwidth with `ib_send_bw` between two ranks before launching a multi-day run.

## Common environment variables

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=COLL,P2P
NCCL_IB_HCA=mlx5_0,mlx5_1,...
NCCL_TIMEOUT=900
NVTE_FUSED_ATTN=1            # default, but verify
TORCHINDUCTOR_CACHE_DIR=/data/.inductor
```

## NVLink + IB topology

For 8× GPUs per node:

* All-to-all NVLink connectivity (`nvidia-smi topo -m` should show `NV4` between every
  pair).
* 4–8 IB NICs per node, one per GPU pair, configured via `NCCL_IB_HCA`.

If `nvidia-smi topo -m` shows `PIX` or `PHB` instead of `NV*`, you've lost a link —
fix before training.

## Quick health probe

Before submitting a job, sanity-check the node:

```bash
nvidia-smi                          # GPUs visible, driver / CUDA versions
nvidia-smi topo -m                  # NVLink mesh (NV* between every pair)
ibstat                              # IB ports up (multi-node only)
python -c "import flash_attn; import transformer_engine"   # kernels importable
```

If anything's wrong, fix it here — chasing it inside Ray is harder.
