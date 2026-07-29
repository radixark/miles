---
title: NVIDIA H / B Series
description: H100, H200, B100, B200 — Miles's primary target.
---
NVIDIA Blackwell (GB300 / GB200 / B200 / B100) and Hopper (H200 / H100) are Miles's first-class targets.

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
| CUDA 12.4+ | Required for FP8 GEMM via cuBLAS |
| FlashAttention-3 (default), Flashinfer| Best-in-class attention kernels |
| DeepGEMM | Kernel for grouped GEMM (MoE) |
| NCCL 2.20+ | NVLink SHARP, IB-aware collectives |
| TransformerEngine | FP8 forward/backward |

## Per-GPU notes

### H100 / H200

* The default target the recipes on this site are tuned against.
* H200 ships with 141 GB HBM (vs. 80 GB on H100), so you can often reduce TP for the
  same model — e.g. TP 8 → TP 4 on a single 8-GPU node.

### B100 / B200

* Same launch flags as H-series; FP8 GEMM uses the same code path.
* First-run kernel compilation can take longer than on H-series. If the rollout engine
  is flagged unhealthy during warm-up, raise `--rollout-health-check-first-wait`
  (e.g. 600s).

### A100

* No FP8 GEMM — the BF16 path is used automatically.
* Supported, but not part of CI; expect rougher edges than on H/B-series.

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

miles sets `NVSHMEM_DISABLE_NCCL=1` on rollout and training actors automatically. DeepEP
initializes NVSHMEM, whose built-in NCCL path opens a second NCCL communicator that collides
with miles' own NCCL and hangs during SGLang CUDA-graph replay. Export `NVSHMEM_DISABLE_NCCL=0`
to opt out.

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
```

If anything's wrong, fix it here — chasing it inside Ray is harder.
