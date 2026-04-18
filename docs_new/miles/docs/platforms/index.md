---
title: Platforms
description: Hardware-specific tutorials. Most users want NVIDIA H/B; AMD MI300X is supported via ROCm.
---

# Platforms

Miles is hardware-agnostic in design but the realities of CUDA, ROCm, and FP8 mean each
platform has its own setup notes.

<div class="grid cards" markdown>

-   :material-chip:{ .lg .middle } **[NVIDIA H / B Series](nvidia.md)**

    ---
    The default. H100 / H200 / B100 / B200 with FP8, NVLink, and InfiniBand.

-   :material-chip:{ .lg .middle } **[AMD MI300X](amd.md)**

    ---
    ROCm 6.3+ with patches for virtual memory management. Same launch scripts.

</div>

## Supported features by GPU

| Feature | H100 / H200 / B-series | A100 | MI300X |
|---|---|---|---|
| BF16 training | ✅ | ✅ | ✅ |
| FP8 GEMM | ✅ Native | ❌ | ✅ Forward only |
| INT4 W4A16 QAT | ✅ | ⚠️ Slow | ⚠️ |
| Speculative decoding | ✅ | ✅ | ✅ |
| Miles Router (R3) | ✅ | ✅ | ✅ |
| P2P weight transfer (RDMA) | ✅ IB / RoCEv2 | ✅ | ✅ Infinity Fabric |
| Megatron CP | ✅ | ✅ | ⚠️ Some limitations |
| Deterministic inference | ✅ | ✅ | ⚠️ |

## Storage and network

Independent of GPU vendor, you'll want:

* **Shared filesystem** for multi-node — NFS, GPFS, Lustre. Reads dominate writes during
  training; provision read bandwidth.
* **High-bandwidth interconnect** — IB (NVIDIA), RoCEv2 (NVIDIA), Slingshot, or
  Infinity Fabric (AMD). 200+ GB/s per node is typical for trillion-param training.
* **NVMe local scratch** for SGLang radix cache and Ray spill — at least 1 TB.
