---
title: Platforms
description: Hardware-specific tutorials. Most users want NVIDIA H/B; AMD MI300X is supported via ROCm.
---

# Platforms

Miles runs on NVIDIA H/B-series and AMD MI300X with the same launch scripts. Each platform page covers driver versions, build flags, and the FP8 / ROCm quirks you need to know before kicking off a job.

<CardGroup cols={2}>

  <Card title="NVIDIA GPUs" icon="microchip" href="nvidia">

    The default GB300 / GB200 / B200 / B100 / H200 / H100 with FP8, NVLink, and InfiniBand.

  </Card>

  <Card title="AMD GPUs" icon="microchip" href="amd">

    ROCm 6.3+ with patches for virtual memory management. Same launch scripts.

  </Card>

</CardGroup>

## Supported features by GPU

| Feature | H / B-series | A100 | MI300X |
|---|---|---|---|
| BF16 training | ✅ | ✅ | ✅ |
| FP8 GEMM | ✅  | - | ✅  |
| INT4 W4A16 QAT | ✅ | ✅ | ⚠️ |
| Speculative decoding | ✅ | ✅ | ✅ |
| Miles Router (R3) | ✅ | ✅ | ✅ |
| P2P weight transfer (RDMA) | ✅ IB / RoCEv2 | ✅ | ✅ Infinity Fabric |
| Megatron CP | ✅ | ✅ | ✅ |
| Deterministic inference | ✅ | ✅ | ⚠️ |

## Storage and network

Independent of GPU vendor, you'll want:

* **Shared filesystem** for multi-node — NFS, GPFS, Lustre. Reads dominate writes during
  training; provision read bandwidth.
* **High-bandwidth interconnect** — IB (NVIDIA), RoCEv2 (NVIDIA), Slingshot, or
  Infinity Fabric (AMD). 200+ GB/s per node is typical for trillion-param training.
* **NVMe local scratch** for SGLang radix cache and Ray spill — at least 1 TB.
