"""Intel GPUs through torch's native xpu device."""

import functools

import torch

from miles.platforms import Platform


class XpuPlatform(Platform):
    device_type = "xpu"
    # miles.kernels' Triton kernels are written for the CUDA Triton backend and are not validated on the XPU one
    triton_kernels = False
    # the modes with no CUDA call on their path: p2p stages its replica on a CUDA device, disk-delta's sender waits on
    # a CUDA stream, and --colocate hands the engine CUDA IPC handles
    weight_transfer_modes = frozenset({"broadcast", "http-lora"})
    # NVML sees no Intel GPU, so the trainer skips NUMA pinning
    nvml_numa_affinity = False

    def bind_device_before_process_group(self, local_rank: int) -> None:
        torch.xpu.set_device(local_rank)

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        return torch.xpu.get_device_name() if torch.xpu.is_available() else None
