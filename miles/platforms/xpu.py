"""Intel GPUs through torch's native xpu device."""

import functools

import torch

from miles.platforms import Platform


class XpuPlatform(Platform):
    device_type = "xpu"
    # NVML sees no Intel GPU, so the trainer skips NUMA pinning
    nvml_numa_affinity = False

    def bind_device_before_process_group(self, local_rank: int) -> None:
        torch.xpu.set_device(local_rank)

    @functools.cache  # noqa: B019 -- one instance per process
    def device_name(self) -> str | None:
        return torch.xpu.get_device_name() if torch.xpu.is_available() else None
