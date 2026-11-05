"""TPU v7x through torch_tpu."""

import torch
import torch_tpu  # noqa: F401  registers the tpu device with torch

from miles.platforms import Platform


class TpuPlatform(Platform):
    device_type = "tpu"
    # the tpu compile backend takes static shapes only
    dynamic_shapes = False
    nvml_numa_affinity = False
    # sglang-torchtpu turns "auto" into an fp8 KV cache without scales, which skews rollout logprobs against the
    # bf16 trainer; fp8 stays available through --sglang-kv-cache-dtype fp8_e4m3
    rollout_kv_cache_dtype = "bfloat16"

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # touching the device before init_process_group breaks the multi-process rendezvous
        pass

    def bind_device_after_process_group(self) -> None:
        # each process sees its one device, which the process group has brought up
        torch.accelerator.set_device_index(0)

    def device_name(self) -> str | None:
        return None
