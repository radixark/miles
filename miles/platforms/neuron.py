"""AWS Trainium through torch-neuronx's native `neuron` device."""

import torch

from miles.platforms import Platform


def _get_memory_info(device=None) -> tuple[int, int]:
    # neuron has no mem_get_info; what the caching allocator has not reserved counts as free
    total = torch.neuron.get_device_properties(device).total_memory
    return total - torch.neuron.memory_reserved(device), total


class NeuronPlatform(Platform):
    device_type = "neuron"
    # neuron compiles each new shape into its own NEFF; dynamic-shape helpers stay eager
    dynamic_shapes = False
    nvml_numa_affinity = False
    # FlexAttention lowers through inductor/Triton, which neuron lacks
    flex_attention = False

    def __init__(self) -> None:
        # torch-neuronx's allocator is not a c10 DeviceAllocator, so torch.accelerator's memory
        # calls assert; route the ones miles makes to torch.neuron's own
        torch.accelerator.empty_cache = torch.neuron.empty_cache
        torch.accelerator.memory_allocated = torch.neuron.memory_allocated
        torch.accelerator.memory_reserved = torch.neuron.memory_reserved
        torch.accelerator.get_memory_info = _get_memory_info

    def bind_device_before_process_group(self, local_rank: int) -> None:
        # one process per logical NeuronCore among NEURON_RT_VISIBLE_CORES, as torchrun does it
        torch.neuron.set_device(local_rank)

    def device_name(self) -> str | None:
        return None
