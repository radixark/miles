"""Host offload for torch-native backends: the modules and optimizers move together."""

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, move_optimizer_state, print_memory


def offload_to_host(modules, optimizers) -> None:
    print_memory("before offload model")
    for module in modules:
        module.cpu()
    move_optimizer_state(optimizers, "cpu")
    clear_memory()
    dist.barrier(group=get_gloo_group())
    print_memory("after offload model")


def reload_to_device(modules, optimizers) -> None:
    for module in modules:
        module.cuda()
    move_optimizer_state(optimizers, "cuda")
    dist.barrier(group=get_gloo_group())
    print_memory("after wake_up model")
