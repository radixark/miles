import gc
import logging
from collections.abc import Sequence

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_quantized_weight_workspaces(models: Sequence[torch.nn.Module]) -> int:
    """Drop TransformerEngine's cached quantized weights so offload does not back them up.

    Under a low-precision recipe TE keeps the quantized weight of every module in
    ``_fp8_workspaces`` for both mxfp8, nvfp4. It is derived from the high-precision weight and TE rebuilds
    it on the next forward via the cache-miss path.
    """
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    for model_chunk in models:
        assert model_chunk.config.cuda_graph_impl == "none", (
            "clear_quantized_weight_workspaces is unsafe with CUDA graphs "
            f"(cuda_graph_impl={model_chunk.config.cuda_graph_impl})"
        )

    num_cleared = 0
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, TransformerEngineBaseModule):
                num_cleared += len(module._fp8_workspaces)
                module._fp8_workspaces.clear()
    return num_cleared


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
