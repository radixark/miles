import ctypes
import ctypes.util
import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_libc = ctypes.CDLL(ctypes.util.find_library("c"))


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()
        # Checkpoint/weight loading churns through millions of MB-scale CPU
        # tensors that glibc serves from its arenas (below the 32MB dynamic
        # mmap threshold); freed arena pages are retained, so RSS stays at the
        # loading high-water mark unless explicitly trimmed.
        _libc.malloc_trim(0)


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
