import contextlib
import gc
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_EXPANDABLE_DURING_TRAIN = os.environ.get("MILES_EXPANDABLE_SEGMENTS_DURING_TRAIN", "0") == "1"


def _set_expandable_segments(enabled: bool) -> None:
    setting = f"expandable_segments:{'True' if enabled else 'False'}"
    try:
        torch._C._accelerator_setAllocatorSettings(setting)
    except AttributeError:
        torch.cuda.memory._set_allocator_settings(setting)


@contextlib.contextmanager
def expandable_segments_during_training():
    if not _EXPANDABLE_DURING_TRAIN:
        yield
        return
    _set_expandable_segments(True)
    try:
        yield
    finally:
        clear_memory()
        _set_expandable_segments(False)


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
