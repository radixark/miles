import gc
import logging

import psutil
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_HOST_MEMINFO_KEYS = ("Mlocked", "Unevictable", "AnonPages", "Cached")

_cpu_memory_profiler = None


def start_cpu_memory_profiler(output_path: str, interval: float = 0.5):
    """Start the node-level CPU memory sampler (tools/cpu_memory_profiler.py) and
    dump its CSV on interpreter exit, so the timeline survives a crash."""
    global _cpu_memory_profiler
    assert _cpu_memory_profiler is None
    import atexit

    from tools.cpu_memory_profiler import CPUMemoryProfiler

    _cpu_memory_profiler = CPUMemoryProfiler(interval=interval, output_path=output_path)
    _cpu_memory_profiler.start()
    atexit.register(_cpu_memory_profiler.stop)


def host_memory():
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    info = {
        "proc_rss_GB": _byte_to_gb(rss),
        "node_used_GB": _byte_to_gb(vm.used),
        "node_avail_GB": _byte_to_gb(vm.available),
    }
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            if key in _HOST_MEMINFO_KEYS:
                info[f"{key}_GB"] = round(int(value.strip().split()[0]) / (1024**2), 2)
    return info


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
        f" host: {host_memory()}"
    )
    if _cpu_memory_profiler is not None:
        _cpu_memory_profiler.mark(f"rank{dist.get_rank()}/{msg}")
    return memory_info
