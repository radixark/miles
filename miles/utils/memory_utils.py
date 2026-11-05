import ctypes
import gc
import logging
import subprocess
import sys

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    """Memory of the current device. ``total``, ``free`` and ``used`` come from
    ``get_memory_info`` and cover every process on the GPU; ``allocated``,
    ``reserved``, ``inactive_split`` (reserved bytes the allocator holds in
    fragments it cannot hand out) and ``segments`` are this process's caching
    allocator; ``max_allocated`` and ``max_reserved`` are its peaks since the
    last ``reset_peak_memory_stats`` (the trainer resets them at every wake-up,
    so the line before an offload carries that step's peak); ``host`` is this
    process's host memory (``host_memory``)."""
    device = torch.accelerator.current_device_index()
    free, total = torch.accelerator.get_memory_info(device)
    stats = torch.accelerator.memory_stats(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.accelerator.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.accelerator.memory_reserved(device)),
        "max_allocated_GB": _byte_to_gb(torch.accelerator.max_memory_allocated(device)),
        "max_reserved_GB": _byte_to_gb(torch.accelerator.max_memory_reserved(device)),
        "inactive_split_GB": _byte_to_gb(stats.get("inactive_split_bytes.all.current", 0)),
        "segments": stats.get("segment.all.current", 0),
        "alloc_retries": stats.get("num_alloc_retries", 0),
        "host": host_memory(),
    }


NVIDIA_SMI_TIMEOUT_S = 15


def gpu_processes(device: int | None = None, query=None) -> list[dict]:
    """Every process with memory on the current device, from ``nvidia-smi``
    (``pid``, ``used_MiB``, ``name``): the only way to tell this process's
    share of ``used_GB`` from a colocated engine's. ``query`` is the
    ``nvidia-smi`` runner, replaced in tests. Returns ``[]`` with a logged
    warning when the tool is missing, times out or answers in another shape."""
    if query is None and not torch.cuda.is_available():
        # nvidia-smi lists CUDA devices only
        return []
    if device is None:
        device = torch.cuda.current_device()
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    args = [
        "nvidia-smi",
        f"--id=GPU-{uuid}",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        text = (query or _run_nvidia_smi)(args)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"gpu_processes: {e}")
        return []
    return parse_compute_apps(text)


def _run_nvidia_smi(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=NVIDIA_SMI_TIMEOUT_S).stdout


def parse_compute_apps(text: str) -> list[dict]:
    """``pid, used_memory, process_name`` csv rows (nounits: MiB) to records;
    a row that does not parse is skipped."""
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "used_MiB": int(float(parts[1])), "name": parts[2]})
        except ValueError:
            continue
    return rows


_STATUS_FIELDS = {
    "VmRSS": "rss_GB",
    "VmHWM": "rss_peak_GB",
    "RssAnon": "anon_GB",
    "RssShmem": "shmem_GB",
    "RssFile": "file_GB",
}


def host_memory(status_text: str | None = None) -> dict:
    """This process's host memory: resident set and its peak, the anonymous
    heap, the shared part (pinned CUDA host memory is counted here) and the
    file-backed part, from ``/proc/self/status``; plus the pinned-memory
    allocator's current and reserved bytes where torch reports them. A
    colocated trainer keeps pinned offload state and a heap that peaks at
    the rollout shard it deserialised; the two move for different reasons."""
    if status_text is None:
        try:
            status_text = open("/proc/self/status").read()
        except OSError:
            status_text = ""
    out = {}
    for line in status_text.splitlines():
        key, _, rest = line.partition(":")
        if key in _STATUS_FIELDS:
            out[_STATUS_FIELDS[key]] = round(int(rest.split()[0]) * 1024 / (1024**3), 2)
    stats = getattr(
        torch.cuda, "host_memory_stats", None
    )  # config-access-exempt: torch API, absent in older torch builds
    if stats is not None and torch.cuda.is_available():
        try:
            host = stats()
            out["pinned_allocated_GB"] = _byte_to_gb(host.get("allocated_bytes.all.current", 0))
            out["pinned_reserved_GB"] = _byte_to_gb(host.get("reserved_bytes.all.current", 0))
        except Exception as e:  # noqa: BLE001
            out["pinned_error"] = str(e)
    return out


def reset_peak_memory() -> None:
    """Start a new peak window for ``available_memory``'s ``max_*`` fields.
    Does nothing before the device runtime is initialized: there is no peak
    to reset, and CPU-only tests run with a stub driver that must not be
    touched."""
    if torch.cuda.is_initialized():
        torch.accelerator.reset_peak_memory_stats(torch.accelerator.current_device_index())


def trim_host_heap() -> bool:
    """Return the heap's free pages to the OS (glibc ``malloc_trim``): a
    process that deserialised a large rollout shard keeps the whole peak as
    resident memory otherwise, and the peak grows with every step whose data
    is bigger than any before. True when the call was made."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


def release_host_memory() -> None:
    """After a train step: drop garbage, give the pinned-memory cache back to
    CUDA (the replay queues pin buffers of a new size every step) and trim the
    heap, so the host footprint between steps stays at the state the trainer
    keeps, not at the largest step so far."""
    gc.collect()
    if hasattr(torch._C, "_host_emptyCache"):  # config-access-exempt: private torch API, absent in some builds
        torch._C._host_emptyCache()
    trim_host_heap()


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False, processes: bool = False):
    """Log the current device's memory; with ``processes`` also every process
    on it (one ``nvidia-smi`` call, for the moments a colocated engine shares
    the GPU: after the trainer releases its cache, ``used_GB`` is the sum of
    both and only the per-process list says which one grew)."""
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    if processes:
        memory_info["processes"] = gpu_processes()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
