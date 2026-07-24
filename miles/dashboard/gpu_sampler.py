"""Per-node GPU utilization sampler feeding the dashboard timeline.

One instance runs per GPU node (the collector spawns it as a Ray actor with
``NodeAffinitySchedulingStrategy``; the class itself is plain Python and
unit-testable). A daemon thread samples every NVML device on the node at
``interval`` seconds — physical device order, independent of
``CUDA_VISIBLE_DEVICES`` — buffers locally, and hands batches to the injected
``push(node, batch)`` callable (the collector wraps its own Ray handle) every
``FLUSH_INTERVAL_SECONDS``, so there is roughly one RPC per node per flush
rather than one per sample.

Degradation: NVML being unavailable (no pynvml, driver mismatch) disables the
sampler with a single warning — the timeline just lacks the util band. A
device that fails mid-run (e.g. during a GPU reset) is skipped for that tick
with rate-limited warnings; the other devices keep reporting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import ClassVar

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import CpuMemorySample, GpuProcessSample, GpuSample

logger = logging.getLogger(__name__)


class GpuSampler:
    FLUSH_INTERVAL_SECONDS: ClassVar[float] = 5.0
    # per-process memory breakdown is a coarser, heavier NVML call (enumerates
    # every process) than the plain util/mem read, so it samples on its own,
    # slower cadence rather than every `interval` tick
    PROCESS_SAMPLE_INTERVAL_SECONDS: ClassVar[float] = 5.0

    def __init__(
        self,
        push: Callable[[str, list[GpuSample]], None],
        *,
        node: str,
        interval: float = 1.0,
        nvml=None,
        push_processes: Callable[[str, list[GpuProcessSample]], None] | None = None,
        cpu_push: Callable[[str, list[CpuMemorySample]], None] | None = None,
        psutil_module=None,
    ):
        assert interval > 0, f"{interval=}"
        self._push = push
        self._push_processes = push_processes
        self.node = node
        self.interval = interval
        self._nvml = nvml
        self._cpu_push = cpu_push
        self._psutil = psutil_module
        self._handles: list = []
        self._uuids: list[str] = []
        self._buffer: list[GpuSample] = []
        self._process_buffer: list[GpuProcessSample] = []
        self._cpu_buffer: list[CpuMemorySample] = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)
        if self._cpu_push is not None and self._psutil is None:
            import psutil

            self._psutil = psutil
        self.available = self._init_nvml()

    def _init_nvml(self) -> bool:
        try:
            if self._nvml is None:
                import pynvml

                self._nvml = pynvml
            self._nvml.nvmlInit()
            count = self._nvml.nvmlDeviceGetCount()
            self._handles = [self._nvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._uuids = [str(self._nvml.nvmlDeviceGetUUID(handle)) for handle in self._handles]
            return True
        except Exception as e:
            logger.warning("NVML unavailable on %s (%s); GPU utilization will not be collected", self.node, e)
            return False

    # ------------------------------ lifecycle -------------------------------

    def gpu_uuids(self) -> list[str]:
        return list(self._uuids)

    def start(self) -> bool:
        """Begin sampling; returns False (and stays inert) when NVML is unavailable."""
        if not self.available:
            return False
        assert self._thread is None, "sampler already started"
        self._thread = threading.Thread(target=self._run, name="dashboard-gpu-sampler", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + self.FLUSH_INTERVAL_SECONDS)
        self.flush()

    def _run(self) -> None:
        next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
        next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
        while not self._stop_event.is_set():
            self.sample_once(time.time())
            if self._push_processes is not None and time.monotonic() >= next_process_sample:
                self.sample_processes_once(time.time())
                next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
            if time.monotonic() >= next_flush:
                self.flush()
                next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
            self._stop_event.wait(self.interval)

    # -------------------------------- sampling ------------------------------

    def sample_once(self, ts: float) -> int:
        """Sample every device once into the buffer. Returns the sample count."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                util = int(self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                mem_mb = int(self._nvml.nvmlDeviceGetMemoryInfo(handle).used) >> 20
                power_w = int(self._nvml.nvmlDeviceGetPowerUsage(handle)) // 1000
            except Exception:
                self._warner.warn(f"NVML read failed for gpu {gpu} on {self.node}; skipping this tick")
                continue
            with self._buffer_lock:
                self._buffer.append(
                    GpuSample(ts=ts, node=self.node, gpu=gpu, util=util, mem_mb=mem_mb, power_w=power_w)
                )
            count += 1
        self._sample_cpu_memory(ts)
        return count

    def sample_processes_once(self, ts: float) -> int:
        """Per-process VRAM breakdown once per GPU: who is actually holding
        the memory, not just the per-GPU aggregate ``sample_once`` reports."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                self._warner.warn(f"NVML process query failed for gpu {gpu} on {self.node}; skipping this tick")
                continue
            for proc in procs:
                mem_mb = (proc.usedGpuMemory or 0) >> 20
                with self._buffer_lock:
                    self._process_buffer.append(
                        GpuProcessSample(
                            ts=ts,
                            node=self.node,
                            gpu=gpu,
                            pid=proc.pid,
                            name=self._process_name(proc.pid),
                            mem_mb=mem_mb,
                        )
                    )
                count += 1
        return count

    def _process_name(self, pid: int) -> str:
        try:
            name = self._nvml.nvmlSystemGetProcessName(pid)
            return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            return f"pid {pid}"  # process exited between enumeration and lookup, or name unavailable

    def _sample_cpu_memory(self, ts: float) -> None:
        if self._cpu_push is None:
            return
        try:
            memory = self._psutil.virtual_memory()
            total = int(memory.total)
            available = int(memory.available)
            used = max(0, total - available)
            percent = 100.0 * used / total if total else 0.0
        except Exception:
            self._warner.warn(f"CPU memory read failed on {self.node}; skipping this tick")
            return
        with self._buffer_lock:
            self._cpu_buffer.append(
                CpuMemorySample(
                    ts=ts,
                    node=self.node,
                    used_bytes=used,
                    available_bytes=available,
                    total_bytes=total,
                    percent=percent,
                )
            )

    def flush(self) -> None:
        with self._buffer_lock:
            batch, self._buffer = self._buffer, []
            process_batch, self._process_buffer = self._process_buffer, []
            cpu_batch, self._cpu_buffer = self._cpu_buffer, []
        if batch:
            self._push(self.node, batch)
        if process_batch and self._push_processes is not None:
            self._push_processes(self.node, process_batch)
        if cpu_batch and self._cpu_push is not None:
            self._cpu_push(self.node, cpu_batch)
