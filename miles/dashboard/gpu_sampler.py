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
from miles.dashboard.store import GpuSample

logger = logging.getLogger(__name__)


class GpuSampler:
    FLUSH_INTERVAL_SECONDS: ClassVar[float] = 5.0

    def __init__(
        self,
        push: Callable[[str, list[GpuSample]], None],
        *,
        node: str,
        interval: float = 1.0,
        nvml=None,
    ):
        assert interval > 0, f"{interval=}"
        self._push = push
        self.node = node
        self.interval = interval
        self._nvml = nvml
        self._handles: list = []
        self._uuids: list[str] = []
        self._buffer: list[GpuSample] = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)
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
        while not self._stop_event.is_set():
            self.sample_once(time.time())
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
        return count

    def flush(self) -> None:
        with self._buffer_lock:
            batch, self._buffer = self._buffer, []
        if batch:
            self._push(self.node, batch)
