import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class _GpuMemorySampler(threading.Thread):
    """Samples whole-GPU and per-pid physical memory. Physical (nvidia-smi /
    mem_get_info) is the only trustworthy view under torch_memory_saver: torch
    allocator stats still count paused (physically released) regions."""

    def __init__(self, interval_s: float = 2.0):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._interval_s = interval_s
        self._gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            free, total = torch.cuda.mem_get_info()
            self.samples.append(
                {
                    "gpu_used_gb": round((total - free) / 2**30, 2),
                    "torch_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                    "pid_mem_gb": self._per_pid_memory(),
                }
            )
            self._stop_event.wait(self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=self._interval_s + 15)

    def _per_pid_memory(self) -> dict[str, float]:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index}",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return {}
        ans = {}
        for line in out.strip().splitlines():
            pid, _, mem = line.partition(",")
            ans[pid.strip()] = round(int(mem) / 1024, 2)
        return ans


class _UpdateWeightsProfiler:
    """Phase time and GPU memory breakdown of update_weights, gated by
    MILES_PROFILE_UPDATE_WEIGHTS=1. Sections are fenced with cuda synchronize,
    which serializes otherwise-async phases - enable for measurement runs only."""

    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = os.environ.get("MILES_PROFILE_UPDATE_WEIGHTS", "0") not in ("0", "")
        self.enabled = enabled and (not dist.is_initialized() or dist.get_rank() == 0)
        self._sections: dict[str, float] = defaultdict(float)
        self._chunks: list[tuple[float, float, str]] = []
        self._chunk_alloc0 = 0
        self._sampler: _GpuMemorySampler | None = None
        if self.enabled:
            self._sampler = _GpuMemorySampler()
            self._sampler.start()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self._sections[name] += time.perf_counter() - start

    def chunk_begin(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            self._chunk_alloc0 = torch.cuda.memory_allocated()

    def record_chunk(self, label: str, seconds: float) -> None:
        if self.enabled:
            transient_gb = (torch.cuda.max_memory_allocated() - self._chunk_alloc0) / 2**30
            self._chunks.append((seconds, transient_gb, label))

    def report(self, total: float) -> None:
        if not self.enabled:
            return
        sections = " | ".join(
            f"{name} {sec:.1f}s" for name, sec in sorted(self._sections.items(), key=lambda kv: -kv[1])
        )
        logger.info(f"update_weights profile: total {total:.1f}s ({len(self._chunks)} chunks) | {sections}")
        top_time = ", ".join(f"{label} {sec:.2f}s" for sec, _, label in sorted(self._chunks, reverse=True)[:10])
        logger.info(f"update_weights slowest chunks: {top_time}")
        top_mem = ", ".join(
            f"{label} {gb:.2f}GB"
            for gb, label in sorted(((gb, label) for _, gb, label in self._chunks), reverse=True)[:10]
        )
        logger.info(f"update_weights chunk transient peaks: {top_mem}")
        self._report_memory()

    def _report_memory(self) -> None:
        sampler = self._sampler
        sampler.stop()
        samples = sampler.samples
        if not samples:
            return
        self_pid = str(os.getpid())
        peak = max(samples, key=lambda s: s["gpu_used_gb"])
        actor_series = [s["pid_mem_gb"].get(self_pid, 0.0) for s in samples]
        engine_series = [sum(m for pid, m in s["pid_mem_gb"].items() if pid != self_pid) for s in samples]
        logger.info(
            f"update_weights memory (physical GB): gpu_used start {samples[0]['gpu_used_gb']} "
            f"peak {peak['gpu_used_gb']} end {samples[-1]['gpu_used_gb']} | "
            f"actor pid start {actor_series[0]} peak {max(actor_series)} | "
            f"engine pids start {engine_series[0]} peak {max(engine_series)} | "
            f"torch_alloc peak {max(s['torch_alloc_gb'] for s in samples)}"
        )
        stride = max(1, len(samples) // 15)
        series = [
            (s["gpu_used_gb"], s["pid_mem_gb"].get(self_pid, 0.0), round(engine_series[i], 2))
            for i, s in enumerate(samples)
        ][::stride]
        logger.info(f"update_weights memory series (gpu_used, actor, engines) every {stride * 2}s: {series}")
        self._release_ladder()

    @staticmethod
    def _release_ladder() -> None:
        def used_gb():
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / 2**30, 2)

        def stats():
            s = torch.cuda.memory_stats()
            return {
                k: round(s[f"{k}.all.current"] / 2**30, 2)
                for k in ("reserved_bytes", "active_bytes", "inactive_split_bytes")
            }

        before, stats_before = used_gb(), stats()
        torch.cuda.ipc_collect()
        after_ipc = used_gb()
        torch.cuda.empty_cache()
        after_ec, stats_after = used_gb(), stats()
        logger.info(
            f"update_weights release ladder (gpu_used GB): {before} -> ipc_collect {after_ipc} "
            f"-> empty_cache {after_ec} | stats before {stats_before} after {stats_after}"
        )


_NOOP_PROFILER = _UpdateWeightsProfiler(enabled=False)

_ACTIVE_PROFILER = _NOOP_PROFILER


def set_active_profiler(profiler: "_UpdateWeightsProfiler | None") -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler if profiler is not None else _NOOP_PROFILER


def active_profiler() -> _UpdateWeightsProfiler:
    return _ACTIVE_PROFILER
