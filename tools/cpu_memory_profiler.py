# CPU Memory Profiler - standalone tool for profiling system memory during training.
#
# Profiler usage (e.g. in train.py):
#   from tools.cpu_memory_profiler import CPUMemoryProfiler
#
#   # beginning of training loop
#   profiler = CPUMemoryProfiler(interval=0.5, output_path="cpu_memory_profile.csv")
#   profiler.start()
#
#   # at somewhere you want to mark with a label
#   profiler.mark("step_0/generate_start")
#
#   # end of training loop
#   profiler.stop()
#
import csv
import logging
import threading
import time

import psutil

logger = logging.getLogger(__name__)


class CPUMemoryProfiler:
    """System-level CPU memory profiler for Ray multi-process training.

    Samples total machine memory usage at regular intervals and supports
    phase markers to align with training stages (generate, offload, train, etc.).
    Uses psutil.virtual_memory() which naturally handles shared memory deduplication
    across Ray workers.
    """

    def __init__(self, interval=0.5, output_path="cpu_memory_profile.csv"):
        self.interval = interval
        self.output_path = output_path
        self._running = False
        self._records = []  # [(elapsed_s, used_bytes, available_bytes, percent)]
        self._markers = []  # [(elapsed_s, label)]
        self._lock = threading.Lock()
        self._t0 = None

    def start(self):
        self._t0 = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"CPU memory profiler started (interval={self.interval}s, output={self.output_path})")

    def _sample_loop(self):
        while self._running:
            vm = psutil.virtual_memory()
            with self._lock:
                self._records.append(
                    (
                        time.time() - self._t0,
                        vm.used,
                        vm.available,
                        vm.percent,
                    )
                )
            time.sleep(self.interval)

    def mark(self, label: str):
        """Place a named marker on the timeline for phase alignment. No-op if not started."""
        if not self._running:
            return
        with self._lock:
            elapsed = time.time() - self._t0
            self._markers.append((elapsed, label))
        vm = psutil.virtual_memory()
        logger.info(f"[CPU profiler] {label} @ {elapsed:.1f}s, used={vm.used / 1e9:.2f}GB ({vm.percent}%)")

    def stop(self):
        self._running = False
        self._thread.join()
        self._dump()
        logger.info(
            f"CPU memory profiler stopped. Peak used: {self.peak_used_gb:.2f}GB. " f"Saved to {self.output_path}"
        )

    def _dump(self):
        markers_sorted = sorted(self._markers)
        with open(self.output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "used_gb", "available_gb", "percent", "marker"])
            mi = iter(markers_sorted)
            next_marker = next(mi, None)
            for t, used, avail, pct in self._records:
                markers_in_interval = []
                while next_marker and next_marker[0] <= t:
                    markers_in_interval.append(next_marker[1])
                    next_marker = next(mi, None)
                w.writerow(
                    [
                        f"{t:.2f}",
                        f"{used / 1e9:.3f}",
                        f"{avail / 1e9:.3f}",
                        f"{pct:.1f}",
                        ";".join(markers_in_interval),
                    ]
                )
        logger.info(f"CPU memory profile saved to {self.output_path}")

    @property
    def peak_used_gb(self):
        if not self._records:
            return 0.0
        return max(r[1] for r in self._records) / 1e9
