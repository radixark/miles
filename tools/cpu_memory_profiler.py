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
# CLI usage:
#   python cpu_memory_profiler.py visualize profile.csv
#   python cpu_memory_profiler.py visualize profile.csv -o chart.png --title "My Run"

import csv
import logging
import threading
import time
import typer

import psutil

logger = logging.getLogger(__name__)
app = typer.Typer()
app.callback()(lambda: None)

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
                marker = ""
                while next_marker and next_marker[0] <= t:
                    marker = next_marker[1]
                    next_marker = next(mi, None)
                w.writerow([f"{t:.2f}", f"{used / 1e9:.3f}", f"{avail / 1e9:.3f}", f"{pct:.1f}", marker])
        logger.info(f"CPU memory profile saved to {self.output_path}")

    @property
    def peak_used_gb(self):
        if not self._records:
            return 0.0
        return max(r[1] for r in self._records) / 1e9


@app.command()
def visualize(csv_path: str, output_path: str = None, title: str = None, show: bool = False):
    """Visualize a CPU memory profile CSV as a timeline chart with phase markers."""
    import csv as csv_mod
    from pathlib import Path

    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    output_path = Path(output_path) if output_path else csv_path.with_suffix(".png")
    title = title or f"CPU Memory Profile — {csv_path.name}"

    # Parse CSV
    times, used, markers = [], [], []
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            t = float(row["time_s"])
            u = float(row["used_gb"])
            m = row["marker"].strip()
            times.append(t)
            used.append(u)
            if m:
                markers.append((t, u, m))

    # Phase colors
    phase_colors = {
        "generate": "#16a34a",
        "train": "#dc2626",
        "offload_train": "#f59e0b",
        "offload_rollout": "#8b5cf6",
        "onload_weights": "#0891b2",
        "onload_kv": "#06b6d4",
        "update_weights": "#ec4899",
    }

    def get_color(label):
        phase = label.split("/", 1)[-1] if "/" in label else label
        base = phase.rsplit("_start", 1)[0].rsplit("_end", 1)[0]
        return phase_colors.get(base, "#6b7280")

    # Plot
    fig, ax = plt.subplots(figsize=(max(16, len(times) * 0.02), 7))
    ax.plot(times, used, linewidth=1.0, color="#2563eb", zorder=2)
    ax.fill_between(times, used, alpha=0.15, color="#2563eb", zorder=1)

    for t, u, label in markers:
        color = get_color(label)
        ax.axvline(x=t, color=color, linestyle="--", alpha=0.5, linewidth=0.8, zorder=3)
        ax.plot(t, u, "o", color=color, markersize=5, zorder=4)

    if markers:
        y_min, y_max = min(used), max(used)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.45)
        n_levels = 4
        for i, (t, u, label) in enumerate(markers):
            color = get_color(label)
            level = i % n_levels
            y_pos = y_max + y_range * (0.08 + level * 0.09)
            ax.annotate(
                label,
                xy=(t, u),
                xytext=(t + (times[-1] - times[0]) * 0.005, y_pos),
                fontsize=7,
                color=color,
                fontweight="bold",
                rotation=45,
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, linewidth=0.6),
            )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("CPU Memory Used (GB)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    peak_idx = used.index(max(used))
    ax.annotate(
        f"Peak: {used[peak_idx]:.1f} GB",
        xy=(times[peak_idx], used[peak_idx]),
        xytext=(times[peak_idx] + (times[-1] - times[0]) * 0.03, used[peak_idx]),
        fontsize=9,
        fontweight="bold",
        color="#dc2626",
        arrowprops=dict(arrowstyle="->", color="#dc2626"),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"CPU memory profile chart saved to {output_path}")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    app()
