# CPU Memory Profiler Visualization
#
# CLI usage:
#   python -m tools.visualize.cpu_memory_profiler_visualize visualize profile.csv
#   python -m tools.visualize.cpu_memory_profiler_visualize visualize profile.csv -o chart.png --title "My Run"
#   python -m tools.visualize.cpu_memory_profiler_visualize visualize-multiple a.csv b.csv -o compare.png

import logging

import typer

logger = logging.getLogger(__name__)
app = typer.Typer()
app.callback()(lambda: None)


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


@app.command()
def visualize_multiple(
    csv_paths: list[str],
    output_path: str = typer.Option(None, "-o", "--output-path"),
    title: str = typer.Option(None, "--title"),
    show: bool = typer.Option(False, "--show"),
):
    """Overlay multiple CPU memory profiles on one chart with different colors."""
    import csv as csv_mod
    from pathlib import Path

    import matplotlib.pyplot as plt

    line_colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6", "#ec4899", "#0891b2"]

    traces = []
    for p in csv_paths:
        times, used = [], []
        with open(p) as f:
            for row in csv_mod.DictReader(f):
                times.append(float(row["time_s"]))
                used.append(float(row["used_gb"]))
        traces.append((times, used))

    out = Path(output_path) if output_path else Path(csv_paths[0]).with_name("compare.png")
    title = title or "CPU Memory Profile Comparison"

    fig, ax = plt.subplots(figsize=(max(16, max(len(t) for t, _ in traces) * 0.02), 7))
    for i, (times, used) in enumerate(traces):
        color = line_colors[i % len(line_colors)]
        label = Path(csv_paths[i]).stem
        ax.plot(times, used, linewidth=1.0, color=color, zorder=2, label=label)
        ax.fill_between(times, used, alpha=0.10, color=color, zorder=1)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("CPU Memory Used (GB)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison chart saved to {out}")

    if show:
        plt.show()
    plt.close(fig)

    return out


if __name__ == "__main__":
    app()
