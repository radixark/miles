"""Timing for the multi-LoRA gateway: where the trainer's time goes per op (OpProfiler), plus GPU peaks, as tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

__all__ = [
    "METRICS_LOG_PREFIX",
    "PROFILE_LOG_PREFIX",
    "OpProfiler",
    "Stats",
    "gpu_peaks",
    "head_tail_means",
    "merge_series",
    "parse_cookbook_rewards",
    "parse_serve_log",
    "render_gpu",
    "render_metrics",
    "render_profile",
    "render_table",
]

PROFILE_LOG_PREFIX = "multi-LoRA profile: "
METRICS_LOG_PREFIX = "multi-LoRA metrics: "
_COOKBOOK_STEP = re.compile(r"^\s*Step (\d+)\s*$")
_COOKBOOK_REWARD = re.compile(r"reward/total\s*│\s*([-0-9.eE+]+)")
_CAPACITY_LINE = re.compile(r"multi-LoRA capacity: (\d+) slots, bound by (.+?) \[")
_LOADED_LORAS_LINE = re.compile(r"engines keep at most (\d+) adapter versions loaded")
_TRAINER_NODE_LINE = re.compile(r"MegatronTrainRayActor pid=\d+, ip=([0-9.]+)")
_ENGINE_NODE_LINE = re.compile(r"CommandActor pid=\d+.*Uvicorn running on http://([0-9.]+):")


@dataclass(frozen=True)
class Stats:
    """Order statistics of one set of durations, in seconds."""

    n: int
    total: float
    mean: float
    p50: float
    p90: float
    p95: float
    min: float
    max: float

    @classmethod
    def of(cls, values: Iterable[float]) -> Stats:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            raise ValueError("Stats.of needs at least one value")

        def quantile(p: float) -> float:
            return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

        return cls(
            n=len(ordered),
            total=sum(ordered),
            mean=statistics.fmean(ordered),
            p50=quantile(0.5),
            p90=quantile(0.9),
            p95=quantile(0.95),
            min=ordered[0],
            max=ordered[-1],
        )

    def describe(self) -> str:
        return f"mean {self.mean:.1f} s, p50 {self.p50:.1f}, p90 {self.p90:.1f}, max {self.max:.1f} (n={self.n})"


class OpProfiler:
    """Wall time per gateway op; ``size`` is what one call covered (datums, slots, samples)."""

    def __init__(self) -> None:
        self._seconds: dict[str, list[float]] = {}
        self._sizes: dict[str, int] = {}
        self._series: dict[str, dict[str, list[float]]] = {}

    def record(self, op: str, seconds: float, size: int = 1) -> None:
        self._seconds.setdefault(op, []).append(seconds)
        self._sizes[op] = self._sizes.get(op, 0) + size

    @asynccontextmanager
    async def timed(self, op: str, size: int = 1):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(op, time.perf_counter() - start, size)

    def snapshot(self) -> dict[str, dict[str, float]]:
        total_all = sum(sum(values) for values in self._seconds.values())
        snapshot = {}
        for op, values in self._seconds.items():
            stats = Stats.of(values)
            snapshot[op] = {
                "calls": stats.n,
                "size": self._sizes[op],
                "total_s": stats.total,
                "mean_s": stats.mean,
                "p50_s": stats.p50,
                "p90_s": stats.p90,
                "max_s": stats.max,
                "share": stats.total / total_all if total_all else 0.0,
            }
        return snapshot

    def observe(self, slot: int, **metrics: float) -> None:
        """One training step's metrics for a slot; call order is step order."""
        for name, value in metrics.items():
            self._series.setdefault(str(slot), {}).setdefault(name, []).append(float(value))

    def series(self) -> dict[str, dict[str, list[float]]]:
        return self._series

    def report(self) -> str:
        return render_profile(self.snapshot())

    def reset(self) -> None:
        self._seconds.clear()
        self._sizes.clear()
        self._series.clear()


def render_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    """A plain-text grid: left-aligned columns sized to their content."""
    headers = [str(header) for header in headers]
    body = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in body)]) for i, header in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in body)])


def head_tail_means(series: dict[str, dict[str, list[float]]], fraction: float = 0.1) -> dict[str, dict[str, float]]:
    """Per metric, the mean over each tenant's first and last ``fraction`` of steps (at least one step each)."""
    table: dict[str, dict[str, list[float]]] = {}
    for per_metric in series.values():
        for metric, values in per_metric.items():
            if not values:
                continue
            k = max(1, math.ceil(fraction * len(values)))
            entry = table.setdefault(metric, {"head": [], "tail": [], "steps": [], "k": []})
            entry["head"].append(statistics.fmean(values[:k]))
            entry["tail"].append(statistics.fmean(values[-k:]))
            entry["steps"].append(len(values))
            entry["k"].append(k)
    return {
        metric: {
            "head": statistics.fmean(e["head"]),
            "tail": statistics.fmean(e["tail"]),
            "tenants": len(e["head"]),
            "steps": statistics.fmean(e["steps"]),
            "k": statistics.fmean(e["k"]),
        }
        for metric, e in table.items()
    }


def render_metrics(series: dict[str, dict[str, list[float]]], fraction: float = 0.1) -> str:
    means = head_tail_means(series, fraction)
    order = ["reward", "loss", "log_prob", "mean_len"]
    rows = [
        [
            metric,
            f"{means[metric]['head']:.4g}",
            f"{means[metric]['tail']:.4g}",
            f"{means[metric]['k']:.0f} of {means[metric]['steps']:.0f} steps, {means[metric]['tenants']} tenants",
        ]
        for metric in sorted(means, key=lambda m: order.index(m) if m in order else len(order))
    ]
    pct = f"{fraction:.0%}"
    return render_table(
        ["reward, loss, log_prob, mean_len", f"first {pct} steps", f"last {pct} steps", "window"], rows
    )


def parse_cookbook_rewards(paths: Iterable[str]) -> dict[str, dict[str, list[float]]]:
    """``reward/total`` per step from tinker-cookbook client logs, one tenant per file."""
    series: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        rewards: list[float] = []
        with open(path, errors="replace") as handle:
            for line in handle:
                if match := _COOKBOOK_REWARD.search(line):
                    rewards.append(float(match.group(1)))
        if rewards:
            series[os.path.basename(path)] = {"reward": rewards}
    return series


def merge_series(*parts: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for part in parts:
        for tenant, metrics in part.items():
            merged.setdefault(tenant, {}).update(metrics)
    return merged


def render_profile(snapshot: dict[str, dict[str, float]]) -> str:
    rows = [
        [
            op,
            int(value["calls"]),
            int(value["size"]),
            f"{value['total_s']:.1f}",
            f"{value['mean_s']:.2f}",
            f"{value['p90_s']:.2f}",
            f"{value['max_s']:.2f}",
            f"{value['share']:.1%}",
        ]
        for op, value in sorted(snapshot.items(), key=lambda item: -item[1]["total_s"])
    ]
    return render_table(
        ["gateway op", "calls", "size", "total s", "mean s", "p90 s", "max s", "share of trainer time"], rows
    )


def gpu_peaks(path: str) -> dict[str, float] | None:
    """Peak memory and SM utilization over one node's nvidia-smi CSV (timestamp, index, used, total, util)."""
    used: list[int] = []
    util: list[int] = []
    total = 0
    with open(path) as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            used.append(int(row[2].split()[0]))
            total = int(row[3].split()[0])
            util.append(int(row[4].split()[0]))
    if not used:
        return None
    return {"mem_peak_mib": max(used), "mem_total_mib": total, "util_peak": max(util), "samples": len(used)}


def render_gpu(peaks_by_node: dict[str, dict[str, float]], roles: dict[str, str] | None = None) -> str:
    """One row per node: peak memory as MiB and as a share of the GPU, peak SM utilization."""
    rows = []
    for node, peaks in peaks_by_node.items():
        label = f"{(roles or {}).get(node, 'node')} GPUs ({node})"
        share = 100 * peaks["mem_peak_mib"] / peaks["mem_total_mib"] if peaks["mem_total_mib"] else 0.0
        rows.append(
            [
                label,
                f"{int(peaks['mem_peak_mib']):,} / {int(peaks['mem_total_mib']):,} MiB ({share:.1f}%)",
                f"{int(peaks['util_peak'])}%",
                int(peaks["samples"]),
            ]
        )
    return render_table(["gpu", "peak memory", "peak SM util", "samples"], rows)


def parse_serve_log(path: str) -> dict:
    """The gateway facts a report needs: slots and binding bound, adapter cap, node roles, the last profile snapshot."""
    facts: dict = {"trainer_nodes": set(), "engine_nodes": set()}
    with open(path, errors="replace") as handle:
        for line in handle:
            if "slots" not in facts and (match := _CAPACITY_LINE.search(line)):
                facts["slots"], facts["binding"] = int(match.group(1)), match.group(2)
            elif match := _LOADED_LORAS_LINE.search(line):
                facts["loaded_loras"] = int(match.group(1))
            elif match := _TRAINER_NODE_LINE.search(line):
                facts["trainer_nodes"].add(match.group(1))
            elif match := _ENGINE_NODE_LINE.search(line):
                facts["engine_nodes"].add(match.group(1))
            elif (start := line.find(PROFILE_LOG_PREFIX)) >= 0:
                try:
                    facts["profile"] = json.loads(line[start + len(PROFILE_LOG_PREFIX) :])
                except json.JSONDecodeError:
                    continue
            elif (start := line.find(METRICS_LOG_PREFIX)) >= 0:
                try:
                    facts["metrics"] = json.loads(line[start + len(METRICS_LOG_PREFIX) :])
                except json.JSONDecodeError:
                    continue
    return facts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the multi-LoRA gateway's timing after a run.")
    parser.add_argument(
        "--serve-log", help="the gateway's log: the capacity line, node roles and the last profile snapshot"
    )
    parser.add_argument(
        "--gpu-csv",
        action="append",
        default=[],
        metavar="gpu-<ip>.csv",
        help="a node's nvidia-smi samples (repeatable); the node is named by the file, its role read from the log",
    )
    parser.add_argument(
        "--client-log", action="append", default=[], help="tinker-cookbook tenant logs (reward/total per step)"
    )
    args = parser.parse_args(argv)
    if not args.serve_log and not args.gpu_csv:
        parser.error("pass --serve-log and/or --gpu-csv")
    sections = []
    facts: dict = {"trainer_nodes": set(), "engine_nodes": set()}
    if args.serve_log:
        facts = parse_serve_log(args.serve_log)
        rows = []
        if "slots" in facts:
            rows.append(["slots", facts["slots"], f"bound by {facts['binding']}"])
        if "loaded_loras" in facts:
            rows.append(["engine adapter versions kept", facts["loaded_loras"], "--sglang-max-loaded-loras"])
        if rows:
            sections.append(render_table(["gateway", "value", "note"], rows))
        if "profile" in facts:
            sections.append(render_profile(facts["profile"]))
    series = merge_series(facts.get("metrics", {}), parse_cookbook_rewards(args.client_log))
    if series:
        sections.append(render_metrics(series))
    peaks_by_node = {}
    for path in args.gpu_csv:
        node = os.path.basename(path).removeprefix("gpu-").removesuffix(".csv")
        if (peaks := gpu_peaks(path)) is not None:
            peaks_by_node[node] = peaks
    if peaks_by_node:
        roles = {node: "trainer" for node in facts["trainer_nodes"] - facts["engine_nodes"]}
        roles.update({node: "SGLang" for node in facts["engine_nodes"] - facts["trainer_nodes"]})
        sections.append(render_gpu(peaks_by_node, roles))
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
