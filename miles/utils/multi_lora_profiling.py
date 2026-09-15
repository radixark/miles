"""Per-LoRA-per-step timing of the multi-LoRA gateway: trainer side, engine side, the tenant's view; GPU peaks."""

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
    "REQUESTS_LOG_PREFIX",
    "OpProfiler",
    "Stats",
    "client_steps",
    "gpu_peaks",
    "head_tail_means",
    "lora_steps",
    "merge_series",
    "parse_cookbook_rewards",
    "parse_cookbook_step_times",
    "parse_serve_log",
    "render_client",
    "render_gpu",
    "render_metrics",
    "render_profile",
    "render_table",
]

PROFILE_LOG_PREFIX = "multi-LoRA profile: "
METRICS_LOG_PREFIX = "multi-LoRA metrics: "
REQUESTS_LOG_PREFIX = "multi-LoRA requests: "
_COOKBOOK_STEP = re.compile(r"^\s*Step (\d+)\s*$")
_COOKBOOK_REWARD = re.compile(r"reward/total\s*│\s*([-0-9.eE+]+)")
_COOKBOOK_STEP_TIME = re.compile(r"time/total\s*│\s*([-0-9.eE+]+)")
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
        self._requests: list[list] = []

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

    def request(self, op: str, model: str, created_at: float, finished_at: float) -> None:
        """One settled tenant request, arrival to result."""
        self._requests.append([model, op, round(created_at, 3), round(finished_at, 3)])

    def drain_requests(self) -> list[list]:
        """The requests settled since the last drain; logged as one line each time."""
        requests, self._requests = self._requests, []
        return requests

    def report(self) -> str:
        return render_profile(self.snapshot())

    def reset(self) -> None:
        self._seconds.clear()
        self._sizes.clear()
        self._series.clear()
        self._requests.clear()


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


def parse_cookbook_step_times(paths: Iterable[str]) -> list[float]:
    """``time/total`` of every step in tinker-cookbook client logs: one LoRA's step as the tenant clocked it."""
    times: list[float] = []
    for path in paths:
        with open(path, errors="replace") as handle:
            times.extend(float(match.group(1)) for line in handle if (match := _COOKBOOK_STEP_TIME.search(line)))
    return times


def merge_series(*parts: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    merged: dict[str, dict[str, list[float]]] = {}
    for part in parts:
        for tenant, metrics in part.items():
            merged.setdefault(tenant, {}).update(metrics)
    return merged


_OP_NOTES = {
    "push_slot": "adapter tensors pushed to every engine",
    "export_slot": "adapter gathered to rank 0, written as safetensors",
    "forward_backward": "one unit packs several LoRAs' datums: total / LoRA-steps",
    "forward_only": "forward_backward without the backward",
    "optim_step": "one barrier steps several LoRAs' optimizers: total / LoRA-steps",
    "load_slot": "once per LoRA, at create_model",
    "save_slot": "save_state",
    "unload_slot": "once per LoRA, at release",
}
_CLIENT_PHASES = (
    ("publish", ("save_weights_for_sampler",)),
    ("rollout", ("sample",)),
    ("forward_backward", ("forward_backward", "forward_only")),
    ("optim_step", ("optim_step",)),
)


def lora_steps(snapshot: dict[str, dict[str, float]]) -> int:
    """LoRA-steps: optimizer steps summed over LoRAs, the per-LoRA-per-step denominator."""
    return int(snapshot.get("optim_step", {}).get("size", 0)) or 1


def render_profile(snapshot: dict[str, dict[str, float]]) -> str:
    """server: what one LoRA's step really occupies the trainer; sample: the engine side, alongside it."""
    steps = lora_steps(snapshot)
    trainer = {op: value for op, value in snapshot.items() if op != "sample"}
    busy = sum(value["total_s"] for value in trainer.values()) or 1.0
    rows = [
        [
            "total",
            f"{busy / steps:.2f}",
            "",
            "",
            "100%",
            f"trainer time one LoRA's step really takes ({steps} LoRA-steps)",
        ]
    ]
    for op, value in sorted(trainer.items(), key=lambda item: -item[1]["total_s"]):
        rows.append(
            [
                op,
                f"{value['total_s'] / steps:.2f}",
                f"{value['mean_s']:.2f}",
                int(value["calls"]),
                f"{value['total_s'] / busy:.1%}",
                _OP_NOTES.get(op, ""),
            ]
        )
    header = [
        "server (trainer side, one LoRA at a time)",
        "per LoRA per step s",
        "per call s",
        "calls",
        "share of trainer busy time",
        "note",
    ]
    table = render_table(header, rows)
    if (sample := snapshot.get("sample")) is not None:
        note = (
            f"{sample['size'] / sample['calls']:.0f} sequences per call, {sample['calls'] / steps:.1f} calls"
            " per LoRA-step; calls overlap, the client table has the wave"
        )
        rows = [["sample", f"{sample['total_s'] / steps:.2f}", f"{sample['mean_s']:.2f}", int(sample["calls"]), note]]
        header = ["sample (engine side, alongside the trainer)", "per LoRA per step s", "per call s", "calls", "note"]
        table += "\n\n" + render_table(header, rows)
    return table


def client_steps(requests: list[list]) -> list[dict[str, float]]:
    """One LoRA's step as the tenant saw it: publish, rollout wave, forward_backward, optim_step, arrival to result."""
    by_model: dict[str, list[tuple[float, str, float]]] = {}
    for model, op, created_at, finished_at in requests:
        by_model.setdefault(model, []).append((created_at, op, finished_at))
    steps: list[dict[str, float]] = []
    for entries in by_model.values():
        entries.sort()
        model_steps: list[dict[str, float]] = []
        step: dict[str, float] = {}
        wave: list[tuple[float, float]] = []
        for created_at, op, finished_at in entries:
            if op == "sample":
                wave.append((created_at, finished_at))
                continue
            phase = next((name for name, ops in _CLIENT_PHASES if op in ops), None)
            if phase is None:
                continue
            step[phase] = step.get(phase, 0.0) + finished_at - created_at
            if phase == "optim_step":  # the optimizer step closes a step; its rollout wave came before it
                if wave:
                    step["rollout"] = max(end for _, end in wave) - min(start for start, _ in wave)
                    wave = []
                model_steps.append(step)
                step = {}
        if model_steps and step.get("publish"):  # the closing publish after the last step
            model_steps[-1]["publish"] = model_steps[-1].get("publish", 0.0) + step["publish"]
        steps.extend(model_steps)
    return steps


def render_client(
    steps: list[dict[str, float]], snapshot: dict[str, dict[str, float]], logged_step_s: list[float] = ()
) -> str:
    """client: one LoRA's step as the tenant saw it, per phase, beside the trainer work each phase really needed."""
    if not steps:
        return ""
    n = lora_steps(snapshot)
    means = {phase: statistics.fmean(step.get(phase, 0.0) for step in steps) for phase, _ in _CLIENT_PHASES}
    one_step = sum(means.values()) or 1.0
    work = {
        "publish": sum(snapshot.get(op, {}).get("total_s", 0.0) for op in ("export_slot", "push_slot")) / n,
        "rollout": means["rollout"],
        "forward_backward": sum(
            snapshot.get(op, {}).get("total_s", 0.0) for op in ("forward_backward", "forward_only")
        )
        / n,
        "optim_step": snapshot.get("optim_step", {}).get("total_s", 0.0) / n,
    }
    notes = {
        "publish": "save_weights_for_sampler: this LoRA's export (+ push); the rest is the queue",
        "rollout": "first sample request in to last sample result out; engine side, counted as work",
        "forward_backward": "its share of the packed unit; the rest is the queue",
        "optim_step": "its share of the barrier; the rest is the queue",
    }
    total_work = sum(work.values())
    rows = [
        [
            "one step",
            f"{one_step:.1f}",
            "100%",
            f"{total_work:.1f}",
            f"{one_step - total_work:.1f} ({(one_step - total_work) / one_step:.0%})",
            f"{len(steps)} LoRA-steps, sum of the phases below",
        ]
    ]
    for phase, _ in _CLIENT_PHASES:
        rows.append(
            [
                phase,
                f"{means[phase]:.1f}",
                f"{means[phase] / one_step:.1%}",
                f"{work[phase]:.1f}",
                f"{means[phase] - work[phase]:.1f}",
                notes[phase],
            ]
        )
    if logged_step_s:
        note = f"time/total over {len(logged_step_s)} tenant steps; adds the tenant's own work and result polling"
        rows.append(["one step, as the tenant logged it", f"{statistics.fmean(logged_step_s):.1f}", "", "", "", note])
    header = [
        "client (one LoRA's step, as the tenant saw it)",
        "per LoRA per step s",
        "share of one step",
        "work s",
        "queueing s",
        "note",
    ]
    return render_table(header, rows)


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
            elif (start := line.find(REQUESTS_LOG_PREFIX)) >= 0:
                try:
                    facts.setdefault("requests", []).extend(json.loads(line[start + len(REQUESTS_LOG_PREFIX) :]))
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
            if table := render_client(
                client_steps(facts.get("requests", [])), facts["profile"], parse_cookbook_step_times(args.client_log)
            ):
                sections.append(table)
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
