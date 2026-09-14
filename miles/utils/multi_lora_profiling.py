"""Timing for the multi-LoRA Tinker gateway.

Two vantage points share one vocabulary and one table renderer.

* :class:`PhaseTimer` is client-side: the wall time one tenant waits through in a
  step, queueing behind other tenants included. The phases are ``fwd_bwd``,
  ``optim``, ``publish`` (save_weights_for_sampler + create_sampling_client) and
  ``rollout``. A client closes each step with :meth:`PhaseTimer.step`;
  :func:`summarize` folds every tenant's records into one :class:`Summary`.
* :class:`OpProfiler` is gateway-side: where the trainer's time goes, per backend
  op (``forward_backward``, ``optim_step``, ``export_slot``, ``push_slot``, ...),
  whichever client drives it. The gateway logs its snapshot after every
  optimizer step as ``multi-LoRA profile: {json}``.

Render both after a run::

    python -m miles.utils.multi_lora_profiling --summary-json summary.json --serve-log serve.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from collections.abc import Iterable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field

__all__ = [
    "PHASES",
    "PROFILE_LOG_PREFIX",
    "OpProfiler",
    "PhaseTimer",
    "Stats",
    "StepRecord",
    "Summary",
    "gpu_peaks",
    "parse_serve_log",
    "render_gpu",
    "render_profile",
    "render_summary",
    "render_table",
    "summarize",
]

PHASES = ("fwd_bwd", "optim", "publish", "rollout")
PROFILE_LOG_PREFIX = "multi-LoRA profile: "
_CAPACITY_LINE = re.compile(r"multi-LoRA capacity: (\d+) slots, bound by (.+?) \[")
_LOADED_LORAS_LINE = re.compile(r"engines keep at most (\d+) adapter versions loaded")
_TRAINER_NODE_LINE = re.compile(r"MegatronTrainRayActor pid=\d+, ip=([0-9.]+)")
_ENGINE_NODE_LINE = re.compile(r"CommandActor pid=\d+.*Uvicorn running on http://([0-9.]+):")
_PHASE_LABELS = {
    "fwd_bwd": "forward/backward wait",
    "optim": "optim_step wait",
    "publish": "publish (save + sampling client)",
    "rollout": "rollout",
}


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


# ------------------------------------------------------------------ client side


@dataclass
class StepRecord:
    """One tenant's step: seconds per phase, plus values the client wants averaged (acc, lengths)."""

    step: int
    phases: dict[str, float]
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.phases.values())


class PhaseTimer:
    """Accumulate wall time per phase; :meth:`step` closes the step and returns its record."""

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + time.perf_counter() - start

    def step(self, step: int, **extras: float) -> StepRecord:
        record = StepRecord(step=step, phases=dict(self._phases), extras={k: float(v) for k, v in extras.items()})
        self._phases.clear()
        return record


@dataclass
class Summary:
    """Every tenant's steps folded together: per-phase statistics and each phase's share of a step."""

    n_clients: int
    passed: int
    failed: list[str]
    elapsed_s: float
    client_steps: int
    phases: dict[str, Stats]
    shares: dict[str, float]
    step_total: Stats | None
    per_step: dict[int, dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Summary:
        return cls(
            n_clients=data["n_clients"],
            passed=data["passed"],
            failed=list(data["failed"]),
            elapsed_s=data["elapsed_s"],
            client_steps=data["client_steps"],
            phases={name: Stats(**value) for name, value in data["phases"].items()},
            shares=dict(data["shares"]),
            step_total=Stats(**data["step_total"]) if data.get("step_total") else None,
            per_step={int(step): dict(entry) for step, entry in data["per_step"].items()},
        )


def summarize(
    records_by_client: dict[str, list[StepRecord]], failed: Iterable[str] = (), elapsed_s: float = 0.0
) -> Summary:
    """Fold every client's records (failed clients included, with whatever steps they finished).

    A phase's share is its mean over the sum of the phase means; when every record carries all
    four phases that is its share of a step."""
    records = [record for client_records in records_by_client.values() for record in client_records]
    phases = {
        name: Stats.of([record.phases[name] for record in records if name in record.phases])
        for name in PHASES
        if any(name in record.phases for record in records)
    }
    mean_sum = sum(stat.mean for stat in phases.values())
    shares = {name: (stat.mean / mean_sum if mean_sum else 0.0) for name, stat in phases.items()}
    complete = [record.total for record in records if all(name in record.phases for name in PHASES)]
    per_step: dict[int, dict[str, float]] = {}
    for step in sorted({record.step for record in records}):
        rows = [record for record in records if record.step == step]
        entry = {"clients": float(len(rows))}
        for key in sorted({key for record in rows for key in record.extras}):
            values = [record.extras[key] for record in rows if key in record.extras]
            entry[key] = max(values) if key.startswith("max_") else statistics.fmean(values)
        per_step[step] = entry
    failed = list(failed)
    return Summary(
        n_clients=len(records_by_client),
        passed=len(records_by_client) - len(failed),
        failed=failed,
        elapsed_s=elapsed_s,
        client_steps=len(records),
        phases=phases,
        shares=shares,
        step_total=Stats.of(complete) if complete else None,
        per_step=per_step,
    )


# ----------------------------------------------------------------- gateway side


class OpProfiler:
    """Wall time per gateway op. ``size`` counts what one call covered (datums of a forward
    pass, slots of an optimizer step, samples of a request) so units and tenant-steps stay apart."""

    def __init__(self) -> None:
        self._seconds: dict[str, list[float]] = {}
        self._sizes: dict[str, int] = {}

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

    def report(self) -> str:
        return render_profile(self.snapshot())

    def reset(self) -> None:
        self._seconds.clear()
        self._sizes.clear()


# -------------------------------------------------------------------- rendering


def render_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    """A plain-text grid: left-aligned columns sized to their content."""
    headers = [str(header) for header in headers]
    body = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in body)]) for i, header in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in body)])


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


def render_summary(summary: Summary) -> str:
    clients = f"{summary.passed}/{summary.n_clients} passed"
    if summary.failed:
        clients += f", failed: {' '.join(summary.failed)}"
    rows: list[list[object]] = [
        ["clients", clients, ""],
        ["client-steps", summary.client_steps, ""],
        ["elapsed", f"{summary.elapsed_s:.0f} s", ""],
    ]
    for name in PHASES:
        if name in summary.phases:
            rows.append(
                [_PHASE_LABELS[name], summary.phases[name].describe(), f"{summary.shares[name]:.1%} of a step"]
            )
    if summary.step_total is not None:
        rows.append(["one step, one client", summary.step_total.describe(), "100%"])
    for step, entry in summary.per_step.items():
        details = ", ".join(f"{key} {value:.3g}" for key, value in entry.items() if key != "clients")
        rows.append([f"step {step}", f"{int(entry['clients'])} clients", details])
    return render_table(["client side", "time / value", "share"], rows)


def gpu_peaks(path: str) -> dict[str, float] | None:
    """Peaks over every sample of every GPU in one node's nvidia-smi CSV
    (``timestamp, index, memory.used, memory.total, utilization.gpu`` rows, ``--format=csv,noheader``)."""
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
    """The gateway facts a report needs: the resolved slots and their binding bound, the engine
    adapter cap, which node ran the trainer and which the engines, and the last profile snapshot."""
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
    return facts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the multi-LoRA gateway's timing after a run.")
    parser.add_argument("--summary-json", help="client-side summary written by run_multi_tenant_example.py")
    parser.add_argument("--serve-log", help="the gateway's log: capacity line and the last profile snapshot")
    parser.add_argument(
        "--gpu-csv",
        action="append",
        default=[],
        metavar="gpu-<ip>.csv",
        help="a node's nvidia-smi samples (repeatable); the node is named by the file, its role read from the log",
    )
    args = parser.parse_args(argv)
    if not args.summary_json and not args.serve_log and not args.gpu_csv:
        parser.error("pass --summary-json, --serve-log and/or --gpu-csv")
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
    if args.summary_json:
        with open(args.summary_json) as handle:
            sections.append(render_summary(Summary.from_dict(json.load(handle))))
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
