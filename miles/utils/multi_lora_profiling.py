import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import time
from functools import wraps

logger = logging.getLogger(__name__)

PROFILE_LOG_PREFIX = "multi-LoRA profile: "
METRICS_LOG_PREFIX = "multi-LoRA metrics: "
REQUESTS_LOG_PREFIX = "multi-LoRA requests: "

TRAINER_OPS = (
    "load_slot",
    "unload_slot",
    "forward_backward",
    "forward_only",
    "optim_step",
    "save_slot",
    "export_slot",
)
ENGINE_OPS = ("sample",)
API_PATHS = {
    "/api/v1/create_model": "create_model",
    "/api/v1/forward_backward": "forward_backward",
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_weights",
    "/api/v1/load_weights": "load_weights",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
    "/api/v1/asample": "asample",
}
RETRIEVE_PATH = "/api/v1/retrieve_future"


def _quantile(values: list[float], fraction: float) -> float:
    return values[min(len(values) - 1, int(fraction * len(values)))]


class Profiler:
    def __init__(self) -> None:
        self._seconds: dict[str, list[float]] = {}
        self._sizes: dict[str, int] = {}
        self._series: dict[str, dict[str, list[float]]] = {}
        self._requests: list[list] = []

    def record(self, op: str, seconds: float, size: int = 1) -> None:
        self._seconds.setdefault(op, []).append(seconds)
        self._sizes[op] = self._sizes.get(op, 0) + size

    def observe(self, slot: int, **metrics: float) -> None:
        for name, value in metrics.items():
            self._series.setdefault(str(slot), {}).setdefault(name, []).append(float(value))

    def request(self, op: str, tenant: str, created_at: float, finished_at: float) -> None:
        self._requests.append([tenant, op, round(created_at, 3), round(finished_at, 3)])

    def snapshot(self) -> dict[str, dict[str, float]]:
        trainer_total = sum(sum(values) for op, values in self._seconds.items() if op in TRAINER_OPS) or 1.0
        snapshot = {}
        for op, values in self._seconds.items():
            ordered = sorted(values)
            snapshot[op] = {
                "calls": len(values),
                "size": self._sizes[op],
                "total_s": sum(values),
                "mean_s": statistics.fmean(values),
                "p50_s": _quantile(ordered, 0.5),
                "p90_s": _quantile(ordered, 0.9),
                "max_s": ordered[-1],
                "share": sum(values) / trainer_total if op in TRAINER_OPS else 0.0,
            }
        return snapshot

    def series(self) -> dict[str, dict[str, list[float]]]:
        return self._series

    def drain_requests(self) -> list[list]:
        requests, self._requests = self._requests, []
        return requests

    def log_requests(self) -> None:
        if requests := self.drain_requests():
            logger.info("%s%s", REQUESTS_LOG_PREFIX, json.dumps(requests))

    def log(self) -> None:
        if self._seconds:
            logger.info("%s%s", PROFILE_LOG_PREFIX, json.dumps(self.snapshot()))
        if self._series:
            logger.info("%s%s", METRICS_LOG_PREFIX, json.dumps(self._series))
        self.log_requests()


def _call_size(op: str, args: tuple, kwargs: dict) -> int:
    if op in ("forward_backward", "forward_only"):
        return len(args[1] if len(args) > 1 else kwargs["slot_datums"])
    if op == "optim_step":
        return len(args[0] if args else kwargs["adam_params_by_slot"])
    if op == "sample":
        return (args[0] if args else kwargs["payload"]).get("num_samples", 1)
    return 1


def _response_logprobs(datum: dict, output: dict) -> list[float]:
    mask = datum.get("weights") or datum.get("advantages")
    if mask is None:
        return list(output["logprobs"])
    return [value for value, keep in zip(output["logprobs"], mask, strict=False) if keep]


def _observe_step(profiler: Profiler, slot_datums: list, outputs) -> None:
    if not isinstance(outputs, list):
        return
    per_slot: dict[int, list[tuple[dict, dict]]] = {}
    for (slot, datum), output in zip(slot_datums, outputs, strict=True):
        per_slot.setdefault(slot, []).append((datum, output))
    for slot, pairs in per_slot.items():
        logprobs = [value for datum, output in pairs for value in _response_logprobs(datum, output)]
        profiler.observe(
            slot,
            loss=sum(output["loss"] for _, output in pairs) / len(pairs),
            log_prob=sum(logprobs) / len(logprobs) if logprobs else 0.0,
            mean_len=len(logprobs) / len(pairs),
        )


def _timed(profiler: Profiler, op: str, method):
    @wraps(method)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await method(*args, **kwargs)
        finally:
            profiler.record(op, time.perf_counter() - start, _call_size(op, args, kwargs))
        if op == "forward_backward":
            _observe_step(profiler, args[1] if len(args) > 1 else kwargs["slot_datums"], result)
        if op in ("forward_backward", "optim_step"):
            profiler.log()
        return result

    return wrapper


def instrument_backend(backend):
    profiler = Profiler()
    for op in TRAINER_OPS + ENGINE_OPS:
        if (method := getattr(backend, op, None)) is not None:
            setattr(backend, op, _timed(profiler, op, method))
    backend.profiler = profiler
    return backend


class _RequestTiming:
    def __init__(self, app, profiler: Profiler) -> None:
        self.app = app
        self.profiler = profiler
        self.pending: dict[str, tuple[str, str, float]] = {}

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        if scope["type"] != "http" or (path not in API_PATHS and path != RETRIEVE_PATH):
            await self.app(scope, receive, send)
            return
        arrival = time.monotonic()
        messages = []
        while True:
            messages.append(await receive())
            if messages[-1]["type"] != "http.request" or not messages[-1].get("more_body"):
                break
        request_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.request")
        response_chunks: list[bytes] = []

        async def replay():
            return messages.pop(0) if messages else await receive()

        async def send_recorded(message):
            if message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))
            await send(message)

        await self.app(scope, replay, send_recorded)
        tenant = next((value.decode() for key, value in scope["headers"] if key == b"x-api-key"), "")
        self._settle(path, tenant, request_body, b"".join(response_chunks), arrival)

    def _settle(self, path: str, tenant: str, request_body: bytes, response_body: bytes, arrival: float) -> None:
        if path in API_PATHS:
            if (request_id := _json_field(response_body, "request_id")) is not None:
                self.pending[request_id] = (API_PATHS[path], tenant, arrival)
            return
        request_id = _json_field(request_body, "request_id")
        if request_id not in self.pending or _json_field(response_body, "type") == "try_again":
            return
        op, tenant, created_at = self.pending.pop(request_id)
        if _json_field(response_body, "error") is None:
            self.profiler.request(op, tenant, created_at, time.monotonic())
        if op != "asample" or not self.pending:
            self.profiler.log_requests()


def _json_field(body: bytes, key: str):
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value.get(key) if isinstance(value, dict) else None


def instrument_app(app, profiler: Profiler):
    app.add_middleware(_RequestTiming, profiler=profiler)
    return app


OP_API = {
    "load_slot": "create_model",
    "forward_backward": "forward_backward: one unit packs several tenants' requests",
    "forward_only": "forward: one unit packs several tenants' requests",
    "optim_step": "optim_step: one barrier packs several tenants' requests",
    "save_slot": "save_weights (save_state)",
    "export_slot": "save_weights_for_sampler: adapter gathered to rank 0, written as safetensors",
    "unload_slot": "lease expiry, no API",
    "sample": "asample: on the inference engines, alongside the trainer, not in the share",
}
API_OPS = {
    "create_model": ("load_slot",),
    "forward_backward": ("forward_backward",),
    "optim_step": ("optim_step",),
    "save_weights": ("save_slot",),
    "load_weights": ("load_slot",),
    "save_weights_for_sampler": ("export_slot",),
    "asample": ("sample",),
}
API_SERVER_OP = {
    "create_model": "load_slot",
    "forward_backward": "forward_backward (packed)",
    "optim_step": "optim_step (packed)",
    "save_weights": "save_slot",
    "load_weights": "load_slot",
    "save_weights_for_sampler": "export_slot",
    "asample": "sample (inference engine, alongside the trainer)",
}
STEP_PHASES = (
    ("publish", "save_weights_for_sampler", ("save_weights_for_sampler",)),
    ("rollout", "asample (wave)", ("asample",)),
    ("train", "forward_backward + optim_step", ("forward_backward", "optim_step")),
)
_CAPACITY_LINE = re.compile(r"multi-LoRA capacity: (\d+) slots, bound by (.+?) (?:\(gpu=|\[)")
_LOADED_LORAS_FLAG = re.compile(r"max-loaded-loras (\d+)")
_TRAINER_NODE_LINE = re.compile(r"TrainRayActor pid=\d+, ip=([0-9.]+)")
_ENGINE_NODE_LINE = re.compile(r"sglang\.launch_server .*?--host ([0-9.]+)")
_COOKBOOK_REWARD = re.compile(r"reward/total\s*│\s*([-0-9.eE+]+)")
_COOKBOOK_STEP_TIME = re.compile(r"time/total\s*│\s*([-0-9.eE+]+)")


def render_table(headers, rows) -> str:
    headers = [str(header) for header in headers]
    body = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in body)]) for i, header in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in body)])


def lora_steps(snapshot: dict) -> int:
    return int(snapshot.get("optim_step", {}).get("size", 0)) or 1


def render_server(snapshot: dict) -> str:
    steps = lora_steps(snapshot)
    trainer = {op: value for op, value in snapshot.items() if op in TRAINER_OPS}
    busy = sum(value["total_s"] for value in trainer.values()) or 1.0
    rows = [
        [
            "total",
            "trainer",
            f"{busy / steps:.2f}",
            "",
            "",
            "",
            "",
            "100%",
            f"trainer time one LoRA's step takes ({steps} LoRA-steps)",
        ]
    ]
    for op, value in sorted(trainer.items(), key=lambda item: -item[1]["total_s"]):
        rows.append(
            [
                op,
                "trainer",
                f"{value['total_s'] / steps:.2f}",
                f"{value['mean_s']:.2f}",
                f"{value['p90_s']:.2f}",
                f"{value['max_s']:.2f}",
                int(value["calls"]),
                f"{value['share']:.1%}",
                OP_API.get(op, ""),
            ]
        )
    for op in ENGINE_OPS:
        if (value := snapshot.get(op)) is not None:
            note = f"{OP_API[op]}; {value['size'] / value['calls']:.0f} sequences per call"
            rows.append(
                [
                    op,
                    "inference engine",
                    f"{value['total_s'] / steps:.2f}",
                    f"{value['mean_s']:.2f}",
                    f"{value['p90_s']:.2f}",
                    f"{value['max_s']:.2f}",
                    int(value["calls"]),
                    "-",
                    note,
                ]
            )
    header = [
        "op",
        "where",
        "per LoRA per step s",
        "mean s",
        "p90 s",
        "max s",
        "calls",
        "share of trainer busy time",
        "tinker API / note",
    ]
    return "server: operation-level time profiling (per LoRA per step)\n" + render_table(header, rows)


def _by_tenant(requests: list[list]) -> dict[str, list[tuple[float, str, float]]]:
    by_tenant: dict[str, list[tuple[float, str, float]]] = {}
    for tenant, op, created_at, finished_at in requests:
        by_tenant.setdefault(tenant, []).append((created_at, op, finished_at))
    return by_tenant


def render_client_api(requests: list[list], snapshot: dict) -> str:
    by_tenant = _by_tenant(requests)
    if not by_tenant:
        return ""
    tenants = len(by_tenant)
    walls = [
        max(end for _, _, end in entries) - min(start for start, _, _ in entries) for entries in by_tenant.values()
    ]
    durations: dict[str, list[float]] = {}
    for _, op, created_at, finished_at in requests:
        durations.setdefault(op, []).append(finished_at - created_at)
    rows = []
    total_work = 0.0
    for api, values in sorted(durations.items(), key=lambda item: -sum(item[1])):
        ops = [snapshot[op] for op in API_OPS.get(api, ()) if op in snapshot]
        work = sum(op["total_s"] for op in ops) / len(values)
        share = sum(op["share"] for op in ops)
        total_work += work * len(values) / tenants
        ordered = sorted(values)
        rows.append(
            [
                api,
                f"{len(values) / tenants:.2f}",
                f"{statistics.fmean(values):.1f}",
                f"{_quantile(ordered, 0.5):.1f}",
                f"{_quantile(ordered, 0.9):.1f}",
                f"{ordered[-1]:.1f}",
                f"{sum(values) / tenants:.1f}",
                f"{work:.2f}",
                f"{statistics.fmean(values) - work:.1f}",
                "-" if api == "asample" else f"{share:.1%}",
                API_SERVER_OP.get(api, ""),
            ]
        )
    wall = statistics.fmean(walls)
    summed = sum(finished_at - created_at for _, _, created_at, finished_at in requests) / tenants
    total = [
        "total",
        f"{len(requests) / tenants:.2f}",
        "",
        "",
        "",
        "",
        f"{wall:.1f}",
        f"{total_work:.1f}",
        f"{wall - total_work:.1f} ({(wall - total_work) / wall:.0%})",
        "100%",
        f"tenant wall time, first request in to last result out; per-API totals sum to {summed:.1f} s (calls in flight together)",
    ]
    header = [
        "tinker API",
        "calls per LoRA",
        "mean s",
        "p50 s",
        "p90 s",
        "max s",
        "total per LoRA s",
        "work per call s",
        "queueing per call s",
        "share of trainer busy time",
        "server op",
    ]
    return "client: API-level time profiling (arrival to result, as the tenant saw it)\n" + render_table(
        header, [total, *rows]
    )


def client_steps(requests: list[list]) -> list[dict[str, float]]:
    steps: list[dict[str, list[float]]] = []
    for entries in _by_tenant(requests).values():
        entries.sort()
        spans: dict[str, list[float]] = {}
        for created_at, api, finished_at in entries:
            phase = next((name for name, _, apis in STEP_PHASES if api in apis), None)
            if phase is None:
                continue
            if phase == "publish" and spans:
                steps.append(spans)
                spans = {}
            span = spans.setdefault(phase, [created_at, finished_at])
            span[0], span[1] = min(span[0], created_at), max(span[1], finished_at)
            if api == "optim_step":
                steps.append(spans)
                spans = {}
        if spans and set(spans) != {"publish"}:
            steps.append(spans)
    return [
        {phase: end - start for phase, (start, end) in spans.items()}
        | {"one step": max(end for _, end in spans.values()) - min(start for start, _ in spans.values())}
        for spans in steps
    ]


def render_client_steps(steps: list[dict[str, float]], snapshot: dict, logged_step_s: list[float]) -> str:
    if not steps:
        return ""
    n = lora_steps(snapshot)
    means = {
        phase: statistics.fmean(step.get(phase, 0.0) for step in steps)
        for phase in ("one step", *(name for name, _, _ in STEP_PHASES))
    }
    one_step = means["one step"] or 1.0
    work = {
        name: (
            means["rollout"]
            if name == "rollout"
            else sum(snapshot.get(op, {}).get("total_s", 0.0) for api in apis for op in API_OPS[api]) / n
        )
        for name, _, apis in STEP_PHASES
    }
    work["one step"] = sum(work.values())
    rows = [
        [
            "one step",
            "first request in, last result out",
            f"{one_step:.1f}",
            "100%",
            f"{work['one step']:.1f}",
            f"{one_step - work['one step']:.1f} ({(one_step - work['one step']) / one_step:.0%})",
        ]
    ]
    for name, api, _ in STEP_PHASES:
        rows.append(
            [
                name,
                api,
                f"{means[name]:.1f}",
                f"{means[name] / one_step:.1%}",
                f"{work[name]:.1f}",
                f"{means[name] - work[name]:.1f}",
            ]
        )
    if logged_step_s:
        rows.append(
            [
                "one step, as the tenant logged it",
                "cookbook time/total",
                f"{statistics.fmean(logged_step_s):.1f}",
                "",
                "",
                "",
            ]
        )
    header = ["phase", "tinker API", "per LoRA per step s", "share of one step", "work s", "queueing s"]
    return f"one LoRA's step, as the tenant saw it (API spans, {len(steps)} LoRA-steps)\n" + render_table(header, rows)


def head_tail_means(series: dict[str, dict[str, list[float]]], fraction: float = 0.1) -> dict[str, dict[str, float]]:
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


def parse_cookbook_logs(paths) -> tuple[dict[str, dict[str, list[float]]], list[float]]:
    series: dict[str, dict[str, list[float]]] = {}
    step_times: list[float] = []
    for path in paths:
        rewards: list[float] = []
        with open(path, errors="replace") as handle:
            for line in handle:
                if match := _COOKBOOK_REWARD.search(line):
                    rewards.append(float(match.group(1)))
                elif match := _COOKBOOK_STEP_TIME.search(line):
                    step_times.append(float(match.group(1)))
        if rewards:
            series[os.path.basename(path)] = {"reward": rewards}
    return series, step_times


def gpu_peaks(path: str) -> dict[str, float] | None:
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


def render_gpu(peaks_by_node: dict[str, dict[str, float]], roles: dict[str, str]) -> str:
    rows = []
    for node, peaks in peaks_by_node.items():
        if peaks["util_peak"] < 5 and peaks["mem_peak_mib"] < 0.02 * peaks["mem_total_mib"]:
            continue
        rows.append(
            [
                f"{roles.get(node, 'node')} GPUs ({node})",
                f"{peaks['mem_peak_mib']:,} / {peaks['mem_total_mib']:,} MiB ({peaks['mem_peak_mib'] / peaks['mem_total_mib']:.1%})",
                f"{peaks['util_peak']:.0f}%",
                int(peaks["samples"]),
            ]
        )
    return render_table(["gpu", "peak memory", "peak SM util", "samples"], rows) if rows else ""


def parse_serve_log(path: str) -> dict:
    facts: dict = {"trainer_nodes": set(), "engine_nodes": set(), "requests": []}
    with open(path, errors="replace") as handle:
        for line in handle:
            if "slots" not in facts and (match := _CAPACITY_LINE.search(line)):
                facts["slots"], facts["binding"] = int(match.group(1)), match.group(2)
            elif "loaded_loras" not in facts and (match := _LOADED_LORAS_FLAG.search(line)):
                facts["loaded_loras"] = int(match.group(1))
            elif match := _TRAINER_NODE_LINE.search(line):
                facts["trainer_nodes"].add(match.group(1))
            elif match := _ENGINE_NODE_LINE.search(line):
                facts["engine_nodes"].add(match.group(1))
            for prefix, key in (
                (PROFILE_LOG_PREFIX, "profile"),
                (METRICS_LOG_PREFIX, "metrics"),
                (REQUESTS_LOG_PREFIX, "requests"),
            ):
                if (start := line.find(prefix)) >= 0:
                    try:
                        value = json.loads(line[start + len(prefix) :])
                    except json.JSONDecodeError:
                        break
                    if key == "requests":
                        facts["requests"].extend(value)
                    else:
                        facts[key] = value
                    break
    return facts


def render_report(facts: dict, gpu_csvs: list[str], client_logs: list[str]) -> str:
    sections = []
    rows = []
    if "slots" in facts:
        rows.append(["slots", facts["slots"], f"bound by {facts['binding']}"])
    if "loaded_loras" in facts:
        rows.append(["engine adapter versions kept", facts["loaded_loras"], "--sglang-max-loaded-loras"])
    if rows:
        sections.append(render_table(["gateway", "value", "note"], rows))
    snapshot = facts.get("profile", {})
    if snapshot:
        sections.append(render_server(snapshot))
    reward_series, step_times = parse_cookbook_logs(client_logs)
    if facts["requests"]:
        sections.append(render_client_api(facts["requests"], snapshot))
        sections.append(render_client_steps(client_steps(facts["requests"]), snapshot, step_times))
    series = {tenant: dict(metrics) for tenant, metrics in facts.get("metrics", {}).items()}
    for tenant, metrics in reward_series.items():
        series.setdefault(tenant, {}).update(metrics)
    if series:
        sections.append(render_metrics(series))
    peaks_by_node = {}
    for path in gpu_csvs:
        if (peaks := gpu_peaks(path)) is not None:
            peaks_by_node[os.path.basename(path).removeprefix("gpu-").removesuffix(".csv")] = peaks
    roles = {node: "trainer" for node in facts["trainer_nodes"] - facts["engine_nodes"]}
    roles.update({node: "SGLang" for node in facts["engine_nodes"] - facts["trainer_nodes"]})
    if peaks_by_node and (table := render_gpu(peaks_by_node, roles)):
        sections.append(table)
    return "\n\n".join(section for section in sections if section)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the multi-LoRA gateway's report from its logs.")
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--gpu-csv", action="append", default=[], metavar="gpu-<ip>.csv")
    parser.add_argument("--client-log", action="append", default=[])
    args = parser.parse_args(argv)
    print(render_report(parse_serve_log(args.serve_log), args.gpu_csv, args.client_log))


if __name__ == "__main__":
    main()
