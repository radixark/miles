"""Time the real dashboard server over a dump, the way the browser drives it."""

from __future__ import annotations

import bisect
import calendar
import json
import resource
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.sglang_scraper import ScrapeMode, SglangScraper
from miles.dashboard.store import GpuSample, MetricStore, Stream

_TIMEOUT_S = 1800
_DEFAULT_LANE_COUNT = 8  # views_timeline.js DEFAULT_LANES
_FOLLOW_REDRAW_S = 5.0  # views_timeline.js FOLLOW_REDRAW_MS


@contextmanager
def serve(dump_details: Path, *, port: int, follow: bool):
    """A ``miles.dashboard.serve`` process with a fresh summary cache; yields (pid, startup_s)."""
    cache_dir = Path(tempfile.mkdtemp(prefix="dashboard_simulator_cache_"))
    command = [
        sys.executable, "-m", "miles.dashboard.serve", "--dump-details", str(dump_details),
        "--port", str(port), "--cache-dir", str(cache_dir),
    ]  # fmt: skip
    # a file, not a pipe: nothing reads stderr while the server runs, and a full pipe would block its writes
    stderr_path = cache_dir / "serve.stderr"
    with open(stderr_path, "wb") as stderr:
        server = subprocess.Popen(command + (["--follow"] if follow else []), stdout=subprocess.DEVNULL, stderr=stderr)
    started = time.perf_counter()
    try:
        while True:
            try:
                timed_get(port, "/api/meta")
                break
            except OSError:
                assert server.poll() is None, stderr_path.read_text()[-3000:]
                time.sleep(0.2)
        yield server.pid, time.perf_counter() - started
    finally:
        server.terminate()
        server.wait()


def timed_get(port: int, path: str, params: dict | None = None) -> tuple[float, int, bytes]:
    query = "" if not params else "?" + urllib.parse.urlencode(params)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}{query}", timeout=_TIMEOUT_S) as response:
            body, status = response.read(), response.status
    except urllib.error.HTTPError as e:
        body, status = e.read(), e.code
    return time.perf_counter() - started, status, body


def _get_json(port: int, path: str, params: dict | None = None):
    _, status, body = timed_get(port, path, params)
    assert status == 200, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


def _rss_mb(pid: int) -> float:
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip()) / 1024


def _default_lane_selection(port: int) -> str:
    """The timeline's no-selection default: lanes evenly spaced over the cluster."""
    lanes = _get_json(port, "/api/timeline/topology")["lanes"]
    return ",".join(f"g:{lanes[(i * len(lanes)) // _DEFAULT_LANE_COUNT]['index']}" for i in range(_DEFAULT_LANE_COUNT))


def bench_static(dump_details: Path, *, port: int, repeats: int) -> dict:
    """Each request the timeline (trailing max window, default lanes) and the
    newest rollout page with one sample's views issue, ``repeats`` times: the
    first is cold, later ones warm."""
    with serve(dump_details, port=port, follow=False) as (pid, startup_s):
        results = dict(startup_s=round(startup_s, 2), idle_rss_mb=round(_rss_mb(pid)))
        meta = _get_json(port, "/api/meta")
        t1 = meta["time_range"][1]
        window = dict(t0=t1 - meta["capabilities"]["max_window_s"], t1=t1)
        selection = _default_lane_selection(port)
        last_rollout_id = _get_json(port, "/api/timeline/bubbles")["bubbles"][-1]["step"]
        requests = [
            ("meta", "/api/meta", None),
            ("topology", "/api/timeline/topology", None),
            ("bubbles", "/api/timeline/bubbles", None),
            ("phases", "/api/timeline/phases", window | dict(lanes=selection)),
            ("gpu", "/api/timeline/gpu", window | dict(max_points=4000, lanes=selection)),
            ("gpu_processes", "/api/timeline/gpu_processes", window | dict(lanes=selection)),
            (
                "engine_series",
                "/api/timeline/engine_series",
                window | dict(metric="sglang_num_running_reqs", max_points=4000),
            ),
            ("advisory", "/api/advisory", None),
            ("fleet", "/api/timeline/fleet", window | dict(x_buckets=600)),
            ("heatmap util", "/api/timeline/heatmap", window | dict(metric="util", x_buckets=1200)),
            ("heatmap lifecycle", "/api/timeline/heatmap", window | dict(metric="lifecycle", x_buckets=1200)),
            ("outliers", "/api/timeline/outliers", window | dict(criterion="lowest_util")),
            ("metrics", "/api/metrics", dict(keys="perf/step_time")),
            ("summary (cold step)", f"/api/rollout/{last_rollout_id - 1}/summary", None),
            ("summary", f"/api/rollout/{last_rollout_id}/summary", None),
            ("groups", f"/api/rollout/{last_rollout_id}/groups", None),
            ("trajectories", f"/api/rollout/{last_rollout_id}/trajectories", None),
        ]
        for name, path, params in requests:
            results[name] = _time_request(port, pid, name, path, params, repeats)
        sample_index = _sample_with_conversation(port, last_rollout_id)
        views = (("sample tokens", "tokens"), ("sample messages", "messages"))
        if sample_index is None:  # single-turn runs record no conversations
            sample_index = _get_json(port, f"/api/rollout/{last_rollout_id}/summary")["rows"][0]["sample_index"]
            views = views[:1]
        for name, view in views:
            path = f"/api/rollout/{last_rollout_id}/sample/{sample_index}/{view}"
            results[name] = _time_request(port, pid, name, path, None, repeats)
        results["final_rss_mb"] = round(_rss_mb(pid))
    return results


def _time_request(port: int, pid: int, name: str, path: str, params: dict | None, repeats: int) -> dict:
    timings, status, size = [], None, 0
    for _ in range(repeats):
        seconds, status, body = timed_get(port, path, params)
        timings.append(round(seconds, 3))
        size = len(body)
    print(f"  {name:22s} {status} {timings} s  {size // 1024:,} KB  rss={_rss_mb(pid):,.0f} MB", flush=True)
    return dict(seconds=timings, status=status, kb=size // 1024)


def _sample_with_conversation(port: int, rollout_id: int) -> int | None:
    """A sample whose sidecar row exists; the real sidecar lacks a few per step."""
    rows = _get_json(port, f"/api/rollout/{rollout_id}/summary")["rows"]
    return next(
        (
            row["sample_index"]
            for row in rows[:32]
            if timed_get(port, f"/api/rollout/{rollout_id}/sample/{row['sample_index']}/messages")[1] == 200
        ),
        None,
    )


# ---------------------------------- follow ----------------------------------


class LiveTelemetryAppender:
    """Grows a finished dump the way a live collector does: every flush
    interval, the next slice of each high-rate stream lands in its current hour
    partition. Slices replay the dump's own last full hour, so rates and row
    shapes are the run's."""

    _STREAMS = (Stream.GPU_UTIL, Stream.GPU_PROCESSES, Stream.ENGINE_SERIES, Stream.TRAJECTORIES, Stream.PHASES)

    def __init__(self, dashboard_dir: Path):
        self.dashboard_dir = dashboard_dir
        self.template_rows_by_stream = {}
        for stream in self._STREAMS:
            hour_files = sorted((dashboard_dir / stream.value).glob("*.jsonl"))
            if len(hour_files) >= 2:
                self.template_rows_by_stream[stream] = _rows_by_offset_in_hour(hour_files[-2])
        last_ts = max(_row_ts(row) for _, row in self.template_rows_by_stream[Stream.GPU_UTIL][1]) + 3600
        # next full hour: appends only create new partition files, never touch the dump's own
        self.next_ts = last_ts - last_ts % 3600 + 3600
        self.appended_rows = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._append_until_stopped, daemon=True)

    def _append_until_stopped(self) -> None:
        interval = CollectorConfig.flush_interval_seconds
        while not self.stopped.wait(interval):
            hour_ts = self.next_ts - self.next_ts % 3600
            lo, hi = self.next_ts - hour_ts, self.next_ts - hour_ts + interval
            partition = time.strftime("%Y%m%d_%H", time.gmtime(self.next_ts))
            for stream, (offsets, rows) in self.template_rows_by_stream.items():
                chunk = rows[bisect.bisect_left(offsets, lo) : bisect.bisect_left(offsets, hi)]
                if chunk:
                    lines = [json.dumps(_rebase_row(row, hour_ts), separators=(",", ":")) + "\n" for _, row in chunk]
                    with open(self.dashboard_dir / stream.value / f"{partition}.jsonl", "a") as f:
                        f.write("".join(lines))
                    self.appended_rows += len(lines)
            self.next_ts += interval


def _rows_by_offset_in_hour(path: Path) -> tuple[list[float], list[tuple[float, dict]]]:
    hour_ts = calendar.timegm(time.strptime(path.stem, "%Y%m%d_%H"))
    with open(path, "rb") as f:
        rows = sorted(((_row_ts(row) - hour_ts, row) for row in map(json.loads, f)), key=lambda item: item[0])
    return [offset for offset, _ in rows], rows


def _row_ts(row: dict) -> float:
    """The ts a row is partitioned by (phases: end, or start while open)."""
    if "t1" in row:
        return row["t0"] if row["t1"] < 0 else row["t1"]
    return row["ts"]


def _rebase_row(row: dict, hour_ts: float) -> dict:
    delta = hour_ts - (_row_ts(row) - _row_ts(row) % 3600)
    if "t1" in row:
        return row | dict(t0=row["t0"] + delta, t1=row["t1"] if row["t1"] < 0 else row["t1"] + delta)
    return row | dict(ts=row["ts"] + delta)


def bench_follow(dump_details: Path, *, port: int, duration_s: float) -> dict:
    """``serve --follow`` over a growing dump, polled like the timeline's follow
    loop: meta, then loadData and loadAdvisories in parallel, then the carpet."""
    appender = LiveTelemetryAppender(dump_details / "dashboard")
    cycles = []
    with serve(dump_details, port=port, follow=True) as (pid, startup_s), ThreadPoolExecutor(max_workers=8) as pool:
        selection = _default_lane_selection(port)
        appender.thread.start()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            cycle = _time_follow_cycle(port, pool, selection)
            cycle["rss_mb"] = round(_rss_mb(pid))
            cycles.append(cycle)
            print(f"  cycle {len(cycles)}: {cycle} appended={appender.appended_rows:,}", flush=True)
            time.sleep(max(0.0, _FOLLOW_REDRAW_S - cycle["cycle_s"]))  # setInterval skips ticks while a cycle runs
        appender.stopped.set()
    cycle_seconds = sorted(cycle["cycle_s"] for cycle in cycles)
    return dict(
        startup_s=round(startup_s, 2),
        cycles=len(cycles),
        cycle_p50_s=cycle_seconds[len(cycle_seconds) // 2],
        cycle_max_s=cycle_seconds[-1],
        rss_max_mb=max(cycle["rss_mb"] for cycle in cycles),
        cycle_detail=cycles,
    )


def _time_follow_cycle(port: int, pool: ThreadPoolExecutor, selection: str) -> dict:
    seconds_by_request = {}

    def get(name: str, path: str, params: dict | None = None) -> bytes:
        seconds, _, body = timed_get(port, path, params)
        seconds_by_request[name] = round(seconds, 2)
        return body

    cycle_start = time.perf_counter()
    t1 = json.loads(get("meta", "/api/meta"))["time_range"][1]
    window = dict(t0=t1 - MetricStore.MAX_WINDOW_S, t1=t1)

    def load_data() -> None:
        get("topology", "/api/timeline/topology")
        get("bubbles", "/api/timeline/bubbles")
        lane_requests = [
            pool.submit(get, "phases", "/api/timeline/phases", window | dict(lanes=selection)),
            pool.submit(get, "gpu", "/api/timeline/gpu", window | dict(max_points=4000, lanes=selection)),
            pool.submit(get, "gpu_processes", "/api/timeline/gpu_processes", window | dict(lanes=selection)),
        ]
        for request in lane_requests:
            request.result()
        get(
            "engine_series",
            "/api/timeline/engine_series",
            window | dict(metric="sglang_num_running_reqs", max_points=4000),
        )

    for request in (pool.submit(load_data), pool.submit(get, "advisory", "/api/advisory")):
        request.result()
    get("heatmap util", "/api/timeline/heatmap", window | dict(metric="util", x_buckets=1200))
    return dict(cycle_s=round(time.perf_counter() - cycle_start, 2)) | seconds_by_request


# --------------------------------- collector ---------------------------------


def bench_collector(metrics_page: Path, *, engines: int, duration_s: float, push_rows_per_s: int) -> dict:
    """The live collector scraping ``engines`` engines that all serve
    ``metrics_page``, on the real scrape and flush cadences, while other
    producers push ``push_rows_per_s`` GPU samples in 5 s node batches: how long
    a scrape round takes, whether the pushes keep up, and the collector's CPU."""
    page = metrics_page.read_text()
    with tempfile.TemporaryDirectory(prefix="dashboard_simulator_collector_") as dashboard_dir:
        collector = DashboardCollector(
            config=CollectorConfig(dashboard_dir=dashboard_dir, run_name="bench-collector", start_ts=time.time()),
            scraper_http_get=lambda url, timeout: page,
        )
        collector._current_engine_addrs = lambda: [f"http://10.0.{i // 256}.{i % 256}:30000" for i in range(engines)]
        collector.start()
        cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        wall_start = time.monotonic()
        collector.set_router("http://router:30000")
        push_seconds = []
        # one sampler batch per node per 5 s, 128 nodes; a zero rate is the scrape-only baseline
        batch_rows = max(1, push_rows_per_s * 5 // 128) if push_rows_per_s else 0
        while time.monotonic() - wall_start < duration_s:
            tick = time.monotonic()
            for node in range(128 if batch_rows else 0):
                batch = [
                    GpuSample(ts=time.time(), node=f"n{node}", gpu=i % 8, util=50, mem_mb=1, power_w=1)
                    for i in range(batch_rows)
                ]
                started = time.perf_counter()
                collector.push_gpu_samples(f"n{node}", batch)
                push_seconds.append(time.perf_counter() - started)
            time.sleep(max(0.0, 5.0 - (time.monotonic() - tick)))
        collector.shutdown()
        wall_s = time.monotonic() - wall_start
        cpu_end = resource.getrusage(resource.RUSAGE_SELF)
        rows = sum(
            1 for path in Path(dashboard_dir, Stream.ENGINE_SERIES.value).glob("*.jsonl") for _ in open(path, "rb")
        )
    rows_per_scrape = rows / engines
    scrape_rounds = rows / max(1, _rows_per_round(page, engines))
    return dict(
        engines=engines,
        page_kb=len(page) // 1024,
        scrape_rounds=round(scrape_rounds, 1),
        achieved_scrape_interval_s=round(wall_s / max(scrape_rounds, 1e-9), 2),
        target_scrape_interval_s=CollectorConfig.scrape_interval_seconds,
        collector_cpu_cores=round(
            (cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime) / wall_s, 2
        ),
        rows_per_engine=round(rows_per_scrape, 1),
        pushed_rows_per_s=push_rows_per_s,
        push_call_p50_ms=round(sorted(push_seconds)[len(push_seconds) // 2] * 1e3, 2) if push_seconds else None,
        push_call_max_ms=round(max(push_seconds) * 1e3, 2) if push_seconds else None,
    )


def _rows_per_round(page: str, engines: int) -> int:
    rows = []
    SglangScraper(rows.append, mode=ScrapeMode.DIRECT, engine_addrs=lambda: [])._ingest("x", page, 0.0)
    return len(rows) * engines
