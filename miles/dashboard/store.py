"""File-backed store for miles dashboard time series.

The live collector buffers records via ``append()`` and persists them with
``flush()`` as append-only JSONL streams under ``{dump_details}/dashboard/``.
The dashboard server reads the same directory with ``MetricStore.load()`` and
picks up appended lines incrementally with ``follow()`` — a plain byte-offset
tail, correct because every stream is append-only.

Each stream holds one record type (one JSON object per line); see the
``Record`` subclasses for the schemas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import numpy as np


class Stream(StrEnum):
    METRICS = "metrics"
    PHASES = "phases"
    TOPOLOGY = "topology"
    GPU_UTIL = "gpu_util"
    ENGINE_SERIES = "engine_series"

    @property
    def filename(self) -> str:
        return f"{self.value}.jsonl"


@dataclass
class Record:
    stream: ClassVar[Stream]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        return cls(**data)

    def timestamps(self) -> tuple[float, ...]:
        return (self.ts,)


@dataclass
class MetricsRecord(Record):
    """One ``tracking_utils.log()`` payload."""

    stream: ClassVar[Stream] = Stream.METRICS
    ts: float
    step_key: str | None
    step: int | None
    metrics: dict[str, Any]


class Role(StrEnum):
    TRAIN = "train"
    ROLLOUT_MANAGER = "rollout_manager"
    DERIVED = "derived"  # synthesized by the read side, not observed


# phase name synthesized per lane for [meta.start_ts, first observed event)
INITIALIZE_PHASE = "initialize"


@dataclass
class PhaseEvent(Record):
    """One completed timer interval on one process (Timer sink event).

    ``rollout_manager`` events carry no GPUs (the manager is a driver-side
    process); the timeline queries expand them onto rollout-engine lanes."""

    stream: ClassVar[Stream] = Stream.PHASES
    name: str
    t0: float
    t1: float
    node: str
    gpus: list[int]
    rank: int
    role: str  # a Role value

    # t1 sentinel: the interval was still running when this event was written
    # (Timer.start emits it so long phases are visible before they end); the
    # closing event with the real t1 supersedes it on the read side
    OPEN_T1: ClassVar[float] = -1.0

    @property
    def open(self) -> bool:
        return self.t1 == self.OPEN_T1

    def timestamps(self) -> tuple[float, ...]:
        return (self.t0,) if self.open else (self.t0, self.t1)


@dataclass
class EngineInfo:
    addr: str
    worker_type: str
    engine_rank: int
    gpus: list[list]  # [node_ip, gpu_id] pairs
    gpu_uuids: list[str | None]


@dataclass
class TopologySnapshot(Record):
    """Full engine topology; snapshot N is valid until snapshot N+1."""

    stream: ClassVar[Stream] = Stream.TOPOLOGY
    ts: float
    engines: list[EngineInfo]

    @classmethod
    def from_dict(cls, data: dict) -> TopologySnapshot:
        return cls(ts=data["ts"], engines=[EngineInfo(**e) for e in data["engines"]])


@dataclass
class GpuSample(Record):
    """One NVML sample of one GPU."""

    stream: ClassVar[Stream] = Stream.GPU_UTIL
    ts: float
    node: str
    gpu: int
    util: int
    mem_mb: int
    power_w: int


@dataclass
class EngineSample(Record):
    """One scraped sglang engine metric value."""

    stream: ClassVar[Stream] = Stream.ENGINE_SERIES
    ts: float
    addr: str
    metric: str
    labels: dict[str, str]
    value: float


_RECORD_TYPE_OF_STREAM: dict[Stream, type[Record]] = {
    cls.stream: cls for cls in (MetricsRecord, PhaseEvent, TopologySnapshot, GpuSample, EngineSample)
}


@dataclass
class Meta:
    run_name: str
    start_ts: float
    args: dict[str, Any]
    schema_version: int = 1

    FILENAME: ClassVar[str] = "meta.json"


class MetricStore:
    def __init__(self, dir: Path | str):
        self.dir = Path(dir)
        self.meta: Meta | None = None
        self.records: dict[Stream, list[Record]] = {s: [] for s in Stream}
        self._buffers: dict[Stream, list[Record]] = {s: [] for s in Stream}
        self._offsets: dict[Stream, int] = {s: 0 for s in Stream}

    # ------------------------------ write side ------------------------------

    def append(self, record: Record) -> None:
        self._buffers[record.stream].append(record)

    def buffered_count(self, stream: Stream) -> int:
        return len(self._buffers[stream])

    def drop_oldest_buffered(self, stream: Stream, *, keep_ratio: float = 0.9) -> int:
        """Drop the oldest buffered (not yet flushed) records of one stream and
        return how many were dropped. Lets the collector bound memory when the
        disk stays unwritable instead of growing without limit."""
        buffer = self._buffers[stream]
        dropped = max(1, int(len(buffer) * (1 - keep_ratio)))
        del buffer[:dropped]
        return dropped

    def write_meta(self, meta: Meta) -> None:
        self.meta = meta
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / Meta.FILENAME).write_text(json.dumps(asdict(meta), indent=1))

    def flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for stream, buffer in self._buffers.items():
            if not buffer:
                continue
            with open(self.dir / stream.filename, "a") as f:
                f.write("".join(json.dumps(record.to_dict(), separators=(",", ":")) + "\n" for record in buffer))
            buffer.clear()

    # ------------------------------ read side -------------------------------

    @classmethod
    def load(cls, dir: Path | str) -> MetricStore:
        store = cls(dir)
        meta_path = store.dir / Meta.FILENAME
        if meta_path.exists():
            store.meta = Meta(**json.loads(meta_path.read_text()))
        store.follow()
        return store

    def follow(self) -> int:
        """Read records appended since the last load()/follow(). Returns the count.

        Only complete lines (terminated by a newline) are consumed, so a
        crash-interrupted partial write at the tail is left for a later call
        once its writer completes it. A malformed *complete* line is real
        corruption and raises.
        """
        num_new = 0
        for stream in Stream:
            path = self.dir / stream.filename
            if not path.exists():
                continue
            with open(path, "rb") as f:
                f.seek(self._offsets[stream])
                chunk = f.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                continue
            complete = chunk[: end + 1]
            record_type = _RECORD_TYPE_OF_STREAM[stream]
            for line in complete.splitlines():
                try:
                    self.records[stream].append(record_type.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    raise ValueError(
                        f"corrupt record in {path} near byte {self._offsets[stream]}: {line[:200]!r}"
                    ) from e
                num_new += 1
            self._offsets[stream] += len(complete)
        return num_new

    # ------------------------------- queries --------------------------------

    def metric_keys(self) -> list[str]:
        keys: set[str] = set()
        for record in self.records[Stream.METRICS]:
            keys.update(record.metrics.keys())
        return sorted(keys)

    def step_keys(self) -> list[str]:
        return sorted({r.step_key for r in self.records[Stream.METRICS] if r.step_key is not None})

    def metric_series(
        self, keys: list[str], *, x_key: str, t0: float | None = None, t1: float | None = None
    ) -> dict[str, dict[str, list]]:
        """Series for each key, restricted to records logged against ``x_key``
        (e.g. "rollout/step") so values from different step axes never mix."""
        out: dict[str, dict[str, list]] = {k: {"x": [], "y": [], "ts": []} for k in keys}
        for record in self.records[Stream.METRICS]:
            if record.step_key != x_key:
                continue
            if (t0 is not None and record.ts < t0) or (t1 is not None and record.ts > t1):
                continue
            for k in keys:
                if k in record.metrics:
                    series = out[k]
                    series["x"].append(record.step)
                    series["y"].append(record.metrics[k])
                    series["ts"].append(record.ts)
        return out

    def time_range(self) -> tuple[float, float] | None:
        stamps = [ts for records in self.records.values() for record in records for ts in record.timestamps()]
        if not stamps:
            return None
        return min(stamps), max(stamps)

    # --------------------------- timeline queries ---------------------------

    def lanes(self) -> list[dict]:
        """Every (node, gpu) seen in any stream — one timeline lane each."""
        seen: set[tuple[str, int]] = set()
        for sample in self.records[Stream.GPU_UTIL]:
            seen.add((sample.node, sample.gpu))
        for event in self.records[Stream.PHASES]:
            for gpu in event.gpus:
                seen.add((event.node, gpu))
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    seen.add((node, gpu))
        # index = position in the deterministic (node, gpu) sort: the cluster-
        # global lane number every view labels lanes with
        return [dict(node=node, gpu=gpu, index=i) for i, (node, gpu) in enumerate(sorted(seen))]

    def topology_windows(self) -> list[dict]:
        """Engine topology snapshots with validity windows: snapshot N is valid
        [ts_N, ts_{N+1}); the last one is open-ended (t1=None)."""
        snapshots = sorted(self.records[Stream.TOPOLOGY], key=lambda s: s.ts)
        return [
            dict(
                t0=snapshot.ts,
                t1=snapshots[i + 1].ts if i + 1 < len(snapshots) else None,
                engines=[asdict(engine) for engine in snapshot.engines],
            )
            for i, snapshot in enumerate(snapshots)
        ]

    def phases_by_lane(self, *, t0: float | None = None, t1: float | None = None) -> list[dict]:
        """Phase intervals resolved onto (node, gpu) lanes.

        Train-side events carry their own GPUs. ``rollout_manager`` events have
        none — the manager is a GPU-less driver process — so they are expanded
        onto every GPU of every rollout engine, clipped at topology-window
        boundaries (an engine restart mid-rollout splits the painted interval).
        """

        def overlaps(a0: float, a1: float) -> bool:
            return (t1 is None or a0 < t1) and (t0 is None or a1 > t0)

        # resolve OPEN intervals (Timer.start markers): a closing event with
        # the real t1 supersedes its open twin; a still-open one is clipped to
        # the newest data timestamp so it renders as a growing in-progress band
        events = list(self.records[Stream.PHASES])
        closed = {(e.node, e.rank, e.name, e.t0) for e in events if not e.open}
        edge = (self.time_range() or (0.0, 0.0))[1]
        events = [
            replace(e, t1=max(edge, e.t0)) if e.open else e
            for e in events
            if not (e.open and (e.node, e.rank, e.name, e.t0) in closed)
        ]

        out = []
        windows = self.topology_windows()
        for event in events:
            if not overlaps(event.t0, event.t1):
                continue
            if event.role == Role.ROLLOUT_MANAGER:
                for window in windows:
                    clip0 = max(event.t0, window["t0"])
                    clip1 = event.t1 if window["t1"] is None else min(event.t1, window["t1"])
                    if clip0 >= clip1:
                        continue
                    lanes = {(node, gpu) for engine in window["engines"] for node, gpu in engine["gpus"]}
                    for node, gpu in sorted(lanes):
                        out.append(
                            dict(
                                name=event.name,
                                t0=clip0,
                                t1=clip1,
                                node=node,
                                gpu=gpu,
                                rank=event.rank,
                                role=event.role,
                            )
                        )
            else:
                for gpu in event.gpus:
                    out.append(
                        dict(
                            name=event.name,
                            t0=event.t0,
                            t1=event.t1,
                            node=event.node,
                            gpu=gpu,
                            rank=event.rank,
                            role=event.role,
                        )
                    )
        # synthesize the initialize band: from collector start (meta.start_ts)
        # to each lane's first observed event — model loading / engine startup
        # has GPU util but no Timer instrumentation
        if self.meta is not None:
            first_event: dict[tuple[str, int], float] = {}
            for event in out:
                key = (event["node"], event["gpu"])
                first_event[key] = min(first_event.get(key, float("inf")), event["t0"])
            for (node, gpu), first_t0 in first_event.items():
                if first_t0 > self.meta.start_ts and overlaps(self.meta.start_ts, first_t0):
                    out.append(
                        dict(
                            name=INITIALIZE_PHASE,
                            t0=self.meta.start_ts,
                            t1=first_t0,
                            node=node,
                            gpu=gpu,
                            rank=-1,
                            role=Role.DERIVED,
                        )
                    )
        out.sort(key=lambda e: (e["node"], e["gpu"], e["t0"]))
        return out

    def gpu_series(
        self, *, t0: float | None = None, t1: float | None = None, max_points: int = 2000
    ) -> dict[str, dict]:
        """Per-lane NVML series, min/max-downsampled on util (the primary
        signal); mem/power take the same indices so all arrays stay aligned."""
        by_lane: dict[str, list[GpuSample]] = {}
        for sample in self.records[Stream.GPU_UTIL]:
            if (t0 is not None and sample.ts < t0) or (t1 is not None and sample.ts > t1):
                continue
            by_lane.setdefault(f"{sample.node}:{sample.gpu}", []).append(sample)
        out = {}
        for lane, samples in sorted(by_lane.items()):
            samples.sort(key=lambda s: s.ts)
            util = np.asarray([s.util for s in samples])
            indices, _ = minmax_downsample(np.arange(len(samples)), util, max_points)
            out[lane] = dict(
                ts=[samples[i].ts for i in indices],
                util=[samples[i].util for i in indices],
                mem_mb=[samples[i].mem_mb for i in indices],
                power_w=[samples[i].power_w for i in indices],
            )
        return out

    def engine_metric_names(self) -> list[str]:
        """Distinct scraped engine metrics — the L0 sglang category catalog."""
        return sorted({record.metric for record in self.records[Stream.ENGINE_SERIES]})

    def engine_series(
        self, metric: str, *, t0: float | None = None, t1: float | None = None, max_points: int = 2000
    ) -> list[dict]:
        """One series per (engine addr, label set) for the given metric."""
        grouped: dict[tuple, list[EngineSample]] = {}
        for sample in self.records[Stream.ENGINE_SERIES]:
            if sample.metric != metric:
                continue
            if (t0 is not None and sample.ts < t0) or (t1 is not None and sample.ts > t1):
                continue
            grouped.setdefault((sample.addr, tuple(sorted(sample.labels.items()))), []).append(sample)
        out = []
        for (addr, labels), samples in sorted(grouped.items()):
            samples.sort(key=lambda s: s.ts)
            ts, values = stride_downsample([s.ts for s in samples], [s.value for s in samples], max_points)
            out.append(dict(addr=addr, labels=dict(labels), ts=ts.tolist(), value=values.tolist()))
        return out

    def bubbles(self) -> list[dict]:
        """Per-step bubble summary strip: step time and wait ratio from the
        perf metrics, with the wall-clock ts as the timeline zoom anchor."""
        out = []
        for record in self.records[Stream.METRICS]:
            if record.step_key != "rollout/step":
                continue
            step_time = record.metrics.get("perf/step_time")
            wait_ratio = record.metrics.get("perf/wait_time_ratio")
            if step_time is None and wait_ratio is None:
                continue
            out.append(dict(step=record.step, ts=record.ts, step_time=step_time, wait_ratio=wait_ratio))
        return out


# ------------------------------ downsampling --------------------------------


def stride_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_points).astype(np.int64)
    return xs[idx], ys[idx]


def minmax_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucketed downsampling keeping each bucket's min and max, so spikes and
    dips survive. Output length is at most ``max_points``."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    num_buckets = max_points // 2
    edges = np.linspace(0, len(xs), num_buckets + 1).astype(np.int64)
    keep: set[int] = set()
    for b in range(num_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if lo >= hi:
            continue
        segment = ys[lo:hi]
        keep.add(lo + int(np.argmin(segment)))
        keep.add(lo + int(np.argmax(segment)))
    idx = sorted(keep)
    return xs[idx], ys[idx]
