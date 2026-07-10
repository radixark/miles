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
from dataclasses import asdict, dataclass
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

    def timestamps(self) -> tuple[float, ...]:
        return (self.t0, self.t1)


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
        return [dict(node=node, gpu=gpu) for node, gpu in sorted(seen)]

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

    def phases_by_lane(
        self, *, t0: float | None = None, t1: float | None = None, lanes: set[tuple[str, int]] | None = None
    ) -> list[dict]:
        """Phase intervals resolved onto (node, gpu) lanes.

        Train-side events carry their own GPUs. ``rollout_manager`` events have
        none — the manager is a GPU-less driver process — so they are expanded
        onto every GPU of every rollout engine, clipped at topology-window
        boundaries (an engine restart mid-rollout splits the painted interval).
        """

        def overlaps(a0: float, a1: float) -> bool:
            return (t1 is None or a0 < t1) and (t0 is None or a1 > t0)

        out = []
        windows = self.topology_windows()
        for event in self.records[Stream.PHASES]:
            if not overlaps(event.t0, event.t1):
                continue
            if event.role == Role.ROLLOUT_MANAGER:
                for window in windows:
                    clip0 = max(event.t0, window["t0"])
                    clip1 = event.t1 if window["t1"] is None else min(event.t1, window["t1"])
                    if clip0 >= clip1:
                        continue
                    covered = {(node, gpu) for engine in window["engines"] for node, gpu in engine["gpus"]}
                    if lanes is not None:
                        covered &= lanes
                    for node, gpu in sorted(covered):
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
                    if lanes is not None and (event.node, gpu) not in lanes:
                        continue
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
                if lanes is not None and (node, gpu) not in lanes:
                    continue
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
        self,
        *,
        t0: float | None = None,
        t1: float | None = None,
        max_points: int = 2000,
        lanes: set[tuple[str, int]] | None = None,
    ) -> dict[str, dict]:
        """Per-lane NVML series, min/max-downsampled on util (the primary
        signal); mem/power take the same indices so all arrays stay aligned."""
        by_lane: dict[str, list[GpuSample]] = {}
        for sample in self.records[Stream.GPU_UTIL]:
            if lanes is not None and (sample.node, sample.gpu) not in lanes:
                continue
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

    def lane_index(self) -> list[dict]:
        """Per-lane metadata for selection resolution: train ranks observed on
        the lane, engine addrs that ever covered it, and derived roles."""
        info: dict[tuple[str, int], dict] = {
            (lane["node"], lane["gpu"]): dict(
                node=lane["node"], gpu=lane["gpu"], ranks=set(), engine_addrs=set(), roles=set()
            )
            for lane in self.lanes()
        }
        for event in self.records[Stream.PHASES]:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                entry = info.get((event.node, gpu))
                if entry is not None:
                    entry["roles"].add(Role.TRAIN.value)
                    if event.rank >= 0:
                        entry["ranks"].add(event.rank)
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    entry = info.get((node, gpu))
                    if entry is not None:
                        entry["roles"].add("rollout")
                        entry["engine_addrs"].add(engine.addr)
        return [
            dict(
                node=entry["node"],
                gpu=entry["gpu"],
                ranks=sorted(entry["ranks"]),
                engine_addrs=sorted(entry["engine_addrs"]),
                roles=sorted(entry["roles"]),
            )
            for entry in info.values()
        ]

    def resolve_lanes(self, grammar: str | None) -> set[tuple[str, int]] | None:
        """Parse the lane-selection grammar into a lane set (None = all lanes).

        Comma-separated selectors: ``rank:5`` / ``rank:0-7`` (train ranks),
        ``node:<ip>``, ``gpu:<node>:<index>``, ``engine:<addr substring>``,
        ``role:train`` / ``role:rollout``, or ``all``. Unknown selector syntax
        raises; a valid selector matching nothing selects nothing.
        """
        if grammar is None or grammar.strip() in ("", "all"):
            return None
        index = self.lane_index()
        selected: set[tuple[str, int]] = set()
        for raw_token in grammar.split(","):
            token = raw_token.strip()
            if not token:
                continue
            kind, _, value = token.partition(":")
            if not value:
                raise ValueError(f"bad lane selector {token!r}: expected kind:value")
            if kind == "rank":
                lo, _, hi = value.partition("-")
                ranks = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if ranks & set(e["ranks"])}
            elif kind == "node":
                selected |= {(e["node"], e["gpu"]) for e in index if e["node"] == value}
            elif kind == "gpu":
                node, _, gpu = value.rpartition(":")
                if not node:
                    raise ValueError(f"bad lane selector {token!r}: expected gpu:<node>:<index>")
                selected.add((node, int(gpu)))
            elif kind == "engine":
                selected |= {(e["node"], e["gpu"]) for e in index if any(value in a for a in e["engine_addrs"])}
            elif kind == "role":
                if value not in ("train", "rollout"):
                    raise ValueError(f"bad lane selector {token!r}: role must be train or rollout")
                selected |= {(e["node"], e["gpu"]) for e in index if value in e["roles"]}
            else:
                raise ValueError(f"unknown lane selector kind {kind!r} in {token!r}")
        return selected

    HEATMAP_MAGNITUDE_FIELDS: ClassVar[tuple[str, ...]] = ("util", "mem_mb", "power_w")

    def heatmap(
        self,
        metric: str,
        *,
        t0: float | None = None,
        t1: float | None = None,
        x_buckets: int = 1200,
        lanes: set[tuple[str, int]] | None = None,
    ) -> dict:
        """Rank-carpet matrix: one uint8 cell per (lane, time bucket).

        Magnitude metrics take the bucket MAX (activity must not average away);
        ``phase`` paints the covering phase id with non-idle winning over idle.
        Returns rows metadata + the raw bytes; the server frames it as binary.
        """
        assert x_buckets >= 2, f"{x_buckets=}"
        window = self.time_range()
        if window is None:
            return dict(
                rows=[], x_buckets=x_buckets, t0=0.0, t1=0.0, metric=metric, values=b"", scale=None, palette=None
            )
        t0 = window[0] if t0 is None else t0
        t1 = window[1] if t1 is None else t1
        assert t1 > t0, f"empty heatmap window [{t0}, {t1})"
        rows = [
            dict(node=entry["node"], gpu=entry["gpu"], roles=entry["roles"])
            for entry in self.lane_index()
            if lanes is None or (entry["node"], entry["gpu"]) in lanes
        ]
        row_of = {(lane["node"], lane["gpu"]): i for i, lane in enumerate(rows)}
        matrix = np.zeros((len(rows), x_buckets), dtype=np.uint8)
        span = t1 - t0

        def bucket(ts: float) -> int:
            return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

        if metric in self.HEATMAP_MAGNITUDE_FIELDS:
            raw: dict[tuple[int, int], float] = {}
            for sample in self.records[Stream.GPU_UTIL]:
                row = row_of.get((sample.node, sample.gpu))
                if row is None or sample.ts < t0 or sample.ts > t1:
                    continue
                key = (row, bucket(sample.ts))
                value = float(getattr(sample, metric))
                if value > raw.get(key, float("-inf")):
                    raw[key] = value
            peak = max(raw.values(), default=1.0)
            scale = dict(max=100.0 if metric == "util" else peak)
            for (row, col), value in raw.items():
                matrix[row, col] = min(255, int(value / scale["max"] * 255))
            return dict(
                rows=rows,
                x_buckets=x_buckets,
                t0=t0,
                t1=t1,
                metric=metric,
                values=matrix.tobytes(),
                scale=scale,
                palette=None,
            )

        if metric != "phase":
            raise ValueError(f"unknown heatmap metric {metric!r}")
        idle = {INITIALIZE_PHASE, "train_wait", "sleep"}
        events = self.phases_by_lane(t0=t0, t1=t1, lanes=lanes)
        names = sorted({e["name"] for e in events})
        palette = [""] + names  # id 0 = no phase observed
        phase_id = {name: i + 1 for i, name in enumerate(names)}
        # idle painted first so any active phase in the same bucket wins
        for event in sorted(events, key=lambda e: e["name"] not in idle):
            row = row_of.get((event["node"], event["gpu"]))
            if row is None:
                continue
            b0 = bucket(max(event["t0"], t0))
            b1 = bucket(min(event["t1"], t1))
            matrix[row, b0 : b1 + 1] = phase_id[event["name"]]
        return dict(
            rows=rows,
            x_buckets=x_buckets,
            t0=t0,
            t1=t1,
            metric=metric,
            values=matrix.tobytes(),
            scale=None,
            palette=palette,
        )

    def outliers(
        self, criterion: str, *, t0: float | None = None, t1: float | None = None, top_k: int = 16
    ) -> list[dict]:
        """Candidate lanes for the quick-pick buttons; the user confirms the
        selection — the machine only proposes."""
        if criterion == "lowest_util":
            totals: dict[tuple[str, int], list[float]] = {}
            for sample in self.records[Stream.GPU_UTIL]:
                if (t0 is not None and sample.ts < t0) or (t1 is not None and sample.ts > t1):
                    continue
                totals.setdefault((sample.node, sample.gpu), []).append(float(sample.util))
            scored = [(sum(v) / len(v), node, gpu) for (node, gpu), v in totals.items()]
            scored.sort()
        elif criterion.startswith("slowest_phase:"):
            phase_name = criterion.removeprefix("slowest_phase:")
            durations: dict[tuple[str, int], list[float]] = {}
            for event in self.phases_by_lane(t0=t0, t1=t1):
                if event["name"] != phase_name:
                    continue
                durations.setdefault((event["node"], event["gpu"]), []).append(event["t1"] - event["t0"])
            scored = [(sum(v) / len(v), node, gpu) for (node, gpu), v in durations.items()]
            scored.sort(reverse=True)
        else:
            raise ValueError(f"unknown outlier criterion {criterion!r}")
        return [dict(node=node, gpu=gpu, score=score) for score, node, gpu in scored[:top_k]]

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
