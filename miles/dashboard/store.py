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


@dataclass
class PhaseEvent(Record):
    """One completed timer interval on one process (Timer sink event)."""

    stream: ClassVar[Stream] = Stream.PHASES
    name: str
    t0: float
    t1: float
    node: str
    gpus: list[int]
    rank: int
    role: str

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
