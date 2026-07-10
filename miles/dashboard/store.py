"""File-backed store for miles dashboard time series.

The live collector buffers records via ``append()`` and persists them with
``flush()`` as append-only JSONL streams under ``{dump_details}/dashboard/``.
The dashboard server reads the same directory with ``MetricStore.load()`` and
picks up appended lines incrementally with ``follow()`` — a plain byte-offset
tail, correct because every stream is append-only.

Each stream holds one record type (one JSON object per line); see the
``Record`` subclasses for the schemas.

The two high-rate streams (``gpu_util``, ``engine_series``) are held in memory
as columnar polars frames instead of dataclass lists (design doc §17):
~16 B/row instead of ~600 B, vectorized parsing and numpy queries.

High-rate streams (those two plus ``phases``) write hourly partition files
``{stream}/{YYYYMMDD_HH}.jsonl`` (phases keyed by END hour) and parse lazily:
open reads nothing from them; a windowed query parses only the hours it
touches, through an LRU block cache with per-file byte-offset tail refresh.
A v1 flat ``{stream}.jsonl`` still reads as one "legacy" partition. Lane
metadata for selection resolution is folded into ``lane_catalog.json`` at
flush time so it never needs a stream scan.
"""

from __future__ import annotations

import calendar
import io
import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import numpy as np
import polars as pl


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


def _hour_key(ts: float) -> str:
    return time.strftime("%Y%m%d_%H", time.gmtime(ts))


def _hour_bounds(key: str) -> tuple[float, float]:
    start = calendar.timegm(time.strptime(key, "%Y%m%d_%H"))
    return float(start), float(start + 3600)


class _PartitionReader:
    """One hourly-partitioned JSONL stream on the read side.

    Files parse lazily at first windowed read into an LRU block cache; a
    cached block picks up appended bytes through a per-file byte offset with
    the same complete-line rule as ``MetricStore.follow``. A v1 flat
    ``{stream}.jsonl`` participates as one "legacy" partition spanning all
    time, so every pre-partition dump dir keeps working.
    """

    LEGACY_KEY: ClassVar[str] = "legacy"

    def __init__(
        self,
        dir: Path,
        legacy_path: Path,
        *,
        parse: Callable[[bytes, Path, int], Any],
        concat: Callable[[list], Any],
        length: Callable[[Any], int],
        line_stamps: Callable[[dict], tuple],
        max_blocks: int,
    ):
        self.dir = dir
        self.legacy_path = legacy_path
        self._parse = parse
        self._concat = concat
        self._length = length
        self._line_stamps = line_stamps
        self._max_blocks = max_blocks
        self._blocks: OrderedDict[str, Any] = OrderedDict()
        self._offsets: dict[str, int] = {}

    def files(self) -> list[tuple[str, Path]]:
        out = [(self.LEGACY_KEY, self.legacy_path)] if self.legacy_path.exists() else []
        if self.dir.is_dir():
            out.extend(sorted((path.stem, path) for path in self.dir.glob("*.jsonl")))
        return out

    def has_data(self) -> bool:
        return any(path.stat().st_size > 0 for _, path in self.files())

    def window(self, t0: float | None, t1: float | None) -> Any:
        """Concatenated block over the partitions intersecting [t0, t1]."""
        blocks = []
        for key, path in self.files():
            if key != self.LEGACY_KEY:
                lo, hi = _hour_bounds(key)
                if (t0 is not None and hi <= t0) or (t1 is not None and lo > t1):
                    continue
            blocks.append(self._block(key, path))
        return self._concat([block for block in blocks if self._length(block)])

    def refresh_cached(self) -> int:
        """Tail-refresh every cached block; returns rows appended. Partitions
        never read remain lazy — they parse fully at their first read."""
        paths = dict(self.files())
        added = 0
        for key in list(self._blocks):
            if key in paths:
                before = self._length(self._blocks[key])
                self._block(key, paths[key])
                added += self._length(self._blocks[key]) - before
        return added

    def _block(self, key: str, path: Path) -> Any:
        offset = self._offsets.get(key, 0)
        if key in self._blocks and path.stat().st_size <= offset:
            self._blocks.move_to_end(key)
            return self._blocks[key]
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end >= 0:
            new = self._parse(chunk[: end + 1], path, offset)
            pieces = [self._blocks[key], new] if key in self._blocks else [new]
            self._blocks[key] = self._concat([piece for piece in pieces if self._length(piece)])
            self._offsets[key] = offset + end + 1
        elif key not in self._blocks:
            self._blocks[key] = self._concat([])
            self._offsets[key] = offset
        self._blocks.move_to_end(key)
        while len(self._blocks) > self._max_blocks:
            evicted, _ = self._blocks.popitem(last=False)
            self._offsets.pop(evicted, None)
        return self._blocks[key]

    def edge_stamps(self) -> list[float]:
        """Global-range candidates from the first line of the oldest file and
        the last complete line of the newest — no full scan."""
        files = [(key, path) for key, path in self.files() if path.stat().st_size > 0]
        if not files:
            return []
        lines = []
        with open(files[0][1], "rb") as f:
            lines.append(f.readline())
        with open(files[-1][1], "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read()
        complete = tail[: tail.rfind(b"\n") + 1]
        if complete:
            lines.append(complete.splitlines()[-1])
        stamps: list[float] = []
        for line in lines:
            try:
                stamps.extend(self._line_stamps(json.loads(line)))
            except json.JSONDecodeError:
                pass  # partial line mid-append: no complete record to stamp yet
        return stamps


class MetricStore:
    COLUMNAR_STREAMS: ClassVar[tuple[Stream, ...]] = (Stream.GPU_UTIL, Stream.ENGINE_SERIES)
    PARTITIONED_STREAMS: ClassVar[tuple[Stream, ...]] = (Stream.GPU_UTIL, Stream.ENGINE_SERIES, Stream.PHASES)
    MAX_WINDOW_S: ClassVar[float] = 4 * 3600.0
    PARTITION_CACHE_BLOCKS: ClassVar[int] = 24
    CATALOG_FILENAME: ClassVar[str] = "lane_catalog.json"
    FRAME_SCHEMAS: ClassVar[dict[Stream, dict[str, Any]]] = {
        Stream.GPU_UTIL: dict(
            ts=pl.Float64, node=pl.String, gpu=pl.Int64, util=pl.Int64, mem_mb=pl.Int64, power_w=pl.Int64
        ),
        # labels dicts are canonicalized to a JSON string column at parse time
        Stream.ENGINE_SERIES: dict(
            ts=pl.Float64, addr=pl.String, metric=pl.String, labels_json=pl.String, value=pl.Float64
        ),
    }

    def __init__(self, dir: Path | str):
        self.dir = Path(dir)
        self.meta: Meta | None = None
        self.records: dict[Stream, list[Record]] = {s: [] for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._buffers: dict[Stream, list[Record]] = {s: [] for s in Stream}
        self._offsets: dict[Stream, int] = {s: 0 for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._readers: dict[Stream, _PartitionReader] = {s: self._make_reader(s) for s in self.PARTITIONED_STREAMS}
        self._catalog: dict[str, dict] | None = None  # write-side lane catalog accumulator

    def _make_reader(self, stream: Stream) -> _PartitionReader:
        if stream in self.COLUMNAR_STREAMS:
            empty = pl.DataFrame(schema=self.FRAME_SCHEMAS[stream])

            def concat_frames(blocks: list, empty: pl.DataFrame = empty) -> pl.DataFrame:
                if not blocks:
                    return empty
                merged = blocks[0] if len(blocks) == 1 else pl.concat(blocks, rechunk=False)
                # bound chunk fragmentation from long tail sessions
                return merged.rechunk() if merged.n_chunks() > 256 else merged

            return _PartitionReader(
                self.dir / stream.value,
                self.dir / stream.filename,
                parse=lambda raw, path, offset, stream=stream: self._parse_columnar(stream, raw, path, offset),
                concat=concat_frames,
                length=lambda frame: frame.height,
                line_stamps=lambda data: (data["ts"],),
                max_blocks=self.PARTITION_CACHE_BLOCKS,
            )
        assert stream is Stream.PHASES, stream
        return _PartitionReader(
            self.dir / stream.value,
            self.dir / stream.filename,
            parse=self._parse_phases,
            concat=lambda blocks: [event for block in blocks for event in block],
            length=len,
            # PhaseEvent.timestamps() skips the open-interval t1 sentinel
            line_stamps=lambda data: PhaseEvent.from_dict(data).timestamps(),
            max_blocks=self.PARTITION_CACHE_BLOCKS,
        )

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
        self._merge_catalog()
        for stream, buffer in self._buffers.items():
            if not buffer:
                continue
            if stream in self.PARTITIONED_STREAMS:
                subdir = self.dir / stream.value
                subdir.mkdir(exist_ok=True)
                groups: dict[str, list[Record]] = {}
                for record in buffer:
                    # phases key by END hour (the completion append is what
                    # lands on disk, keeping the window lower bound exact);
                    # OPEN markers have no end yet and key by their start
                    if stream is Stream.PHASES:
                        ts = record.t0 if record.open else record.t1
                    else:
                        ts = record.ts
                    groups.setdefault(_hour_key(ts), []).append(record)
                for key, group in sorted(groups.items()):
                    with open(subdir / f"{key}.jsonl", "a") as f:
                        f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in group))
            else:
                with open(self.dir / stream.filename, "a") as f:
                    f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in buffer))
            buffer.clear()

    def _merge_catalog(self) -> None:
        """Fold buffered records into ``lane_catalog.json``: per-lane train
        ranks, engine addrs and roles, so selection resolution and heatmap
        rows never need a stream scan. Atomic rewrite, only when changed."""
        if self._catalog is None:
            path = self.dir / self.CATALOG_FILENAME
            self._catalog = json.loads(path.read_text())["lanes"] if path.exists() else {}
        lanes = self._catalog
        dirty = False

        def entry(node: str, gpu: int) -> dict:
            nonlocal dirty
            key = f"{node}:{gpu}"
            if key not in lanes:
                lanes[key] = dict(ranks=[], engine_addrs=[], roles=[])
                dirty = True
            return lanes[key]

        def add(values: list, value) -> None:
            nonlocal dirty
            if value not in values:
                values.append(value)
                dirty = True

        for event in self._buffers[Stream.PHASES]:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                lane = entry(event.node, gpu)
                add(lane["roles"], Role.TRAIN.value)
                if event.rank >= 0:
                    add(lane["ranks"], event.rank)
        for snapshot in self._buffers[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    lane = entry(node, gpu)
                    add(lane["roles"], "rollout")
                    add(lane["engine_addrs"], engine.addr)
        for sample in self._buffers[Stream.GPU_UTIL]:
            entry(sample.node, sample.gpu)
        if dirty:
            for lane in lanes.values():
                lane["ranks"].sort()
                lane["engine_addrs"].sort()
                lane["roles"].sort()
            tmp = self.dir / (self.CATALOG_FILENAME + ".tmp")
            tmp.write_text(json.dumps(dict(version=1, lanes=lanes)))
            os.replace(tmp, self.dir / self.CATALOG_FILENAME)

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
        corruption and raises — for hour-partitioned streams that happens at
        the first windowed read of the bad partition, since those parse
        lazily; here only already-cached partition blocks refresh.
        """
        num_new = 0
        for stream in self.records:
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
        for reader in self._readers.values():
            num_new += reader.refresh_cached()
        return num_new

    def _parse_columnar(self, stream: Stream, raw: bytes, path: Path, offset: int) -> pl.DataFrame:
        schema = self.FRAME_SCHEMAS[stream]
        try:
            frame = pl.read_ndjson(io.BytesIO(raw))
            if stream is Stream.ENGINE_SERIES:
                # all-empty labels: read_ndjson drops the column entirely
                dtype = frame.schema.get("labels")
                empty = dtype is None or dtype == pl.Null or (isinstance(dtype, pl.Struct) and not dtype.fields)
                # struct schemas unify per chunk (same-writer key order is stable,
                # so equal label dicts encode to equal strings within a store)
                encoded = pl.lit("{}") if empty else pl.col("labels").struct.json_encode()
                frame = frame.with_columns(encoded.alias("labels_json")).drop("labels", strict=False)
            if set(frame.columns) != set(schema):
                raise ValueError(f"fields {sorted(frame.columns)} != schema {sorted(schema)}")
            return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
        except (pl.exceptions.PolarsError, ValueError) as e:
            raise ValueError(f"corrupt record in {path} near byte {offset}: {e}") from e

    def _parse_phases(self, raw: bytes, path: Path, offset: int) -> list[PhaseEvent]:
        events = []
        for line in raw.splitlines():
            try:
                events.append(PhaseEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                raise ValueError(f"corrupt record in {path} near byte {offset}: {line[:200]!r}") from e
        return events

    def _phase_events(self, t0: float | None, t1: float | None) -> list[PhaseEvent]:
        # closed phases partition by END hour (lower bound exact, slack one
        # max phase duration FORWARD — design §17); OPEN markers partition by
        # their START hour, so the lower bound gets the same slack BACKWARD
        lower = None if t0 is None else t0 - self.MAX_WINDOW_S
        upper = None if t1 is None else t1 + self.MAX_WINDOW_S
        return self._readers[Stream.PHASES].window(lower, upper)

    def has_stream(self, stream: Stream) -> bool:
        if stream in self.PARTITIONED_STREAMS:
            return self._readers[stream].has_data()
        return bool(self.records[stream])

    def iter_records(self, stream: Stream) -> list[Record]:
        """Records as dataclass objects — slow materialization for tests/tools."""
        if stream is Stream.GPU_UTIL:
            frame = self._readers[stream].window(None, None)
            return [GpuSample(**row) for row in frame.iter_rows(named=True)]
        if stream is Stream.ENGINE_SERIES:
            return [
                EngineSample(
                    ts=row["ts"],
                    addr=row["addr"],
                    metric=row["metric"],
                    labels={k: v for k, v in json.loads(row["labels_json"]).items() if v is not None},
                    value=row["value"],
                )
                for row in self._readers[stream].window(None, None).iter_rows(named=True)
            ]
        if stream is Stream.PHASES:
            return list(self._phase_events(None, None))
        return list(self.records[stream])

    @staticmethod
    def _window(frame: pl.DataFrame, t0: float | None, t1: float | None) -> pl.DataFrame:
        if t0 is not None:
            frame = frame.filter(pl.col("ts") >= t0)
        if t1 is not None:
            frame = frame.filter(pl.col("ts") <= t1)
        return frame

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
        for reader in self._readers.values():
            stamps.extend(reader.edge_stamps())
        if not stamps:
            return None
        return min(stamps), max(stamps)

    # --------------------------- timeline queries ---------------------------

    def lanes(self) -> list[dict]:
        """Every (node, gpu) seen in any stream — one timeline lane each."""
        return [dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"]) for entry in self.lane_index()]

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

        # resolve OPEN intervals (Timer.start markers): a closing event with
        # the real t1 supersedes its open twin; a still-open one is clipped to
        # the newest data timestamp so it renders as a growing in-progress band
        events = list(self._phase_events(t0, t1))
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
        # has GPU util but no Timer instrumentation. Only when the window
        # covers the run start: a later window's first event is not the
        # lane's first event ever
        if self.meta is not None and (t0 is None or t0 <= self.meta.start_ts):
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
        frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
        out = {}
        for (node, gpu), part in sorted(frame.partition_by(["node", "gpu"], as_dict=True).items()):
            if lanes is not None and (node, gpu) not in lanes:
                continue
            part = part.sort("ts")
            util = part["util"].to_numpy()
            indices, _ = minmax_downsample(np.arange(part.height), util, max_points)
            out[f"{node}:{gpu}"] = dict(
                ts=part["ts"].to_numpy()[indices].tolist(),
                util=util[indices].tolist(),
                mem_mb=part["mem_mb"].to_numpy()[indices].tolist(),
                power_w=part["power_w"].to_numpy()[indices].tolist(),
            )
        return out

    def engine_metric_names(self) -> list[str]:
        """Distinct scraped engine metrics — the L0 sglang category catalog."""
        return sorted(self._readers[Stream.ENGINE_SERIES].window(None, None).get_column("metric").unique())

    def engine_series(
        self, metric: str, *, t0: float | None = None, t1: float | None = None, max_points: int = 2000
    ) -> list[dict]:
        """One series per (engine addr, label set) for the given metric."""
        frame = self._window(self._readers[Stream.ENGINE_SERIES].window(t0, t1), t0, t1).filter(
            pl.col("metric") == metric
        )
        out = []
        for (addr, labels_json), part in sorted(frame.partition_by(["addr", "labels_json"], as_dict=True).items()):
            part = part.sort("ts")
            ts, values = stride_downsample(part["ts"].to_numpy(), part["value"].to_numpy(), max_points)
            labels = {k: v for k, v in json.loads(labels_json).items() if v is not None}
            out.append(dict(addr=addr, labels=labels, ts=ts.tolist(), value=values.tolist()))
        return out

    def lane_index(self) -> list[dict]:
        """Per-lane metadata for selection resolution: train ranks observed on
        the lane, engine addrs that ever covered it, and derived roles.

        Reads ``lane_catalog.json`` (merge-written at flush time) when
        present; a legacy dir without one falls back to a full-stream scan.
        """
        path = self.dir / self.CATALOG_FILENAME
        if path.exists():
            lanes = json.loads(path.read_text())["lanes"]
            out = []
            for key, entry in lanes.items():
                node, _, gpu = key.rpartition(":")
                out.append(
                    dict(
                        node=node,
                        gpu=int(gpu),
                        ranks=entry["ranks"],
                        engine_addrs=entry["engine_addrs"],
                        roles=entry["roles"],
                    )
                )
            out.sort(key=lambda entry: (entry["node"], entry["gpu"]))
            for i, entry in enumerate(out):
                entry["index"] = i
            return out

        seen: set[tuple[str, int]] = set(
            self._readers[Stream.GPU_UTIL].window(None, None).select("node", "gpu").unique().iter_rows()
        )
        events = self._phase_events(None, None)
        for event in events:
            for gpu in event.gpus:
                seen.add((event.node, gpu))
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    seen.add((node, gpu))
        info: dict[tuple[str, int], dict] = {
            (node, gpu): dict(node=node, gpu=gpu, ranks=set(), engine_addrs=set(), roles=set())
            for node, gpu in sorted(seen)
        }
        for event in events:
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
                index=i,
                ranks=sorted(entry["ranks"]),
                engine_addrs=sorted(entry["engine_addrs"]),
                roles=sorted(entry["roles"]),
            )
            for i, entry in enumerate(info.values())
        ]

    def resolve_lanes(self, grammar: str | None) -> set[tuple[str, int]] | None:
        """Parse the lane-selection grammar into a lane set (None = all lanes).

        Comma-separated selectors: ``rank:5`` / ``rank:0-7`` (train ranks),
        ``g:5`` / ``g:0-31`` (global lane numbers), ``node:<ip>``,
        ``gpu:<node>:<index>``, ``engine:<addr substring>``,
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
            elif kind == "g":
                # cluster-global lane numbers (the g{index} labels on every view)
                lo, _, hi = value.partition("-")
                positions = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if e["index"] in positions}
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
            dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"], roles=entry["roles"])
            for entry in self.lane_index()
            if lanes is None or (entry["node"], entry["gpu"]) in lanes
        ]
        row_of = {(lane["node"], lane["gpu"]): i for i, lane in enumerate(rows)}
        matrix = np.zeros((len(rows), x_buckets), dtype=np.uint8)
        span = t1 - t0

        def bucket(ts: float) -> int:
            return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

        if metric in self.HEATMAP_MAGNITUDE_FIELDS:
            frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
            values = np.zeros((len(rows), x_buckets))
            filled = np.zeros((len(rows), x_buckets), dtype=bool)
            for (node, gpu), part in frame.partition_by(["node", "gpu"], as_dict=True).items():
                row = row_of.get((node, gpu))
                if row is None:
                    continue
                buckets = np.clip(((part["ts"].to_numpy() - t0) / span * x_buckets).astype(np.int64), 0, x_buckets - 1)
                np.maximum.at(values[row], buckets, part[metric].to_numpy().astype(float))
                filled[row, buckets] = True
            peak = float(values[filled].max()) if filled.any() else 1.0
            scale = dict(max=100.0 if metric == "util" else peak)
            cell = np.minimum(255, (values[filled] / scale["max"] * 255).astype(np.int64))
            matrix[filled] = cell.astype(np.uint8)
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
            frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
            grouped = frame.group_by("node", "gpu").agg(pl.col("util").mean().alias("score"))
            scored = [(row["score"], row["node"], row["gpu"]) for row in grouped.iter_rows(named=True)]
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
