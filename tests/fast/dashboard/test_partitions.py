import json

import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import GpuSample, Meta, MetricStore, PhaseEvent, Role, Stream, _hour_key

# hour-aligned base so partition boundaries land where the test says they do
BASE = ((1_000_000 // 3600) + 1) * 3600.0
HOUR = 3600.0


def _three_hour_store(tmp_path):
    """Two lanes of gpu samples across three hours, one rollout-role train
    event per hour, and one long train event straddling the h0/h1 boundary."""
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="hours", start_ts=BASE, args={}))
    for hour in range(3):
        for sec in (10.0, 1800.0, 3590.0):
            ts = BASE + hour * HOUR + sec
            for gpu in (0, 1):
                writer.append(GpuSample(ts=ts, node="n", gpu=gpu, util=10 * (hour + 1), mem_mb=100, power_w=50))
        writer.append(
            PhaseEvent(
                name="actor_train",
                t0=BASE + hour * HOUR + 100,
                t1=BASE + hour * HOUR + 200,
                node="n",
                gpus=[0],
                rank=0,
                role=Role.TRAIN,
            )
        )
    # straddler: starts in hour 0, completes in hour 1 -> lands in the h1 file
    writer.append(
        PhaseEvent(
            name="straddle",
            t0=BASE + 3500,
            t1=BASE + HOUR + 300,
            node="n",
            gpus=[1],
            rank=1,
            role=Role.TRAIN,
        )
    )
    writer.flush()
    return MetricStore.load(tmp_path)


def test_flush_routes_hourly_files(tmp_path):
    _three_hour_store(tmp_path)
    gpu_files = sorted(p.name for p in (tmp_path / "gpu_util").glob("*.jsonl"))
    assert gpu_files == sorted(f"{_hour_key(BASE + h * HOUR)}.jsonl" for h in range(3))
    assert not (tmp_path / Stream.GPU_UTIL.filename).exists()
    # the straddler is keyed by its END hour
    h1 = tmp_path / "phases" / f"{_hour_key(BASE + HOUR)}.jsonl"
    assert any(json.loads(line)["name"] == "straddle" for line in h1.read_text().splitlines())


def test_windowed_read_parses_only_touched_partitions(tmp_path):
    store = _three_hour_store(tmp_path)
    reader = store._readers[Stream.GPU_UTIL]
    parsed = []
    original = reader._parse
    reader._parse = lambda raw, path, offset: (parsed.append(path.stem), original(raw, path, offset))[-1]

    series = store.gpu_series(t0=BASE + HOUR + 5, t1=BASE + HOUR + 1900)
    assert parsed == [_hour_key(BASE + HOUR)]
    assert set(series) == {"n:0", "n:1"}
    assert series["n:0"]["util"] == [20, 20]  # samples at +10 and +1800 of hour 1

    # second read inside the same hour: served from cache, no re-parse
    store.gpu_series(t0=BASE + HOUR + 5, t1=BASE + HOUR + 2000)
    assert parsed == [_hour_key(BASE + HOUR)]


def test_cached_partition_tails_appended_rows(tmp_path):
    store = _three_hour_store(tmp_path)
    window = dict(t0=BASE + 2 * HOUR, t1=BASE + 2 * HOUR + 3599)
    assert len(store.gpu_series(**window)["n:0"]["ts"]) == 3

    writer = MetricStore(tmp_path)
    writer.append(GpuSample(ts=BASE + 2 * HOUR + 3595.0, node="n", gpu=0, util=99, mem_mb=100, power_w=50))
    writer.flush()
    assert store.follow() == 1  # cached hour-2 block picks up the append
    assert len(store.gpu_series(**window)["n:0"]["ts"]) == 4


def test_phases_forward_slack_finds_straddler(tmp_path):
    store = _three_hour_store(tmp_path)
    # window entirely inside hour 0; the straddler's file is hour 1
    phases = store.phases_by_lane(t0=BASE + 3400, t1=BASE + 3550)
    assert "straddle" in {p["name"] for p in phases}


def test_initialize_only_when_window_covers_run_start(tmp_path):
    store = _three_hour_store(tmp_path)
    full = {p["name"] for p in store.phases_by_lane()}
    assert "initialize" in full
    later = {p["name"] for p in store.phases_by_lane(t0=BASE + 50, t1=BASE + 300)}
    assert "initialize" not in later


def test_legacy_flat_file_still_reads(tmp_path):
    (tmp_path / Meta.FILENAME).write_text(json.dumps(dict(run_name="v1", start_ts=1.0, args={})))
    with open(tmp_path / Stream.GPU_UTIL.filename, "w") as f:
        f.write('{"ts": 5.0, "node": "n", "gpu": 0, "util": 42, "mem_mb": 1, "power_w": 1}\n')
    store = MetricStore.load(tmp_path)
    assert store.has_stream(Stream.GPU_UTIL)
    assert store.gpu_series()["n:0"]["util"] == [42]
    # no catalog in a v1 dir: lane_index falls back to the stream scan
    assert not (tmp_path / MetricStore.CATALOG_FILENAME).exists()
    assert [(e["node"], e["gpu"]) for e in store.lane_index()] == [("n", 0)]


def test_lane_catalog_matches_stream_scan(tmp_path):
    dump_dummy_telemetry(tmp_path)
    dashboard = tmp_path / "dashboard"
    assert (dashboard / MetricStore.CATALOG_FILENAME).exists()
    from_catalog = MetricStore.load(dashboard).lane_index()
    (dashboard / MetricStore.CATALOG_FILENAME).unlink()
    from_scan = MetricStore.load(dashboard).lane_index()
    assert from_catalog == from_scan


def test_time_range_from_edge_stamps(tmp_path):
    store = _three_hour_store(tmp_path)
    t0, t1 = store.time_range()
    assert t0 == BASE + 10.0  # first gpu sample
    assert t1 == BASE + 2 * HOUR + 3590.0  # last gpu sample


def test_server_enforces_window_cap(tmp_path):
    dump_dummy_telemetry(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["max_window_s"] == MetricStore.MAX_WINDOW_S

    over = MetricStore.MAX_WINDOW_S + 1
    for path, params in (
        ("/api/timeline/gpu", {}),
        ("/api/timeline/phases", {}),
        ("/api/timeline/engine_series", {"metric": "sglang_num_running_reqs"}),
        ("/api/timeline/heatmap", {"metric": "util"}),
    ):
        ok = client.get(path, params=params)
        assert ok.status_code == 200, path
        bad = client.get(path, params={**params, "t0": 0.0, "t1": over})
        assert bad.status_code == 400, path
        assert "max_window_s" in bad.json()["detail"]


def test_corrupt_partition_line_surfaces_at_read(tmp_path):
    store = _three_hour_store(tmp_path)
    partition = tmp_path / "gpu_util" / f"{_hour_key(BASE)}.jsonl"
    with open(partition, "a") as f:
        f.write("not json\n")
    with pytest.raises(ValueError, match="corrupt record"):
        store.gpu_series(t0=BASE, t1=BASE + 100)


def test_open_marker_partitions_by_start_hour_and_is_found(tmp_path):
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="open", start_ts=BASE, args={}))
    writer.append(GpuSample(ts=BASE + 2 * HOUR, node="n", gpu=0, util=1, mem_mb=1, power_w=1))
    writer.append(
        PhaseEvent(
            name="rollout", t0=BASE + 100, t1=PhaseEvent.OPEN_T1, node="n", gpus=[0], rank=0, role=Role.TRAIN
        )
    )
    writer.flush()
    # the marker sits in hour 0 (keyed by t0, not by the -1 sentinel)
    h0 = tmp_path / "phases" / f"{_hour_key(BASE)}.jsonl"
    assert h0.exists() and "rollout" in h0.read_text()

    store = MetricStore.load(tmp_path)
    # windowed query two hours later still sees the growing band (backward slack)
    phases = [p for p in store.phases_by_lane(t0=BASE + 2 * HOUR - 10, t1=BASE + 2 * HOUR) if p["name"] == "rollout"]
    assert phases and phases[0]["t1"] == BASE + 2 * HOUR  # clipped to the data edge
