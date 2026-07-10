import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_telemetry import DRAIN_A, DRAIN_B, ENGINE_B_NEW, GPU_NODE, dump_dummy_telemetry

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import GpuSample, Meta, MetricStore


@pytest.fixture
def loaded(tmp_path):
    truth = dump_dummy_telemetry(tmp_path)
    return MetricStore.load(tmp_path / "dashboard"), truth


# ------------------------------ lane grammar ---------------------------------


def test_resolve_lanes_grammar(loaded):
    store, truth = loaded
    assert store.resolve_lanes(None) is None
    assert store.resolve_lanes("all") is None
    # dummy fixture: train rank r runs on gpu r
    assert store.resolve_lanes("rank:1") == {(GPU_NODE, 1)}
    assert store.resolve_lanes("rank:0-2") == {(GPU_NODE, 0), (GPU_NODE, 1), (GPU_NODE, 2)}
    assert store.resolve_lanes(f"gpu:{GPU_NODE}:3") == {(GPU_NODE, 3)}
    assert store.resolve_lanes(f"node:{GPU_NODE}") == {(GPU_NODE, g) for g in range(truth.gpus)}
    assert store.resolve_lanes("node:10.9.9.9") == set()  # valid selector, no match
    # engine B (either addr) covers gpus 2-3
    assert store.resolve_lanes("engine:15004") == {(GPU_NODE, 2), (GPU_NODE, 3)}
    assert store.resolve_lanes("rank:0, engine:15000") == {(GPU_NODE, 0), (GPU_NODE, 1)}
    # dummy telemetry is colocate-shaped: every lane carries both roles
    assert store.resolve_lanes("role:train") == store.resolve_lanes("role:rollout")


def test_resolve_lanes_rejects_bad_grammar(loaded):
    store, _ = loaded
    for bad in ("bananas", "rank", "role:banana", "gpu:3"):
        with pytest.raises(ValueError):
            store.resolve_lanes(bad)


def test_queries_respect_lane_filter(loaded):
    store, truth = loaded
    only = {(GPU_NODE, 2)}
    phases = store.phases_by_lane(lanes=only)
    assert phases and {(p["node"], p["gpu"]) for p in phases} == only
    assert {p["name"] for p in phases} >= {"rollout", "actor_train", "initialize"}

    series = store.gpu_series(lanes=only)
    assert set(series) == {f"{GPU_NODE}:2"}


# -------------------------------- heatmap ------------------------------------


def _tiny_store(tmp_path):
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="tiny", start_ts=0.0, args={}))
    for ts in range(10):
        writer.append(GpuSample(ts=float(ts), node="n", gpu=0, util=10, mem_mb=1000, power_w=100))
        writer.append(GpuSample(ts=float(ts), node="n", gpu=1, util=50, mem_mb=4000, power_w=100))
    writer.append(GpuSample(ts=4.0, node="n", gpu=0, util=90, mem_mb=8000, power_w=100))  # spike
    writer.flush()
    return MetricStore.load(tmp_path / "dashboard")


def test_heatmap_util_bucket_max(tmp_path):
    store = _tiny_store(tmp_path)
    result = store.heatmap("util", t0=0.0, t1=10.0, x_buckets=5)
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(2, 5)
    assert [r["gpu"] for r in result["rows"]] == [0, 1]
    assert result["scale"] == {"max": 100.0}
    # lane 0: util 10 everywhere except the spike bucket (t=4 -> bucket 2)
    expected_low = int(10 / 100 * 255)
    assert matrix[0, 2] == int(90 / 100 * 255)
    assert matrix[0, 0] == matrix[0, 4] == expected_low
    assert (matrix[1] == int(50 / 100 * 255)).all()


def test_heatmap_mem_scales_to_peak(tmp_path):
    store = _tiny_store(tmp_path)
    result = store.heatmap("mem_mb", t0=0.0, t1=10.0, x_buckets=5)
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(2, 5)
    assert result["scale"] == {"max": 8000.0}
    assert matrix[0, 2] == 255  # the 8000 MB spike
    assert matrix[1, 0] == int(4000 / 8000 * 255)


def test_heatmap_phase_paints_dominant_and_initialize(loaded):
    store, truth = loaded
    result = store.heatmap("phase", x_buckets=400)
    palette = result["palette"]
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(len(result["rows"]), 400)
    assert palette[0] == ""
    assert {"initialize", "rollout", "actor_train", "train_wait"} <= set(palette)

    # rows carry roles so the carpet can draw the train/rollout divider
    assert result["rows"][0]["roles"] == ["rollout", "train"]

    lane0 = matrix[0]
    # the run starts with the initialize band, then rollout
    assert palette[lane0[0]] == "initialize"
    rollout_id = palette.index("rollout")
    assert (lane0 == rollout_id).sum() > 0
    # active phases win over idle in shared buckets: actor_train present
    assert (lane0 == palette.index("actor_train")).sum() > 0


def test_heatmap_respects_lane_selection(loaded):
    store, _ = loaded
    result = store.heatmap("util", lanes={(GPU_NODE, 1)}, x_buckets=50)
    assert [(r["node"], r["gpu"]) for r in result["rows"]] == [(GPU_NODE, 1)]
    assert len(result["values"]) == 50


# -------------------------------- outliers -----------------------------------


def test_outliers_lowest_util(loaded):
    store, truth = loaded
    ranked = store.outliers("lowest_util", top_k=4)
    # engine-B lanes (gpus 2-3) drain earlier each rollout -> lower mean util
    assert DRAIN_B < DRAIN_A
    assert {entry["gpu"] for entry in ranked[:2]} == {2, 3}
    assert ranked[0]["score"] < ranked[-1]["score"]


def test_outliers_slowest_phase(loaded):
    store, truth = loaded
    ranked = store.outliers("slowest_phase:update_weights", top_k=4)
    # fixture skew: update_weights duration grows with rank (4 + 0.3*rank)
    assert [entry["gpu"] for entry in ranked] == [3, 2, 1, 0]

    with pytest.raises(ValueError):
        store.outliers("bananas")


# --------------------------------- server ------------------------------------


def test_server_endpoints(tmp_path):
    dump_dummy_telemetry(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))

    response = client.get("/api/timeline/heatmap", params={"metric": "util", "x_buckets": 100})
    assert response.status_code == 200
    body = response.content
    header_len = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + header_len])
    matrix = np.frombuffer(body[4 + header_len :], dtype=np.uint8)
    assert header["metric"] == "util" and header["x_buckets"] == 100
    assert len(matrix) == len(header["rows"]) * 100

    phases = client.get("/api/timeline/phases", params={"lanes": "rank:0"}).json()["phases"]
    assert {(p["node"], p["gpu"]) for p in phases} == {(GPU_NODE, 0)}

    gpu = client.get("/api/timeline/gpu", params={"lanes": f"engine:{ENGINE_B_NEW.rsplit(':', 1)[1]}"}).json()
    assert set(gpu["lanes"]) == {f"{GPU_NODE}:2", f"{GPU_NODE}:3"}

    outliers = client.get("/api/timeline/outliers", params={"criterion": "lowest_util", "top_k": 2}).json()
    assert len(outliers["outliers"]) == 2

    assert client.get("/api/timeline/phases", params={"lanes": "bananas"}).status_code == 400
    assert client.get("/api/timeline/heatmap", params={"metric": "nope"}).status_code == 400
    assert client.get("/api/timeline/outliers", params={"criterion": "nope"}).status_code == 400
    assert client.get("/api/timeline/heatmap", params={"x_buckets": 1}).status_code == 400
