import time

import pytest
import torch
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import Meta, MetricsRecord, MetricStore


@pytest.fixture
def dump_dir(tmp_path):
    dump_dummy_run(tmp_path, steps=2, with_eval=True)
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="dummy-run", start_ts=100.0, args={}))
    writer.append(MetricsRecord(ts=101.0, step_key="rollout/step", step=0, metrics={"rollout/rewards_mean": 0.5}))
    writer.append(MetricsRecord(ts=102.0, step_key="rollout/step", step=1, metrics={"rollout/rewards_mean": 0.6}))
    writer.append(MetricsRecord(ts=103.0, step_key="train/step", step=7, metrics={"train/loss": 1.5}))
    writer.flush()
    return tmp_path


@pytest.fixture
def client(dump_dir):
    store = MetricStore.load(dump_dir / "dashboard")
    return TestClient(make_app(store, DumpReader(dump_dir)))


def test_meta(client):
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "static"
    assert meta["run_name"] == "dummy-run"
    assert meta["rollout_ids"] == {"train": [0, 1], "eval": [0]}
    assert "rollout/rewards_mean" in meta["metric_keys"]
    # telemetry present: the catalog stays wandb-shaped, dump aggregates
    # are not advertised (they remain servable on explicit request)
    assert "dump/reward_mean" not in meta["metric_keys"]
    assert meta["step_keys"] == ["rollout/step", "train/step"]
    assert meta["capabilities"]["has_metrics"] is True
    assert meta["capabilities"]["has_tokenizer"] is True
    assert meta["capabilities"]["has_timeline"] is False


def test_metrics_from_store(client):
    series = client.get("/api/metrics", params={"keys": "rollout/rewards_mean"}).json()
    assert series["rollout/rewards_mean"] == {"x": [0, 1], "y": [0.5, 0.6], "ts": [101.0, 102.0]}

    series = client.get("/api/metrics", params={"keys": "train/loss", "x": "train/step"}).json()
    assert series["train/loss"]["x"] == [7]

    series = client.get("/api/metrics", params={"keys": "no/such_key"}).json()
    assert series["no/such_key"] == {"x": [], "y": [], "ts": []}


def test_metrics_dump_derived(client, dump_dir):
    series = client.get("/api/metrics", params={"keys": "dump/reward_mean,rollout/rewards_mean"}).json()
    aggregates = DumpReader(dump_dir).step_aggregates()
    assert series["dump/reward_mean"]["x"] == [0, 1]
    assert series["dump/reward_mean"]["y"] == pytest.approx(aggregates["reward_mean"].to_list())
    assert series["rollout/rewards_mean"]["y"] == [0.5, 0.6]

    assert client.get("/api/metrics", params={"keys": "dump/nope"}).status_code == 400
    assert client.get("/api/metrics", params={"keys": ""}).status_code == 400


def test_summary_and_groups_endpoints(client):
    body = client.get("/api/rollout/0/summary").json()
    assert body["rollout_id"] == 0 and body["evaluation"] is False
    assert len(body["rows"]) == 8
    assert "sample_index" in body["columns"] and "mean_abs_lp_diff" in body["columns"]

    eval_body = client.get("/api/rollout/0/summary", params={"eval": "true"}).json()
    assert eval_body["evaluation"] is True
    assert len(eval_body["rows"]) == 4

    groups = client.get("/api/rollout/0/groups").json()
    assert len(groups["rows"]) == 4
    assert "zero_std" in groups["columns"]

    assert client.get("/api/rollout/42/summary").status_code == 404


def test_tokens_endpoint(client):
    payload = client.get("/api/rollout/0/sample/0/tokens").json()
    assert payload["token_text"] is not None
    assert len(payload["token_ids"]) == payload["end"] - payload["start"] == payload["total_len"]

    window = client.get("/api/rollout/0/sample/0/tokens", params={"start": 1, "end": 4}).json()
    assert len(window["token_ids"]) == 3

    eval_payload = client.get("/api/rollout/0/sample/0/tokens", params={"eval": "true"}).json()
    assert eval_payload["train_log_probs"] is None
    assert eval_payload["rollout_log_probs"] is not None

    assert client.get("/api/rollout/0/sample/999999/tokens").status_code == 404
    assert client.get("/api/rollout/0/sample/0/tokens", params={"start": 5, "end": 5}).status_code == 400
    assert client.get("/api/rollout/42/sample/0/tokens").status_code == 404


def test_still_writing_maps_to_503(client, dump_dir):
    (dump_dir / "rollout_data" / "9.pt").write_bytes(b"garbage")  # fresh mtime
    assert client.get("/api/rollout/9/summary").status_code == 503


def test_nan_values_serialize_as_null(dump_dir):
    path = dump_dir / "train_data" / "0_0.pt"
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["log_probs"][0][:] = float("nan")
    torch.save(pack, path)
    stamp = time.time() - 100
    import os

    os.utime(path, (stamp, stamp))

    client = TestClient(make_app(MetricStore.load(dump_dir / "dashboard"), DumpReader(dump_dir)))
    body = client.get("/api/rollout/0/summary").json()  # json.loads would fail on bare NaN
    poisoned = pack["rollout_data"]["sample_indices"][0]
    row = next(r for r in body["rows"] if r["sample_index"] == poisoned)
    assert row["mean_abs_lp_diff"] is None
    assert row["mean_imp_ratio"] is None


def test_static_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "miles dashboard" in response.text


def test_dump_keys_advertised_only_without_telemetry(tmp_path):
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path)  # dumps only: no dashboard/ telemetry at all
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["has_metrics"] is False
    assert "dump/reward_mean" in meta["metric_keys"]


def test_engine_metric_catalog(tmp_path):
    from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry

    dump_dummy_telemetry(tmp_path)
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path)
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    meta = client.get("/api/meta").json()
    assert "sglang_num_running_reqs" in meta["engine_metric_keys"]


def test_make_demo_dir(tmp_path):
    from miles.dashboard.serve import make_demo_dir

    make_demo_dir(tmp_path)
    reader = DumpReader(tmp_path)
    assert reader.rollout_ids().train == [0, 1, 2]
    store = MetricStore.load(tmp_path / "dashboard")
    client = TestClient(make_app(store, reader))
    meta = client.get("/api/meta").json()
    assert meta["capabilities"]["has_timeline"] is True
    assert meta["rollout_ids"]["train"] == [0, 1, 2]
