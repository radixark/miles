import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_dummy_run
from tests.fast.dashboard.dummy_telemetry import SAMPLES_PER_STEP, dump_dummy_telemetry

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, TrajectoryEvent, TrajectoryEventKind


@pytest.fixture
def combined(tmp_path):
    """dump + telemetry in one dir — the same composition serve --demo uses."""
    dump_dummy_run(tmp_path)
    truth = dump_dummy_telemetry(tmp_path)
    return MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path), truth


# ------------------------------ lane assembly --------------------------------


def _event(ts, kind, index=1, turn=-1, version="2", detail=""):
    return TrajectoryEvent(
        ts=ts, kind=kind, sample_index=index, group_index=0, turn=turn, weight_version=version, detail=detail
    )


def test_trajectory_lanes_pairs_spans_and_attempts(tmp_path):
    writer = MetricStore(tmp_path)
    kinds = TrajectoryEventKind
    for event in (
        _event(10.0, kinds.ATTEMPT_START),
        _event(11.0, kinds.GEN_START, turn=1),
        _event(15.0, kinds.GEN_END, turn=1),
        _event(15.0, kinds.TOOL_START, turn=1),
        _event(18.0, kinds.TOOL_END, turn=1),
        _event(18.0, kinds.GEN_START, turn=2, version="3"),
        # turn-2 gen never ends: aborted mid-generation, then a second attempt
        _event(19.0, kinds.ATTEMPT_END, detail="aborted"),
        _event(30.0, kinds.ATTEMPT_START),
        _event(31.0, kinds.GEN_START, turn=1, version="4"),
        _event(35.0, kinds.GEN_END, turn=1, version="4"),
        _event(36.0, kinds.ATTEMPT_END, detail="completed"),
    ):
        writer.append(event)
    writer.flush()

    [lane] = MetricStore.load(tmp_path).trajectory_lanes()
    assert lane["sample_index"] == 1
    assert lane["status"] == "completed"
    assert lane["versions"] == [2, 3, 4]
    assert lane["attempts"] == [dict(t0=10.0, t1=19.0), dict(t0=30.0, t1=36.0)]
    by_kind = [(s["kind"], s["t0"], s["t1"], s["turn"]) for s in lane["segments"]]
    assert by_kind == [
        ("gen", 11.0, 15.0, 1),
        ("tool", 15.0, 18.0, 1),
        ("gen", 18.0, None, 2),  # unclosed: aborted mid-generation
        ("gen", 31.0, 35.0, 1),
    ]


def test_lifecycle_heatmap_row_cap(combined, monkeypatch):
    store, _, truth = combined
    monkeypatch.setattr(MetricStore, "LIFECYCLE_MAX_ROWS", 5)
    result = store.heatmap("lifecycle", x_buckets=50)
    assert len(result["rows"]) == 5
    assert result["rows_total"] == truth.steps * SAMPLES_PER_STEP
    # earliest submits kept (rows are submit-ordered)
    assert [r["sample_index"] for r in result["rows"]] == [0, 1, 2, 3, 4]


def test_lifecycle_heatmap_palette_and_rows(combined):
    store, _, truth = combined
    result = store.heatmap("lifecycle", x_buckets=200)
    assert result["palette"] == ["", "queue", "generating", "tool_wait"]
    assert len(result["rows"]) == truth.steps * SAMPLES_PER_STEP
    matrix = np.frombuffer(result["values"], dtype=np.uint8).reshape(len(result["rows"]), 200)
    assert (matrix == result["palette"].index("generating")).sum() > 0
    assert (matrix == result["palette"].index("tool_wait")).sum() > 0
    # rows are submit-ordered
    firsts = [r["sample_index"] for r in result["rows"][: SAMPLES_PER_STEP + 1]]
    assert firsts == sorted(firsts)


# --------------------------------- endpoint ----------------------------------


def test_rollout_trajectories_endpoint(combined):
    store, reader, truth = combined
    client = TestClient(make_app(store, reader))

    payload = client.get("/api/rollout/1/trajectories").json()
    lanes = payload["lanes"]
    assert {lane["sample_index"] for lane in lanes} == set(range(SAMPLES_PER_STEP, 2 * SAMPLES_PER_STEP))
    agentic = next(lane for lane in lanes if lane["sample_index"] % 3 == 0)
    plain = next(lane for lane in lanes if lane["sample_index"] % 3 == 1)
    assert len(agentic["versions"]) == 2 and [s["kind"] for s in agentic["segments"]] == ["gen", "tool", "gen"]
    assert len(plain["versions"]) == 1 and [s["kind"] for s in plain["segments"]] == ["gen"]
    assert all(lane["status"] == "completed" for lane in lanes)
    # events land within the step's rollout window, before the consume anchor
    if payload["consume_ts"] is not None:
        assert all(lane["last_ts"] <= payload["consume_ts"] + 1e-6 for lane in lanes)


def test_rollout_trajectories_empty_for_probe_less_runs(tmp_path):
    dump_dummy_run(tmp_path)  # dump only: no telemetry, no events
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    payload = client.get("/api/rollout/0/trajectories").json()
    assert payload["lanes"] == []
