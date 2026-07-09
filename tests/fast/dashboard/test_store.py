import numpy as np
import pytest

from miles.dashboard.store import (
    EngineInfo,
    EngineSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Stream,
    TopologySnapshot,
    minmax_downsample,
    stride_downsample,
)


def _one_of_each() -> list:
    return [
        MetricsRecord(ts=10.0, step_key="rollout/step", step=3, metrics={"rollout/rewards_mean": 0.5}),
        PhaseEvent(name="actor_train", t0=9.0, t1=11.0, node="10.0.0.2", gpus=[0, 1], rank=5, role="train"),
        TopologySnapshot(
            ts=8.0,
            engines=[
                EngineInfo(
                    addr="http://10.0.0.2:15000",
                    worker_type="regular",
                    engine_rank=0,
                    gpus=[["10.0.0.2", 0]],
                    gpu_uuids=[None],
                )
            ],
        ),
        GpuSample(ts=10.5, node="10.0.0.2", gpu=0, util=87, mem_mb=61234, power_w=612),
        EngineSample(ts=10.6, addr="http://10.0.0.2:15000", metric="sglang_num_running_reqs", labels={}, value=42.0),
    ]


def test_roundtrip_all_streams(tmp_path):
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="test-run", start_ts=1.0, args={"colocate": True}))
    records = _one_of_each()
    for record in records:
        writer.append(record)
    writer.flush()

    reader = MetricStore.load(tmp_path)
    assert reader.meta == Meta(run_name="test-run", start_ts=1.0, args={"colocate": True}, schema_version=1)
    for record in records:
        assert reader.records[record.stream] == [record]


def test_flush_clears_buffers_and_appends(tmp_path):
    writer = MetricStore(tmp_path)
    for record in _one_of_each():
        writer.append(record)
    writer.flush()
    writer.flush()  # empty buffers: no duplicate lines
    for record in _one_of_each():
        writer.append(record)
    writer.flush()

    reader = MetricStore.load(tmp_path)
    for stream in Stream:
        assert len(reader.records[stream]) == 2, stream


def test_follow_incremental(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()

    reader = MetricStore.load(tmp_path)
    assert len(reader.records[Stream.METRICS]) == 1
    assert reader.follow() == 0

    writer.append(MetricsRecord(ts=2.0, step_key="rollout/step", step=1, metrics={"a": 2}))
    writer.append(GpuSample(ts=2.5, node="n", gpu=0, util=10, mem_mb=1, power_w=1))
    writer.flush()
    assert reader.follow() == 2
    assert [r.step for r in reader.records[Stream.METRICS]] == [0, 1]
    assert reader.follow() == 0


def test_partial_trailing_line_left_for_later(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()

    path = tmp_path / Stream.METRICS.filename
    full_line = '{"ts": 2.0, "step_key": "rollout/step", "step": 1, "metrics": {"a": 2}}\n'
    with open(path, "a") as f:
        f.write(full_line[:20])  # crash mid-write: no trailing newline

    reader = MetricStore.load(tmp_path)
    assert len(reader.records[Stream.METRICS]) == 1

    with open(path, "a") as f:
        f.write(full_line[20:])  # writer completes the record
    assert reader.follow() == 1
    assert reader.records[Stream.METRICS][1].step == 1


def test_corrupt_complete_line_raises(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    writer.flush()
    with open(tmp_path / Stream.METRICS.filename, "a") as f:
        f.write("not json\n")

    with pytest.raises(ValueError, match="corrupt record"):
        MetricStore.load(tmp_path)


def test_wrong_field_complete_line_raises(tmp_path):
    with open(tmp_path / Stream.GPU_UTIL.filename, "w") as f:
        f.write('{"ts": 1.0, "unexpected_field": 1}\n')
    with pytest.raises(ValueError, match="corrupt record"):
        MetricStore.load(tmp_path)


def test_empty_dir(tmp_path):
    reader = MetricStore.load(tmp_path / "does_not_exist")
    assert reader.meta is None
    assert reader.follow() == 0
    assert reader.time_range() is None
    assert reader.metric_keys() == []
    assert reader.metric_series(["a"], x_key="rollout/step") == {"a": {"x": [], "y": [], "ts": []}}


def test_metric_series_filters_by_step_key_and_time(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1.0, "b": 10.0}))
    writer.append(MetricsRecord(ts=2.0, step_key="train/step", step=100, metrics={"a": -1.0}))
    writer.append(MetricsRecord(ts=3.0, step_key="rollout/step", step=1, metrics={"a": 2.0}))
    writer.flush()
    reader = MetricStore.load(tmp_path)

    series = reader.metric_series(["a", "b"], x_key="rollout/step")
    assert series["a"] == {"x": [0, 1], "y": [1.0, 2.0], "ts": [1.0, 3.0]}
    assert series["b"] == {"x": [0], "y": [10.0], "ts": [1.0]}

    series = reader.metric_series(["a"], x_key="train/step")
    assert series["a"] == {"x": [100], "y": [-1.0], "ts": [2.0]}

    series = reader.metric_series(["a"], x_key="rollout/step", t0=2.5, t1=3.5)
    assert series["a"] == {"x": [1], "y": [2.0], "ts": [3.0]}


def test_metric_keys_and_step_keys(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"b": 1, "a": 1}))
    writer.append(MetricsRecord(ts=2.0, step_key="eval/step", step=0, metrics={"c": 1}))
    writer.flush()
    reader = MetricStore.load(tmp_path)
    assert reader.metric_keys() == ["a", "b", "c"]
    assert reader.step_keys() == ["eval/step", "rollout/step"]


def test_time_range_spans_all_streams(tmp_path):
    writer = MetricStore(tmp_path)
    writer.append(PhaseEvent(name="rollout", t0=0.5, t1=7.0, node="n", gpus=[], rank=-1, role="rollout_manager"))
    writer.append(GpuSample(ts=9.0, node="n", gpu=0, util=1, mem_mb=1, power_w=1))
    writer.flush()
    reader = MetricStore.load(tmp_path)
    assert reader.time_range() == (0.5, 9.0)


def test_stride_downsample():
    xs, ys = np.arange(1000), np.arange(1000) * 2.0
    dx, dy = stride_downsample(xs, ys, 100)
    assert len(dx) == len(dy) == 100
    assert dx[0] == 0 and dx[-1] == 999  # endpoints preserved

    dx, dy = stride_downsample(xs, ys, 2000)  # already small enough
    assert len(dx) == 1000


def test_minmax_downsample_preserves_extremes():
    rng = np.random.default_rng(0)
    ys = rng.normal(size=5000)
    ys[1234] = 100.0  # spike
    ys[4321] = -100.0  # dip
    xs = np.arange(len(ys))

    dx, dy = minmax_downsample(xs, ys, 200)
    assert len(dx) <= 200
    assert dy.max() == 100.0
    assert dy.min() == -100.0
    assert (np.diff(dx) > 0).all()  # x stays sorted

    dx, dy = minmax_downsample(xs[:50], ys[:50], 200)
    assert len(dx) == 50  # small input unchanged
