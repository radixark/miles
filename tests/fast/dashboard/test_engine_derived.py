"""Derived engine series: counter rates and histogram means."""

from pathlib import Path

from miles.dashboard.store import EngineSample, Meta, MetricStore


def make_store(tmp_path: Path, samples: list[EngineSample]) -> MetricStore:
    store = MetricStore(tmp_path / "dashboard")
    store.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    for sample in samples:
        store.append(sample)
    store.flush()
    return MetricStore.load(tmp_path / "dashboard")


def counter(addr, points, metric="sglang_generation_tokens_total"):
    return [EngineSample(ts=ts, addr=addr, metric=metric, labels={}, value=v) for ts, v in points]


def test_counter_rate_and_catalog(tmp_path):
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0), (4.0, 300.0)]))
    names = store.engine_metric_names()
    assert "sglang_generation_tokens_per_s" in names
    assert "sglang_generation_tokens_total" not in names

    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0, 4.0]
    assert series["value"] == [50.0, 100.0]


def test_counter_reset_leaves_gap(tmp_path):
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0), (4.0, 20.0), (6.0, 40.0)]))
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0, 6.0]  # the reset interval is a gap, not a negative rate
    assert series["value"] == [50.0, 10.0]


def test_histogram_mean_skips_idle_intervals(tmp_path):
    samples = counter(
        "http://e:1", [(0.0, 0.0), (2.0, 10.0), (4.0, 10.0)], metric="sglang_time_to_first_token_seconds_sum"
    ) + counter("http://e:1", [(0.0, 0.0), (2.0, 5.0), (4.0, 5.0)], metric="sglang_time_to_first_token_seconds_count")
    store = make_store(tmp_path, samples)
    assert "sglang_ttft_mean_s" in store.engine_metric_names()

    (series,) = store.engine_series("sglang_ttft_mean_s")
    assert series["ts"] == [2.0]  # no completions in [2, 4): gap
    assert series["value"] == [2.0]


def test_mean_requires_both_families(tmp_path):
    store = make_store(
        tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 10.0)], metric="sglang_time_to_first_token_seconds_sum")
    )
    assert "sglang_ttft_mean_s" not in store.engine_metric_names()
    assert store.engine_series("sglang_ttft_mean_s") == []
