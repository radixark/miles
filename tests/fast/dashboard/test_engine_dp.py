"""dp-attention engine series: per-rank samples fold into engine-level values
by default (sum, or mean for intensive metrics) and stay apart under
``per_dp_rank=True``."""

from pathlib import Path

from miles.dashboard.store import EngineSample, Meta, MetricStore

ADDR = "http://e:1"


def make_store(tmp_path: Path, samples: list[EngineSample]) -> MetricStore:
    store = MetricStore(tmp_path / "dashboard")
    store.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    for sample in samples:
        store.append(sample)
    store.flush()
    return MetricStore.load(tmp_path / "dashboard")


def dp_samples(metric, points_by_rank):
    return [
        EngineSample(ts=ts, addr=ADDR, metric=metric, labels={"dp_rank": rank}, value=v)
        for rank, points in points_by_rank.items()
        for ts, v in points
    ]


def test_gauge_sums_across_dp_ranks(tmp_path):
    store = make_store(
        tmp_path,
        dp_samples("sglang_num_running_reqs", {"0": [(1.0, 11.0), (2.0, 3.0)], "1": [(1.0, 1.0), (2.0, 0.0)]}),
    )
    (series,) = store.engine_series("sglang_num_running_reqs")
    assert series["addr"] == ADDR
    assert series["labels"] == {}
    assert series["ts"] == [1.0, 2.0]
    assert series["value"] == [12.0, 3.0]


def test_intensive_gauge_averages_across_dp_ranks(tmp_path):
    store = make_store(tmp_path, dp_samples("sglang_token_usage", {"0": [(1.0, 0.9)], "1": [(1.0, 0.1)]}))
    (series,) = store.engine_series("sglang_token_usage")
    assert series["value"] == [0.5]


def test_per_dp_rank_splits_series(tmp_path):
    store = make_store(
        tmp_path,
        dp_samples("sglang_num_running_reqs", {"0": [(1.0, 11.0)], "1": [(1.0, 1.0)]}),
    )
    series = store.engine_series("sglang_num_running_reqs", per_dp_rank=True)
    by_rank = {s["labels"]["dp_rank"]: s for s in series}
    assert set(by_rank) == {"0", "1"}
    assert all(s["addr"] == ADDR for s in series)
    assert by_rank["0"]["value"] == [11.0]
    assert by_rank["1"]["value"] == [1.0]


def test_legacy_folded_duplicates_aggregate(tmp_path):
    # dumps written before dp_rank was kept: N same-ts samples, identical labels
    samples = [
        EngineSample(ts=ts, addr=ADDR, metric="sglang_num_running_reqs", labels={}, value=v)
        for ts, v in [(1.0, 11.0), (1.0, 1.0), (2.0, 3.0), (2.0, 0.0)]
    ]
    store = make_store(tmp_path, samples)
    (series,) = store.engine_series("sglang_num_running_reqs")
    assert series["ts"] == [1.0, 2.0]
    assert series["value"] == [12.0, 3.0]


def test_counter_rate_sums_ranks_before_diff(tmp_path):
    # rank counters: 0 -> 100 and 0 -> 60 over 2s; engine rate must be the
    # total 80/s, not a cross-rank interleave (which would leave gaps/garbage)
    store = make_store(
        tmp_path,
        dp_samples(
            "sglang_generation_tokens_total",
            {"0": [(0.0, 0.0), (2.0, 100.0)], "1": [(0.0, 0.0), (2.0, 60.0)]},
        ),
    )
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["ts"] == [2.0]
    assert series["value"] == [80.0]

    per_rank = store.engine_series("sglang_generation_tokens_per_s", per_dp_rank=True)
    assert {s["labels"]["dp_rank"]: s["value"] for s in per_rank} == {"0": [50.0], "1": [30.0]}


def test_histogram_mean_weights_ranks_by_count(tmp_path):
    # rank 0: 4 completions totalling 8s; rank 1: 1 completion of 3s
    # engine-level ttft mean = 11s / 5 completions, not mean-of-means
    samples = dp_samples(
        "sglang_time_to_first_token_seconds_sum", {"0": [(0.0, 0.0), (2.0, 8.0)], "1": [(0.0, 0.0), (2.0, 3.0)]}
    ) + dp_samples(
        "sglang_time_to_first_token_seconds_count", {"0": [(0.0, 0.0), (2.0, 4.0)], "1": [(0.0, 0.0), (2.0, 1.0)]}
    )
    store = make_store(tmp_path, samples)
    (series,) = store.engine_series("sglang_ttft_mean_s")
    assert series["ts"] == [2.0]
    assert series["value"] == [11.0 / 5.0]
