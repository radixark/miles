"""Derived engine series: counter rates and histogram means."""

import threading
import time
from pathlib import Path

import polars as pl
import pytest

import miles.dashboard.store as store_module
from miles.dashboard.store import EngineSample, Meta, MetricStore, _ThinnedEngineSeries


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


def test_thinned_engine_series_survives_concurrent_cold_readers(tmp_path, monkeypatch):
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0)]))
    view = store._thinned_engine_series
    real_parse = view._parse
    monkeypatch.setattr(view, "_parse", lambda *a: (time.sleep(0.2), real_parse(*a))[1])

    results, errors = [], []
    threads = [threading.Thread(target=lambda: _collect(store, results, errors), name=f"reader-{i}") for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert all(r == results[0] and r[0]["value"] == [50.0] for r in results)


def _collect(store, results, errors):
    try:
        results.append(store.engine_series("sglang_generation_tokens_per_s"))
    except Exception as e:  # noqa: BLE001
        errors.append(e)


def test_failed_read_is_read_again_on_the_next_query(tmp_path, monkeypatch):
    store = make_store(tmp_path, counter("http://e:1", [(0.0, 0.0), (2.0, 100.0)]))
    view = store._thinned_engine_series
    real_parse = view._parse
    monkeypatch.setattr(view, "_parse", _raise_parse_error)
    with pytest.raises(OSError):
        store.engine_series("sglang_generation_tokens_per_s")
    monkeypatch.setattr(view, "_parse", real_parse)
    (series,) = store.engine_series("sglang_generation_tokens_per_s")
    assert series["value"] == [50.0]


def _raise_parse_error(*args):
    raise OSError("partition read failed")


HOUR = 3600.0


def _hourly_counter_store(tmp_path, *, hours: int) -> MetricStore:
    return make_store(
        tmp_path,
        counter("http://e:1", [(h * HOUR + s, (h * HOUR + s) * 10.0) for h in range(hours) for s in (0.0, 2.0)]),
    )


def _count_engine_parses(store, monkeypatch) -> list[str]:
    parsed_partitions = []
    parse = store._parse_columnar
    monkeypatch.setattr(
        store,
        "_parse_columnar",
        lambda stream, raw, path, offset: parsed_partitions.append(path.stem) or parse(stream, raw, path, offset),
    )
    return parsed_partitions


def test_run_long_query_reads_only_appended_bytes(tmp_path, monkeypatch):
    """An append must not turn the next run-long query into a re-parse of
    every partition (the live view appends every few seconds)."""
    hours = MetricStore.PARTITION_CACHE_BLOCKS + 6
    store = _hourly_counter_store(tmp_path, hours=hours)
    parsed_partitions = _count_engine_parses(store, monkeypatch)
    store.engine_series("sglang_generation_tokens_per_s")
    parsed_before_append = len(parsed_partitions)

    writer = MetricStore(tmp_path / "dashboard")
    writer.append(
        EngineSample(
            ts=(hours - 1) * HOUR + 4.0,
            addr="http://e:1",
            metric="sglang_generation_tokens_total",
            labels={},
            value=1.0,
        )
    )
    writer.flush()
    store.engine_series("sglang_generation_tokens_per_s")
    assert (
        len(parsed_partitions) - parsed_before_append == 1
    ), f"an append re-parsed {len(parsed_partitions) - parsed_before_append} partitions"


def test_engine_catalog_needs_no_stream_scan(tmp_path, monkeypatch):
    store = _hourly_counter_store(tmp_path, hours=3)
    parsed_partitions = _count_engine_parses(store, monkeypatch)
    assert "sglang_generation_tokens_per_s" in store.engine_metric_names()
    assert parsed_partitions == []


def test_long_run_keeps_bounded_scrapes_of_the_raw_values(tmp_path, monkeypatch):

    monkeypatch.setattr(_ThinnedEngineSeries, "SCRAPES_PER_ENGINE", 50)
    points = [(i * 2.0, float(i)) for i in range(1000)]  # 2 s scrapes, value = scrape index
    store = make_store(
        tmp_path,
        [
            EngineSample(ts=ts, addr="http://e:1", metric="sglang_num_running_reqs", labels={}, value=v)
            for ts, v in points
        ],
    )
    (series,) = store.engine_series("sglang_num_running_reqs", max_points=10_000)
    assert 50 <= len(series["ts"]) <= 100
    assert all(value == ts / 2.0 for ts, value in zip(series["ts"], series["value"], strict=True))


def test_scrape_split_across_flushes_keeps_every_dp_rank(tmp_path, monkeypatch):

    monkeypatch.setattr(_ThinnedEngineSeries, "SCRAPES_PER_ENGINE", 5)
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="t", start_ts=0.0, args={}))

    def scrape(ts, ranks):
        for rank in ranks:
            writer.append(
                EngineSample(
                    ts=ts,
                    addr="http://e:1",
                    metric="sglang_num_running_reqs",
                    labels={"dp_rank": str(rank)},
                    value=1.0,
                )
            )

    for k in range(40):
        scrape(k * 10.0, range(4))
    writer.flush()
    store = MetricStore.load(tmp_path / "dashboard")
    store.engine_series("sglang_num_running_reqs")  # thin once, so the next scrape is judged against a bucket
    scrape(400.0, range(2))
    writer.flush()
    store.engine_series("sglang_num_running_reqs")
    scrape(400.0, range(2, 4))
    writer.flush()
    series_per_dp_rank = store.engine_series("sglang_num_running_reqs", per_dp_rank=True, max_points=10_000)
    last_ts_by_dp_rank = {s["labels"]["dp_rank"]: s["ts"][-1] for s in series_per_dp_rank}
    assert len(set(last_ts_by_dp_rank.values())) == 1, f"a split scrape kept only some ranks: {last_ts_by_dp_rank}"


def _engine_sample(metric: str, ts: float) -> EngineSample:
    return EngineSample(ts=ts, addr="http://e:1", metric=metric, labels={}, value=1.0)


def test_failed_catalog_rewrite_is_retried(tmp_path, monkeypatch):
    """The collector retries a flush that raised: the retry must still add the
    names whose catalog rewrite failed."""

    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    writer.append(_engine_sample("sglang_num_running_reqs", 1.0))
    writer.flush()
    writer.append(_engine_sample("sglang_gen_throughput", 2.0))
    replace = store_module.os.replace
    monkeypatch.setattr(store_module.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        writer.flush()
    monkeypatch.setattr(store_module.os, "replace", replace)
    writer.flush()
    assert "sglang_gen_throughput" in MetricStore.load(tmp_path).engine_metric_names()


def test_resumed_dir_without_a_catalog_keeps_its_earlier_engine_metrics(tmp_path):
    """A dir written before the catalog existed, resumed by a writer that has
    one: a catalog of the new names alone would hide the earlier ones."""
    old_writer = MetricStore(tmp_path)
    old_writer.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    old_writer.append(_engine_sample("sglang_cache_hit_rate", 1.0))
    old_writer.flush()
    (tmp_path / MetricStore.ENGINE_CATALOG_FILENAME).unlink()  # as written before the catalog existed

    resumed_writer = MetricStore(tmp_path)
    resumed_writer.append(_engine_sample("sglang_num_running_reqs", 2.0))
    resumed_writer.flush()
    names = MetricStore.load(tmp_path).engine_metric_names()
    assert {"sglang_cache_hit_rate", "sglang_num_running_reqs"} <= set(names)


def test_kept_scrape_counts_track_the_view_across_split_scrapes(tmp_path, monkeypatch):
    """The bound check reads per-engine counts kept from each append; they must
    equal a recount of the view, a scrape split across two flushes included."""

    monkeypatch.setattr(_ThinnedEngineSeries, "SCRAPES_PER_ENGINE", 5)
    writer = MetricStore(tmp_path / "dashboard")
    writer.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    store = MetricStore.load(tmp_path / "dashboard")
    for k in range(60):
        for rank in range(2):
            writer.append(
                EngineSample(ts=k * 3.0, addr="http://e:1", metric="m", labels={"dp_rank": str(rank)}, value=1.0)
            )
            writer.flush()  # every scrape straddles a flush
        store.engine_series("m")
    view = store._thinned_engine_series
    kept, _ = view.kept_scrapes()
    recount = dict(kept.group_by("addr").agg(pl.col("ts").n_unique()).iter_rows())
    assert view._kept_scrape_count_by_addr == recount
    assert max(recount.values()) <= 2 * _ThinnedEngineSeries.SCRAPES_PER_ENGINE


def test_unreadable_engine_catalog_neither_stops_flushes_nor_hides_names(tmp_path):
    """A catalog that does not parse cannot vouch for the stream: the writer
    drops it and keeps writing every stream, readers fall back to the stream."""
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="t", start_ts=0.0, args={}))
    writer.append(EngineSample(ts=1.0, addr="http://e:1", metric="sglang_num_running_reqs", labels={}, value=1.0))
    writer.flush()
    (tmp_path / MetricStore.ENGINE_CATALOG_FILENAME).write_text("{truncated")
    assert "sglang_num_running_reqs" in MetricStore.load(tmp_path).engine_metric_names()

    resumed = MetricStore(tmp_path)
    resumed.append(EngineSample(ts=2.0, addr="http://e:1", metric="sglang_num_queue_reqs", labels={}, value=1.0))
    resumed.flush()
    assert not (tmp_path / MetricStore.ENGINE_CATALOG_FILENAME).exists()
    assert {"sglang_num_running_reqs", "sglang_num_queue_reqs"} <= set(
        MetricStore.load(tmp_path).engine_metric_names()
    )
