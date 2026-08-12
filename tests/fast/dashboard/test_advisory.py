import polars as pl

from miles.dashboard.advisory import compute_advisories
from miles.dashboard.dump_reader import RolloutIds
from miles.dashboard.store import EngineSample, Meta, MetricStore, PhaseEvent


def _store(tmp_path, *, args: dict, engine_samples: list[EngineSample]) -> MetricStore:
    writer = MetricStore(tmp_path)
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args=args))
    for sample in engine_samples:
        writer.append(sample)
    writer.flush()
    return MetricStore.load(tmp_path)


def _engine(metric: str, value: float, ts: float = 1.0, addr: str = "http://n:1") -> EngineSample:
    return EngineSample(ts=ts, addr=addr, metric=metric, labels={}, value=value)


def test_no_engine_series_returns_empty(tmp_path):
    store = _store(tmp_path, args={}, engine_samples=[])
    assert compute_advisories(store) == []


def test_low_concurrency_flagged(tmp_path):
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[_engine("sglang_num_running_reqs", 20.0)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"
    assert "max-running-requests" in advisory.message


def test_high_concurrency_not_flagged(tmp_path):
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[_engine("sglang_num_running_reqs", 80.0)],
    )
    assert compute_advisories(store) == []


def test_low_cache_hit_flagged_when_not_colocate(tmp_path):
    store = _store(
        tmp_path,
        args={"colocate": False, "sglang_mem_fraction_static": 0.7},
        engine_samples=[_engine("sglang_cache_hit_rate", 0.02)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"
    assert "mem-fraction-static" in advisory.message


def test_low_cache_hit_not_flagged_when_colocate(tmp_path):
    # colocate runs deliberately trade cache size for training memory — a low
    # hit rate there is an expected cost, not a misconfiguration to flag
    store = _store(
        tmp_path,
        args={"colocate": True, "sglang_mem_fraction_static": 0.5},
        engine_samples=[_engine("sglang_cache_hit_rate", 0.02)],
    )
    assert compute_advisories(store) == []


def test_high_token_usage_flagged(tmp_path):
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_engine("sglang_token_usage", 0.99)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "throughput" in advisory.message


def test_healthy_run_has_no_advisories(tmp_path):
    store = _store(
        tmp_path,
        args={"colocate": False, "sglang_max_running_requests": 100, "sglang_mem_fraction_static": 0.7},
        engine_samples=[
            _engine("sglang_num_running_reqs", 80.0),
            _engine("sglang_cache_hit_rate", 0.6),
            _engine("sglang_token_usage", 0.5),
        ],
    )
    assert compute_advisories(store) == []


def test_window_narrows_to_requested_range(tmp_path):
    # a stale spike outside [t0, t1] must not leak into the windowed view
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_running_reqs", 20.0, ts=1.0),
            _engine("sglang_num_running_reqs", 90.0, ts=100.0),
        ],
    )
    assert compute_advisories(store, t0=0.0, t1=10.0) != []
    assert compute_advisories(store, t0=50.0, t1=150.0) == []


# ------------------------- v2: health alarms + gating -------------------------


def _phase(name, t0, t1, rank=0):
    return PhaseEvent(name=name, t0=t0, t1=t1, node="n1", gpus=[0], rank=rank, role="train")


def _closed_steps(n=3, duration=10.0, start=0.0):
    return [_phase("train_wait", start + i * 100.0, start + i * 100.0 + duration) for i in range(n)]


def test_stalled_open_phase_is_critical_and_suppresses_tuning(tmp_path):
    # the open train_wait is 3900s old vs a 10s median: a stall — and the
    # low-concurrency observation must NOT surface as tuning advice with it
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_running_reqs", 5.0, ts=1.0),
            _engine("sglang_num_running_reqs", 5.0, ts=4200.0),
        ],
    )
    writer = MetricStore(tmp_path)
    for event in [*_closed_steps(), _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)]:
        writer.append(event)
    writer.flush()
    store = MetricStore.load(tmp_path)

    [advisory] = compute_advisories(store)
    assert advisory.level == "critical"
    assert "train_wait" in advisory.message
    assert "stalled" in advisory.message


def test_open_phase_with_closing_twin_is_not_a_stall(tmp_path):
    writer = MetricStore(tmp_path)
    open_marker = _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)
    for event in [*_closed_steps(), open_marker, _phase("train_wait", 300.0, 310.0)]:
        writer.append(event)
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=4200.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_previous_attempt_open_marker_is_not_a_stall(tmp_path):
    # resuming into the same dump dir appends: the crashed attempt's open
    # markers are still in the stream and never close, and their age against
    # this attempt's clock would latch a critical for the rest of the run
    writer = MetricStore(tmp_path)
    for event in [*_closed_steps(), _phase("train_wait", 300.0, PhaseEvent.OPEN_T1)]:
        writer.append(event)
    for event in _closed_steps(start=10000.0):
        writer.append(event)
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=20000.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=10000.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_forever_open_phase_without_baseline_is_not_a_stall(tmp_path):
    # fully-async rollout keeps one manager phase open for the whole run by
    # design; with no closed instances there is no baseline and no claim
    writer = MetricStore(tmp_path)
    writer.append(_phase("rollout", 0.0, PhaseEvent.OPEN_T1))
    writer.append(_engine("sglang_num_running_reqs", 80.0, ts=90000.0))
    writer.write_meta(Meta(run_name="advisory-test", start_ts=0.0, args={}))
    writer.flush()
    assert compute_advisories(MetricStore.load(tmp_path)) == []


def test_abort_storm_is_warning_and_suppresses_tuning(tmp_path):
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100, "n_samples_per_prompt": 8},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 100.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 40.0, ts=2.0),
            _engine("sglang_num_running_reqs", 5.0, ts=1.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "40%" in advisory.message
    assert "8-sample group" in advisory.message


def test_abort_storm_survives_an_engine_restart(tmp_path):
    # an engine coming back on the same address resets its counters mid-series;
    # last-minus-first would see only the healthy post-restart segment and drop
    # the storm entirely
    store = _store(
        tmp_path,
        args={},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 1000.0, ts=2.0),
            _engine("sglang_num_requests_total", 0.0, ts=3.0),
            _engine("sglang_num_requests_total", 500.0, ts=4.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=3.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=4.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "60%" in advisory.message
    assert "(900/1500" in advisory.message


def test_cleared_abort_storm_stops_alarming(tmp_path):
    # 900/9000 over the run stays above the threshold forever, but the engine
    # has been healthy for hours: the alarm — and the tuning tier it suppresses
    # — must follow the current state
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 100},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 1000.0, ts=100.0),
            _engine("sglang_num_requests_total", 5000.0, ts=9000.0),
            _engine("sglang_num_requests_total", 9000.0, ts=10000.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=100.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=9000.0),
            _engine("sglang_num_aborted_requests_total", 900.0, ts=10000.0),
            _engine("sglang_num_running_reqs", 5.0, ts=10000.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "info"  # the storm is history; tuning advice is back


def test_low_volume_window_is_not_an_abort_storm(tmp_path):
    # an idle window: 6 requests, 1 aborted is 17% but says nothing
    store = _store(
        tmp_path,
        args={},
        engine_samples=[
            _engine("sglang_num_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_requests_total", 6.0, ts=2.0),
            _engine("sglang_num_aborted_requests_total", 0.0, ts=1.0),
            _engine("sglang_num_aborted_requests_total", 1.0, ts=2.0),
        ],
    )
    assert compute_advisories(store) == []


def test_kv_bound_concurrency_names_the_real_bottleneck(tmp_path):
    # token usage pegged while concurrency sits far under the cap: the pool is
    # the limit — the advice must be "raise mem-fraction", never "lower the cap"
    store = _store(
        tmp_path,
        args={"sglang_max_running_requests": 256, "sglang_mem_fraction_static": 0.75},
        engine_samples=[
            _engine("sglang_token_usage", 0.96, ts=1.0),
            _engine("sglang_num_running_reqs", 14.0, ts=1.0),
        ],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "KV-pool-bound" in advisory.message
    assert "do NOT lower" in advisory.message


class _StubReader:
    def __init__(self, groups_df, summary_df):
        self._groups = groups_df
        self._summary = summary_df

    def rollout_ids(self):
        return RolloutIds(train=[3], eval=[])

    def groups(self, rollout_id, *, evaluation=False):
        return self._groups

    def summary(self, rollout_id, *, evaluation=False):
        return self._summary


def _summary_df(truncated):
    return pl.DataFrame({"truncated": truncated})


def test_zero_std_groups_are_warning(tmp_path):
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.0, 0.0, 0.5], "zero_std": [True, True, False]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([False] * 8)))
    assert advisory.level == "warning"
    assert "zero reward std" in advisory.message


def test_all_zero_rewards_note_systematic_failure(tmp_path):
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.0, 0.0], "zero_std": [True, True]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([False] * 8)))
    assert "systematic failure" in advisory.message


def test_null_rewards_are_missing_data_not_zero_std(tmp_path):
    store = _store(tmp_path, args={}, engine_samples=[])
    groups = pl.DataFrame(
        {"reward_mean": [None, None], "zero_std": [False, False]},
        schema={"reward_mean": pl.Float64, "zero_std": pl.Boolean},
    )
    assert compute_advisories(store, _StubReader(groups, _summary_df([False] * 8))) == []


def test_truncation_warning_cites_per_turn_cap(tmp_path):
    store = _store(tmp_path, args={"rollout_max_response_len": 8192}, engine_samples=[])
    groups = pl.DataFrame({"reward_mean": [0.5], "zero_std": [False]})
    [advisory] = compute_advisories(store, _StubReader(groups, _summary_df([True] * 3 + [False] * 5)))
    assert advisory.level == "warning"
    assert "--rollout-max-response-len" in advisory.message
