from miles.dashboard.advisory import compute_advisories
from miles.dashboard.store import EngineSample, Meta, MetricStore


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


def _dp_engine(rank: str, value: float, ts: float = 1.0) -> EngineSample:
    return EngineSample(
        ts=ts, addr="http://n:1", metric="sglang_num_running_reqs", labels={"dp_rank": rank}, value=value
    )


def _imbalanced(tmp_path, args: dict) -> str:
    store = _store(
        tmp_path,
        args=args,
        engine_samples=[_dp_engine("0", 11.0), _dp_engine("1", 0.5), _dp_engine("2", 0.0), _dp_engine("3", 1.0)],
    )
    [advisory] = compute_advisories(store)
    assert advisory.level == "warning"
    assert "dp ranks imbalanced" in advisory.message
    return advisory.message


def test_dp_imbalance_names_the_knob_that_applies(tmp_path):
    assert "--router-dp-aware" in _imbalanced(tmp_path / "a", {})
    assert "--sglang-load-balance-method" in _imbalanced(tmp_path / "b", {"use_miles_router": True})
    assert "--router-policy (cache_aware)" in _imbalanced(
        tmp_path / "c", {"router_dp_aware": True, "router_policy": "cache_aware"}
    )
    # miles overrides router_policy with sglang_router_policy when both are set
    assert "--router-assignment-mode (random)" in _imbalanced(
        tmp_path / "d",
        {
            "router_dp_aware": True,
            "router_policy": "cache_aware",
            "sglang_router_policy": "manual",
            "router_assignment_mode": "random",
        },
    )


def test_dp_balanced_not_flagged(tmp_path):
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_dp_engine("0", 5.0), _dp_engine("1", 4.0), _dp_engine("2", 6.0), _dp_engine("3", 5.0)],
    )
    assert compute_advisories(store) == []


def test_dp_idle_engine_not_flagged(tmp_path):
    # everything near zero (drained engine): no load, no imbalance signal
    store = _store(
        tmp_path,
        args={},
        engine_samples=[_dp_engine("0", 0.5), _dp_engine("1", 0.0)],
    )
    assert compute_advisories(store) == []
