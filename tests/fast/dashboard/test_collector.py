import logging
import time
from pathlib import Path

from tests.fast.dashboard.dummy_telemetry import BASE_TS, dump_dummy_telemetry

from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.sglang_scraper import ScrapeMode
from miles.dashboard.store import (
    EngineInfo,
    GpuSample,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Role,
    Stream,
    TopologySnapshot,
)

ROUTER_FIXTURE = (Path(__file__).parent / "fixtures" / "engine_metrics_router.txt").read_text()


def make_collector(tmp_path, **kwargs) -> DashboardCollector:
    config = kwargs.pop("config", None) or CollectorConfig(
        dashboard_dir=str(tmp_path / "dashboard"), run_name="collector-test", start_ts=1.0
    )
    return DashboardCollector(config, **kwargs)


def test_collector_satisfies_dummy_telemetry_contract(tmp_path):
    """Records routed through the collector's RPC surface must be interpreted
    by the timeline queries exactly like the fixture's direct writes."""
    dump_dummy_telemetry(tmp_path / "direct")
    reference = MetricStore.load(tmp_path / "direct" / "dashboard")

    collector = make_collector(
        tmp_path,
        config=CollectorConfig(
            # start_ts must match the fixture's: the synthesized initialize
            # band in phases_by_lane derives from meta.start_ts
            dashboard_dir=str(tmp_path / "via_collector" / "dashboard"),
            run_name="dummy-telemetry",
            start_ts=BASE_TS,
        ),
    )
    for record in reference.records[Stream.METRICS]:
        collector.push_metrics(record)
    collector.push_phases(list(reference.records[Stream.PHASES]))
    for snapshot in reference.records[Stream.TOPOLOGY]:
        collector.update_topology(snapshot)
    by_node: dict[str, list[GpuSample]] = {}
    for sample in reference.iter_records(Stream.GPU_UTIL):
        by_node.setdefault(sample.node, []).append(sample)
    for node, batch in by_node.items():
        collector.push_gpu_samples(node, batch)
    for sample in reference.iter_records(Stream.ENGINE_SERIES):
        collector._append(sample)  # the scraper sink path
    collector.flush()

    replayed = MetricStore.load(tmp_path / "via_collector" / "dashboard")
    assert replayed.lanes() == reference.lanes()
    assert replayed.topology_windows() == reference.topology_windows()
    assert replayed.phases_by_lane() == reference.phases_by_lane()
    assert replayed.gpu_series() == reference.gpu_series()
    assert replayed.engine_series("sglang_num_running_reqs") == reference.engine_series("sglang_num_running_reqs")
    assert replayed.bubbles() == reference.bubbles()
    assert replayed.meta.run_name == reference.meta.run_name


def _engine(addr: str) -> EngineInfo:
    return EngineInfo(addr=addr, worker_type="regular", engine_rank=0, gpus=[["n", 0]], gpu_uuids=[None])


def test_topology_dedup_keeps_changes_only(tmp_path):
    collector = make_collector(tmp_path)
    collector.update_topology(TopologySnapshot(ts=1.0, engines=[_engine("http://a:1")]))
    collector.update_topology(TopologySnapshot(ts=2.0, engines=[_engine("http://a:1")]))  # same engines
    collector.update_topology(TopologySnapshot(ts=3.0, engines=[_engine("http://a:2")]))  # restart
    collector.flush()

    loaded = MetricStore.load(collector.config.dashboard_dir)
    assert [w["t0"] for w in loaded.topology_windows()] == [1.0, 3.0]


def test_flush_thread_persists_periodically(tmp_path):
    config = CollectorConfig(
        dashboard_dir=str(tmp_path / "dashboard"), run_name="r", start_ts=0.0, flush_interval_seconds=0.05
    )
    collector = DashboardCollector(config)
    collector.start()
    collector.push_metrics(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    time.sleep(0.2)
    try:
        assert len(MetricStore.load(config.dashboard_dir).records[Stream.METRICS]) == 1
    finally:
        collector.shutdown()


def test_shutdown_flushes_and_stops_scraper(tmp_path):
    collector = make_collector(tmp_path, scraper_http_get=lambda url, timeout: ROUTER_FIXTURE)
    collector.set_router("http://router:3000", use_miles_router=False)
    assert collector._scraper is not None
    collector.push_metrics(MetricsRecord(ts=1.0, step_key="rollout/step", step=0, metrics={"a": 1}))
    collector.shutdown()

    assert collector._scraper._stop_event.is_set()
    loaded = MetricStore.load(collector.config.dashboard_dir)
    assert len(loaded.records[Stream.METRICS]) == 1


def test_scraper_mode_resolution_and_repoint(tmp_path):
    collector = make_collector(tmp_path, scraper_http_get=lambda url, timeout: ROUTER_FIXTURE)
    collector.set_router("http://router:3000", use_miles_router=False)
    assert collector._scraper.mode == ScrapeMode.ROUTER
    first = collector._scraper

    collector.set_router("http://router:3000", use_miles_router=False)  # no-op
    assert collector._scraper is first

    collector.update_topology(TopologySnapshot(ts=1.0, engines=[_engine("http://e:1")]))
    collector.set_router("http://router2:3000", use_miles_router=True)  # router restarted, miles router now
    assert collector._scraper is not first
    assert first._stop_event.is_set()
    assert collector._scraper.mode == ScrapeMode.DIRECT
    assert collector._scraper.engine_addrs() == ["http://e:1"]
    collector.shutdown()


def test_disk_failure_is_loud_and_buffers_stay_bounded(tmp_path, monkeypatch, caplog):
    collector = make_collector(tmp_path)
    monkeypatch.setattr(DashboardCollector, "MAX_BUFFERED_PER_STREAM", 100)
    monkeypatch.setattr(MetricStore, "flush", lambda self: (_ for _ in ()).throw(OSError("no space left")))

    with caplog.at_level(logging.ERROR):
        for step in range(150):
            collector.push_metrics(MetricsRecord(ts=float(step), step_key="rollout/step", step=step, metrics={}))
        collector.flush()

    assert collector._store.buffered_count(Stream.METRICS) <= 105  # bounded despite dead disk
    messages = [r.message for r in caplog.records]
    assert any("failed" in m for m in messages)  # the OSError is loud
    assert any("dropped" in m for m in messages)  # so is the data loss


def test_prometheus_forwarding_snapshot(tmp_path):
    updates: list[dict] = []

    class FakeRemote:
        def remote(self, snapshot):
            updates.append(snapshot)

    class FakeHandle:
        update = FakeRemote()

    config = CollectorConfig(
        dashboard_dir=str(tmp_path / "dashboard"), run_name="r", start_ts=0.0, forward_prometheus=True
    )
    collector = DashboardCollector(config, prometheus_handle_factory=lambda: FakeHandle())
    collector.push_gpu_samples(
        "10.0.0.1", [GpuSample(ts=1.0, node="10.0.0.1", gpu=0, util=87, mem_mb=1000, power_w=600)]
    )
    collector.push_phases(
        [PhaseEvent(name="actor_train", t0=1.0, t1=61.0, node="10.0.0.1", gpus=[0], rank=0, role=Role.TRAIN)]
    )
    collector.flush()
    collector.flush()  # latest-value cache must survive the first flush

    assert len(updates) == 2
    snapshot = updates[-1]
    assert snapshot["dashboard/gpu_10_0_0_1_0_util"] == 87.0
    assert snapshot["dashboard/phase_actor_train_seconds"] == 60.0
    assert all("." not in k and ":" not in k for k in snapshot)


def test_forwarding_disabled_by_default(tmp_path):
    called = []
    collector = make_collector(tmp_path, prometheus_handle_factory=lambda: called.append(1))
    collector.flush()
    assert called == []
