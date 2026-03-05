from __future__ import annotations

from prometheus_client import CollectorRegistry

from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.models import MetricSample


def _make_exporter() -> PrometheusExporter:
    return PrometheusExporter()


def _scrape_value(
    registry: CollectorRegistry,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    for family in registry.collect():
        for sample in family.samples:
            if sample.name != metric_name:
                continue
            if labels is not None and dict(sample.labels) != labels:
                continue
            return sample.value
    return None


class TestUpdateGauge:
    def test_gauge_set(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=72.0),
        ])

        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 72.0
        exporter.shutdown()

    def test_gauge_overwrite(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=72.0),
        ])
        exporter.update_metrics([
            MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=85.0),
        ])

        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 85.0
        exporter.shutdown()

    def test_gauge_without_labels(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="uptime_seconds", labels={}, value=3600.0),
        ])

        assert _scrape_value(exporter.registry, "uptime_seconds") == 3600.0
        exporter.shutdown()


class TestUpdateCounter:
    def test_counter_increment(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="xid_count_total", labels={"gpu": "0"}, value=3.0, metric_type="counter"),
        ])

        assert _scrape_value(exporter.registry, "xid_count_total", {"gpu": "0"}) == 3.0
        exporter.shutdown()

    def test_counter_accumulates(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="err_total", labels={}, value=1.0, metric_type="counter"),
        ])
        exporter.update_metrics([
            MetricSample(name="err_total", labels={}, value=2.0, metric_type="counter"),
        ])

        assert _scrape_value(exporter.registry, "err_total") == 3.0
        exporter.shutdown()

    def test_counter_zero_value_suppressed(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="event_total", labels={}, value=5.0, metric_type="counter"),
        ])
        exporter.update_metrics([
            MetricSample(name="event_total", labels={}, value=0.0, metric_type="counter"),
        ])

        assert _scrape_value(exporter.registry, "event_total") == 5.0
        exporter.shutdown()

    def test_counter_total_suffix_stripped(self) -> None:
        """prometheus_client auto-appends _total for Counters, so we strip it from the name."""
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="xid_count_total", labels={}, value=1.0, metric_type="counter"),
        ])

        assert _scrape_value(exporter.registry, "xid_count_total") == 1.0
        assert _scrape_value(exporter.registry, "xid_count_total_total") is None
        exporter.shutdown()

    def test_counter_without_total_suffix(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="errors", labels={}, value=2.0, metric_type="counter"),
        ])

        assert _scrape_value(exporter.registry, "errors_total") == 2.0
        exporter.shutdown()


class TestGetOrCreateCache:
    def test_same_metric_reuses_instance(self) -> None:
        exporter = _make_exporter()
        sample = MetricSample(name="temp", labels={"gpu": "0"}, value=70.0)
        exporter.update_metrics([sample])
        exporter.update_metrics([sample])

        assert len(exporter._gauges) == 1
        exporter.shutdown()

    def test_different_names_cached_separately(self) -> None:
        exporter = _make_exporter()
        exporter.update_metrics([
            MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0),
            MetricSample(name="disk_temp", labels={"device": "sda"}, value=40.0),
        ])

        assert len(exporter._gauges) == 2
        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 70.0
        assert _scrape_value(exporter.registry, "disk_temp", {"device": "sda"}) == 40.0
        exporter.shutdown()


class TestLifecycle:
    def test_get_address(self) -> None:
        exporter = _make_exporter()
        assert exporter.get_address().startswith("http://localhost:")
        exporter.shutdown()

    def test_shutdown(self) -> None:
        exporter = _make_exporter()
        exporter.shutdown()

    def test_double_shutdown_does_not_raise(self) -> None:
        exporter = _make_exporter()
        exporter.shutdown()
        exporter.shutdown()
