from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from tests.fast.utils.ft.utils.metric_injectors import get_sample_value as _scrape_value

from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.agents.types import CounterSample, GaugeSample


@pytest.fixture
def exporter() -> Iterator[PrometheusExporter]:
    exp = PrometheusExporter()
    yield exp
    exp.shutdown()


class TestUpdateGauge:
    def test_gauge_set(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=72.0),
            ]
        )
        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 72.0

    def test_gauge_overwrite(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=72.0),
            ]
        )
        exporter.update_metrics(
            [
                GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=85.0),
            ]
        )
        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 85.0

    def test_gauge_without_labels(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                GaugeSample(name="uptime_seconds", labels={}, value=3600.0),
            ]
        )
        assert _scrape_value(exporter.registry, "uptime_seconds") == 3600.0


class TestUpdateCounter:
    def test_counter_increment(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                CounterSample(name="xid_count_total", labels={"gpu": "0"}, delta=3.0),
            ]
        )
        assert _scrape_value(exporter.registry, "xid_count_total", {"gpu": "0"}) == 3.0

    def test_counter_accumulates(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                CounterSample(name="err_total", labels={}, delta=1.0),
            ]
        )
        exporter.update_metrics(
            [
                CounterSample(name="err_total", labels={}, delta=2.0),
            ]
        )
        assert _scrape_value(exporter.registry, "err_total") == 3.0

    def test_counter_zero_delta_suppressed(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                CounterSample(name="event_total", labels={}, delta=5.0),
            ]
        )
        exporter.update_metrics(
            [
                CounterSample(name="event_total", labels={}, delta=0.0),
            ]
        )
        assert _scrape_value(exporter.registry, "event_total") == 5.0

    def test_counter_total_suffix_stripped(self, exporter: PrometheusExporter) -> None:
        """prometheus_client auto-appends _total for Counters, so we strip it from the name."""
        exporter.update_metrics(
            [
                CounterSample(name="xid_count_total", labels={}, delta=1.0),
            ]
        )
        assert _scrape_value(exporter.registry, "xid_count_total") == 1.0
        assert _scrape_value(exporter.registry, "xid_count_total_total") is None

    def test_counter_without_total_suffix(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                CounterSample(name="errors", labels={}, delta=2.0),
            ]
        )
        assert _scrape_value(exporter.registry, "errors_total") == 2.0


class TestCounterNonPositiveDeltaLogged:
    def test_zero_delta_logs_debug(
        self, exporter: PrometheusExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Previously non-positive counter deltas were silently dropped.
        Now they emit a debug log for observability."""
        with caplog.at_level(logging.DEBUG, logger="miles.utils.ft.agents.metrics.prometheus_exporter"):
            exporter.update_metrics(
                [CounterSample(name="event_total", labels={}, delta=0.0)]
            )

        assert "counter_delta_non_positive" in caplog.text
        assert "event_total" in caplog.text

    def test_negative_delta_logs_debug(
        self, exporter: PrometheusExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="miles.utils.ft.agents.metrics.prometheus_exporter"):
            exporter.update_metrics(
                [CounterSample(name="err_total", labels={}, delta=-1.0)]
            )

        assert "counter_delta_non_positive" in caplog.text


class TestGetOrCreateCache:
    def test_same_metric_reuses_instance(self, exporter: PrometheusExporter) -> None:
        sample = GaugeSample(name="temp", labels={"gpu": "0"}, value=70.0)
        exporter.update_metrics([sample])
        exporter.update_metrics([sample])
        assert len(exporter._gauges) == 1

    def test_different_names_cached_separately(self, exporter: PrometheusExporter) -> None:
        exporter.update_metrics(
            [
                GaugeSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0),
                GaugeSample(name="disk_temp", labels={"device": "sda"}, value=40.0),
            ]
        )
        assert len(exporter._gauges) == 2
        assert _scrape_value(exporter.registry, "gpu_temp", {"gpu": "0"}) == 70.0
        assert _scrape_value(exporter.registry, "disk_temp", {"device": "sda"}) == 40.0


class TestLifecycle:
    def test_get_address_contains_real_ip(self, exporter: PrometheusExporter) -> None:
        address = exporter.get_address()
        assert address.startswith("http://")
        port_str = address.rsplit(":", 1)[-1]
        assert port_str.isdigit() and int(port_str) > 0

    def test_shutdown(self, exporter: PrometheusExporter) -> None:
        exporter.shutdown()

    def test_double_shutdown_does_not_raise(self, exporter: PrometheusExporter) -> None:
        exporter.shutdown()
        exporter.shutdown()
