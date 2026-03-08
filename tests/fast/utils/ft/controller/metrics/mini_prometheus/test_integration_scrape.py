"""Integration tests: MiniPrometheus scraping real prometheus_client exporters."""

from __future__ import annotations

import logging
import socket
from threading import Thread
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from prometheus_client.registry import CollectorRegistry

from tests.fast.utils.ft.utils.metric_injectors import make_fake_metric_store


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_exporter(
    metrics: list[tuple[str, dict[str, str], float]] | None = None,
    port: int | None = None,
) -> tuple[int, CollectorRegistry]:
    import time

    from prometheus_client import Gauge, start_http_server
    from prometheus_client.registry import CollectorRegistry

    if port is None:
        port = _find_free_port()

    registry = CollectorRegistry()

    if metrics is None:
        metrics = [
            ("gpu_temperature_celsius", {"gpu": "0"}, 75.0),
            ("gpu_temperature_celsius", {"gpu": "1"}, 82.0),
            ("gpu_available", {"gpu": "0"}, 1.0),
            ("gpu_available", {"gpu": "1"}, 1.0),
        ]

    gauges: dict[str, Gauge] = {}
    for name, labels, value in metrics:
        if name not in gauges:
            label_keys = sorted(labels.keys()) if labels else []
            gauges[name] = Gauge(name, name, label_keys, registry=registry)
        if labels:
            gauges[name].labels(**labels).set(value)
        else:
            gauges[name].set(value)

    Thread(
        target=lambda: start_http_server(port=port, registry=registry),
        daemon=True,
    ).start()
    time.sleep(0.5)

    return port, registry


class TestMiniPrometheusScrapeReal:
    async def test_scrape_single_exporter(self) -> None:
        port, _ = _start_exporter()

        store = make_fake_metric_store()
        store.add_scrape_target(
            target_id="node-0",
            address=f"http://localhost:{port}",
        )

        await store.scrape_once()

        df = store.query_latest("gpu_temperature_celsius")
        assert len(df) >= 2
        values = sorted(df["value"].to_list())
        assert 75.0 in values
        assert 82.0 in values

    async def test_scrape_updates_values(self) -> None:
        from prometheus_client import Gauge
        from prometheus_client.registry import CollectorRegistry

        port = _find_free_port()
        registry = CollectorRegistry()
        temp = Gauge("test_temp", "test", ["gpu"], registry=registry)
        temp.labels(gpu="0").set(60.0)

        import time
        from threading import Thread

        from prometheus_client import start_http_server

        Thread(
            target=lambda: start_http_server(port=port, registry=registry),
            daemon=True,
        ).start()
        time.sleep(0.5)

        store = make_fake_metric_store()
        store.add_scrape_target(target_id="node-0", address=f"http://localhost:{port}")

        await store.scrape_once()
        df1 = store.query_latest("test_temp")
        assert df1["value"][0] == 60.0

        temp.labels(gpu="0").set(90.0)
        await store.scrape_once()
        df2 = store.query_latest("test_temp")
        assert df2["value"][0] == 90.0

    async def test_scrape_unreachable_target_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        store = make_fake_metric_store()
        store.add_scrape_target(
            target_id="bad-node",
            address="http://localhost:19999",
        )

        with caplog.at_level(logging.WARNING):
            await store.scrape_once()

        assert any("Failed to scrape" in r.message for r in caplog.records)

    async def test_scrape_multiple_exporters(self) -> None:
        port1, _ = _start_exporter(metrics=[("node_metric", {"type": "a"}, 1.0)])
        port2, _ = _start_exporter(metrics=[("node_metric", {"type": "a"}, 2.0)])

        store = make_fake_metric_store()
        store.add_scrape_target(target_id="node-0", address=f"http://localhost:{port1}")
        store.add_scrape_target(target_id="node-1", address=f"http://localhost:{port2}")

        await store.scrape_once()

        df = store.query_latest("node_metric")
        assert len(df) == 2
        node_ids = sorted(df["node_id"].to_list())
        assert node_ids == ["node-0", "node-1"]

    async def test_scrape_reuses_httpx_client(self) -> None:
        """ScrapeLoop must reuse the same httpx.AsyncClient across calls
        to avoid per-request connection setup overhead."""
        port, _ = _start_exporter()

        store = make_fake_metric_store()
        store.add_scrape_target(
            target_id="node-0",
            address=f"http://localhost:{port}",
        )

        await store.scrape_once()
        client_after_first = store._scrape_loop._client
        assert client_after_first is not None

        await store.scrape_once()
        client_after_second = store._scrape_loop._client
        assert client_after_second is client_after_first

    async def test_stop_closes_httpx_client(self) -> None:
        """ScrapeLoop.stop() must close the httpx client and set it to None."""
        port, _ = _start_exporter()

        store = make_fake_metric_store()
        store.add_scrape_target(
            target_id="node-0",
            address=f"http://localhost:{port}",
        )

        await store.scrape_once()
        assert store._scrape_loop._client is not None

        await store._scrape_loop.stop()
        assert store._scrape_loop._client is None

    async def test_scrape_bad_target_doesnt_affect_good(self) -> None:
        port, _ = _start_exporter(metrics=[("good_metric", {}, 42.0)])

        store = make_fake_metric_store()
        store.add_scrape_target(target_id="bad-node", address="http://localhost:19999")
        store.add_scrape_target(target_id="good-node", address=f"http://localhost:{port}")

        await store.scrape_once()

        df = store.query_latest("good_metric")
        assert len(df) == 1
        assert df["node_id"][0] == "good-node"
