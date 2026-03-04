"""Integration tests: MiniPrometheus scraping real prometheus_client exporters."""

import logging
import socket
from datetime import timedelta
from threading import Thread

import pytest

from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_exporter(port: int) -> Thread:
    from prometheus_client import Gauge, start_http_server
    from prometheus_client.registry import CollectorRegistry

    registry = CollectorRegistry()
    gpu_temp = Gauge(
        "gpu_temperature_celsius",
        "GPU temperature",
        ["gpu"],
        registry=registry,
    )
    gpu_temp.labels(gpu="0").set(75.0)
    gpu_temp.labels(gpu="1").set(82.0)

    gpu_avail = Gauge(
        "gpu_available",
        "GPU availability",
        ["gpu"],
        registry=registry,
    )
    gpu_avail.labels(gpu="0").set(1.0)
    gpu_avail.labels(gpu="1").set(1.0)

    def _serve() -> None:
        start_http_server(port=port, registry=registry)

    thread = Thread(target=_serve, daemon=True)
    thread.start()

    import time
    time.sleep(0.5)

    return thread


@pytest.mark.asyncio
class TestMiniPrometheusScrapeReal:
    async def test_scrape_single_exporter(self) -> None:
        port = _find_free_port()
        _start_exporter(port)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
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

        from prometheus_client import start_http_server
        from threading import Thread
        import time

        Thread(
            target=lambda: start_http_server(port=port, registry=registry),
            daemon=True,
        ).start()
        time.sleep(0.5)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        store.add_scrape_target(target_id="node-0", address=f"http://localhost:{port}")

        await store.scrape_once()
        df1 = store.query_latest("test_temp")
        assert df1["value"][0] == 60.0

        temp.labels(gpu="0").set(90.0)
        await store.scrape_once()
        df2 = store.query_latest("test_temp")
        assert df2["value"][0] == 90.0

    async def test_scrape_unreachable_target_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        store.add_scrape_target(
            target_id="bad-node",
            address="http://localhost:19999",
        )

        with caplog.at_level(logging.WARNING):
            await store.scrape_once()

        assert any("Failed to scrape" in r.message for r in caplog.records)

    async def test_scrape_multiple_exporters(self) -> None:
        from prometheus_client import Gauge
        from prometheus_client.registry import CollectorRegistry
        from prometheus_client import start_http_server
        from threading import Thread
        import time

        port1 = _find_free_port()
        reg1 = CollectorRegistry()
        Gauge("node_metric", "m", ["type"], registry=reg1).labels(type="a").set(1.0)
        Thread(
            target=lambda: start_http_server(port=port1, registry=reg1),
            daemon=True,
        ).start()

        port2 = _find_free_port()
        reg2 = CollectorRegistry()
        Gauge("node_metric", "m", ["type"], registry=reg2).labels(type="a").set(2.0)
        Thread(
            target=lambda: start_http_server(port=port2, registry=reg2),
            daemon=True,
        ).start()

        time.sleep(0.5)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
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
        port = _find_free_port()
        _start_exporter(port)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
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
        port = _find_free_port()
        _start_exporter(port)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        store.add_scrape_target(
            target_id="node-0",
            address=f"http://localhost:{port}",
        )

        await store.scrape_once()
        assert store._scrape_loop._client is not None

        await store._scrape_loop.stop()
        assert store._scrape_loop._client is None

    async def test_scrape_bad_target_doesnt_affect_good(self) -> None:
        from prometheus_client import Gauge
        from prometheus_client.registry import CollectorRegistry
        from prometheus_client import start_http_server
        from threading import Thread
        import time

        port = _find_free_port()
        reg = CollectorRegistry()
        Gauge("good_metric", "m", registry=reg).set(42.0)
        Thread(
            target=lambda: start_http_server(port=port, registry=reg),
            daemon=True,
        ).start()
        time.sleep(0.5)

        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        store.add_scrape_target(target_id="bad-node", address="http://localhost:19999")
        store.add_scrape_target(target_id="good-node", address=f"http://localhost:{port}")

        await store.scrape_once()

        df = store.query_latest("good_metric")
        assert len(df) == 1
        assert df["node_id"][0] == "good-node"
