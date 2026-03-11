from __future__ import annotations

import aiohttp
import pytest
from prometheus_client import CollectorRegistry, Gauge

from miles.utils.ft.rollout.metrics_server import MetricsServer


class TestMetricsServerStartAndServe:
    @pytest.mark.anyio
    async def test_get_metrics_returns_prometheus_format(self) -> None:
        registry = CollectorRegistry()
        gauge = Gauge("test_metric", "for testing", registry=registry)
        gauge.set(42.0)
        server = MetricsServer(registry=registry, port=0)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server.address}/metrics") as resp:
                    assert resp.status == 200
                    body = await resp.text()
                    assert "test_metric 42.0" in body
        finally:
            await server.shutdown()


class TestDynamicPortAllocation:
    @pytest.mark.anyio
    async def test_port_zero_assigns_actual_port(self) -> None:
        server = MetricsServer(registry=CollectorRegistry(), port=0)
        await server.start()

        try:
            assert "http://localhost:" in server.address
            port = int(server.address.split(":")[-1])
            assert port > 0
        finally:
            await server.shutdown()

    def test_address_raises_before_start(self) -> None:
        server = MetricsServer(registry=CollectorRegistry(), port=0)

        with pytest.raises(RuntimeError, match="not started"):
            _ = server.address


class TestShutdownCleanup:
    @pytest.mark.anyio
    async def test_shutdown_releases_port(self) -> None:
        server = MetricsServer(registry=CollectorRegistry(), port=0)
        await server.start()
        address = server.address
        await server.shutdown()

        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientConnectorError):
                async with session.get(f"{address}/metrics"):
                    pass
