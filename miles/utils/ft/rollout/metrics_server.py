from __future__ import annotations

import logging

from aiohttp import web
from prometheus_client import CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)


class MetricsServer:
    """Lightweight async HTTP server exposing /metrics from a CollectorRegistry."""

    def __init__(self, *, registry: CollectorRegistry, port: int = 0) -> None:
        self._registry = registry
        self._port = port
        self._runner: web.AppRunner | None = None
        self._actual_port: int | None = None

    @property
    def address(self) -> str:
        if self._actual_port is None:
            raise RuntimeError("MetricsServer not started")
        return f"http://localhost:{self._actual_port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host="0.0.0.0", port=self._port)
        await site.start()

        assert self._runner.addresses
        self._actual_port = self._runner.addresses[0][1]
        logger.info("metrics_server_started port=%d", self._actual_port)

    async def shutdown(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._actual_port = None

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        output = generate_latest(self._registry)
        return web.Response(body=output, content_type="text/plain; charset=utf-8")
