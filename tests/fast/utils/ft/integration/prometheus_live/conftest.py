"""Fixtures for live Prometheus integration tests and backend conformance.

Provides a shared DynamicExporter (HTTP /metrics endpoint), a real
Prometheus server, and a parametrized ``backend`` fixture that runs
each test against both MiniPrometheus and real Prometheus — both
scraping the same exporter endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from prometheus_client import CollectorRegistry, Gauge, start_http_server

from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import TimeSeriesQueryProtocol

logger = logging.getLogger(__name__)

_PROMETHEUS_VERSION = "3.5.0"
_PROMETHEUS_TARBALL = f"prometheus-{_PROMETHEUS_VERSION}.linux-amd64.tar.gz"
_PROMETHEUS_URL = (
    f"https://github.com/prometheus/prometheus/releases/download/" f"v{_PROMETHEUS_VERSION}/{_PROMETHEUS_TARBALL}"
)
_BINARY_DIR = Path("/tmp/prometheus_test_binary")
_BINARY_PATH = _BINARY_DIR / "prometheus"

_STARTUP_TIMEOUT_SECONDS = 15
_DOWNLOAD_TIMEOUT_SECONDS = 120
_PORT_BIND_RETRIES = 5


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _download_prometheus() -> Path:
    if _BINARY_PATH.exists():
        logger.info("prometheus_binary already exists at %s", _BINARY_PATH)
        return _BINARY_PATH

    _BINARY_DIR.mkdir(parents=True, exist_ok=True)
    strip_prefix = f"prometheus-{_PROMETHEUS_VERSION}.linux-amd64/prometheus"
    cmd = (
        f"curl -sL --max-time {_DOWNLOAD_TIMEOUT_SECONDS} {_PROMETHEUS_URL} "
        f"| tar xz --strip-components=1 -C {_BINARY_DIR} {strip_prefix}"
    )
    logger.info("downloading prometheus: %s", cmd)
    subprocess.run(cmd, shell=True, check=True)

    if not _BINARY_PATH.exists():
        raise FileNotFoundError(f"Prometheus binary not found after download: {_BINARY_PATH}")

    _BINARY_PATH.chmod(0o755)
    return _BINARY_PATH


def _wait_for_prometheus(url: str, timeout: float = _STARTUP_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/api/v1/status/config", timeout=2.0)
            if resp.status_code == 200:
                logger.info("prometheus_ready at %s", url)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Prometheus did not become ready at {url} within {timeout}s")


# ---------------------------------------------------------------------------
# DynamicExporter — shared HTTP /metrics endpoint
# ---------------------------------------------------------------------------


class DynamicExporter:
    """Wraps a CollectorRegistry and creates prometheus_client Gauges on demand."""

    def __init__(self, registry: CollectorRegistry, port: int) -> None:
        self._registry = registry
        self.port = port
        self._gauges: dict[tuple[str, tuple[str, ...]], Gauge] = {}

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        label_names = tuple(sorted((labels or {}).keys()))
        key = (name, label_names)
        if key not in self._gauges:
            self._gauges[key] = Gauge(
                name,
                f"test gauge {name}",
                labelnames=list(label_names),
                registry=self._registry,
            )
        gauge = self._gauges[key]
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)


# ---------------------------------------------------------------------------
# MetricBackend abstraction
# ---------------------------------------------------------------------------


class MetricBackend(ABC):
    @property
    @abstractmethod
    def store(self) -> TimeSeriesQueryProtocol: ...

    @abstractmethod
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    @abstractmethod
    async def flush(self) -> None: ...


class MiniBackend(MetricBackend):
    def __init__(self, exporter: DynamicExporter) -> None:
        self._exporter = exporter
        self._store = MiniPrometheus()
        self._store.add_scrape_target(
            target_id="test",
            address=f"http://127.0.0.1:{exporter.port}",
        )

    @property
    def store(self) -> MiniPrometheus:
        return self._store

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._exporter.set_gauge(name=name, value=value, labels=labels)

    async def flush(self) -> None:
        await self._store.scrape_once()

    async def teardown(self) -> None:
        await self._store.stop()


class LiveBackend(MetricBackend):
    _FLUSH_POLL_INTERVAL: float = 0.3
    _FLUSH_TIMEOUT: float = 10.0

    def __init__(self, exporter: DynamicExporter, prometheus_url: str) -> None:
        self._exporter = exporter
        self._client = PrometheusClient(url=prometheus_url, range_query_step_seconds=1)
        self._pending_metric: str | None = None
        self._pending_value: float | None = None
        self._pending_labels: dict[str, str] | None = None

    @property
    def store(self) -> PrometheusClient:
        return self._client

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._exporter.set_gauge(name=name, value=value, labels=labels)
        self._pending_metric = name
        self._pending_value = value
        self._pending_labels = labels

    async def flush(self) -> None:
        """Poll until Prometheus has scraped and returns the expected value."""
        if self._pending_metric is None:
            await asyncio.sleep(2)
            return

        metric = self._pending_metric
        expected = self._pending_value
        labels = self._pending_labels
        deadline = time.monotonic() + self._FLUSH_TIMEOUT
        while time.monotonic() < deadline:
            df = self._client.query_latest(metric, label_filters=labels)
            if not df.is_empty() and df["value"][0] == expected:
                return
            await asyncio.sleep(self._FLUSH_POLL_INTERVAL)

        raise TimeoutError(
            f"Prometheus did not return expected value {expected} for "
            f"metric '{metric}' (labels={labels}) within {self._FLUSH_TIMEOUT}s"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def prometheus_binary() -> Path:
    if platform.system() != "Linux":
        pytest.skip("Live Prometheus tests require Linux")

    return _download_prometheus()


@pytest.fixture(scope="module")
def dynamic_exporter() -> Iterator[DynamicExporter]:
    registry = CollectorRegistry()
    for attempt in range(_PORT_BIND_RETRIES):
        port = _find_free_port()
        try:
            httpd, _thread = start_http_server(port=port, registry=registry)
            break
        except OSError:
            if attempt == _PORT_BIND_RETRIES - 1:
                raise
            logger.warning("Port %d already in use, retrying with a new port", port)
    else:
        raise OSError("Failed to bind exporter HTTP server after retries")

    yield DynamicExporter(registry=registry, port=port)

    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def prometheus_server(
    prometheus_binary: Path,
    dynamic_exporter: DynamicExporter,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    data_dir = tmp_path_factory.mktemp("prom_data")

    config_path = data_dir / "prometheus.yml"
    config_path.write_text(
        f"""\
global:
  scrape_interval: 1s
  evaluation_interval: 1s

scrape_configs:
  - job_name: test_exporter
    static_configs:
      - targets: ["127.0.0.1:{dynamic_exporter.port}"]
"""
    )

    proc = None
    prom_url = ""
    for attempt in range(_PORT_BIND_RETRIES):
        prom_port = _find_free_port()
        proc = subprocess.Popen(
            [
                str(prometheus_binary),
                f"--config.file={config_path}",
                f"--storage.tsdb.path={data_dir / 'tsdb'}",
                f"--web.listen-address=127.0.0.1:{prom_port}",
                "--storage.tsdb.retention.time=5m",
                "--log.level=warn",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        prom_url = f"http://127.0.0.1:{prom_port}"
        try:
            _wait_for_prometheus(prom_url)
            break
        except TimeoutError:
            proc.terminate()
            proc.wait(timeout=5)
            if attempt == _PORT_BIND_RETRIES - 1:
                raise
            logger.warning("Prometheus failed to start on port %d, retrying", prom_port)
    assert proc is not None

    yield prom_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(params=["mini", "live"])
async def backend(
    request: pytest.FixtureRequest,
    dynamic_exporter: DynamicExporter,
) -> AsyncIterator[MetricBackend]:
    if request.param == "mini":
        be = MiniBackend(exporter=dynamic_exporter)
        yield be
        await be.teardown()
    else:
        prometheus_server_url: str = request.getfixturevalue("prometheus_server")
        yield LiveBackend(
            exporter=dynamic_exporter,
            prometheus_url=prometheus_server_url,
        )
