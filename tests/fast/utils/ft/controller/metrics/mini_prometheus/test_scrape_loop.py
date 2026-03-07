"""Tests for ScrapeLoop — target management, client lifecycle, stop edge cases."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.scrape_loop import ScrapeLoop, parse_prometheus_text
from miles.utils.ft.models.metrics import GaugeSample


class _FakeStore:
    """Records ingest_samples calls."""

    def __init__(self) -> None:
        self.ingested: list[tuple[str, list[GaugeSample]]] = []

    def ingest_samples(self, target_id: str, samples: list[GaugeSample]) -> None:
        self.ingested.append((target_id, samples))


@pytest.fixture
def scrape_loop() -> ScrapeLoop:
    return ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)


# ===================================================================
# Target management
# ===================================================================


class TestTargetManagement:
    def test_add_target(self, scrape_loop: ScrapeLoop) -> None:
        scrape_loop.add_target("t1", "http://host:9090")

        assert scrape_loop.targets == {"t1": "http://host:9090"}

    def test_remove_target(self, scrape_loop: ScrapeLoop) -> None:
        scrape_loop.add_target("t1", "http://host:9090")

        scrape_loop.remove_target("t1")

        assert scrape_loop.targets == {}

    def test_remove_nonexistent_target_is_safe(self, scrape_loop: ScrapeLoop) -> None:
        scrape_loop.remove_target("nonexistent")

    def test_targets_property_returns_copy(self, scrape_loop: ScrapeLoop) -> None:
        scrape_loop.add_target("t1", "http://host:9090")

        targets = scrape_loop.targets
        targets["t2"] = "http://other:9090"

        assert "t2" not in scrape_loop.targets


# ===================================================================
# scrape_once
# ===================================================================


class TestScrapeOnce:
    @pytest.mark.anyio
    async def test_empty_targets_is_noop(self, scrape_loop: ScrapeLoop) -> None:
        await scrape_loop.scrape_once()

        store: _FakeStore = scrape_loop._store  # type: ignore[assignment]
        assert len(store.ingested) == 0


# ===================================================================
# _ensure_client
# ===================================================================


class TestEnsureClient:
    def test_creates_client_when_none(self, scrape_loop: ScrapeLoop) -> None:
        assert scrape_loop._client is None

        client = scrape_loop._ensure_client()

        assert client is not None
        assert not client.is_closed

    def test_reuses_existing_open_client(self, scrape_loop: ScrapeLoop) -> None:
        client1 = scrape_loop._ensure_client()
        client2 = scrape_loop._ensure_client()

        assert client1 is client2

    @pytest.mark.anyio
    async def test_recreates_client_after_close(self, scrape_loop: ScrapeLoop) -> None:
        client1 = scrape_loop._ensure_client()
        await client1.aclose()
        assert client1.is_closed

        client2 = scrape_loop._ensure_client()

        assert client2 is not client1
        assert not client2.is_closed
        await client2.aclose()


# ===================================================================
# stop
# ===================================================================


class TestStop:
    @pytest.mark.anyio
    async def test_stop_sets_running_false(self, scrape_loop: ScrapeLoop) -> None:
        scrape_loop._running = True

        await scrape_loop.stop()

        assert scrape_loop._running is False

    @pytest.mark.anyio
    async def test_stop_closes_client(self, scrape_loop: ScrapeLoop) -> None:
        client = scrape_loop._ensure_client()
        assert not client.is_closed

        await scrape_loop.stop()

        assert client.is_closed
        assert scrape_loop._client is None

    @pytest.mark.anyio
    async def test_stop_when_client_is_none(self, scrape_loop: ScrapeLoop) -> None:
        await scrape_loop.stop()

    @pytest.mark.anyio
    async def test_stop_when_client_already_closed(self, scrape_loop: ScrapeLoop) -> None:
        client = scrape_loop._ensure_client()
        await client.aclose()

        await scrape_loop.stop()


# ===================================================================
# parse_prometheus_text
# ===================================================================


class TestParsePrometheusText:
    """Smoke tests for the parse_prometheus_text wrapper.

    Detailed parsing behaviour is covered by prometheus_client itself;
    these tests verify our wrapper returns MetricSample objects correctly.
    """

    def test_simple_metric(self) -> None:
        text = "# TYPE gpu_temp gauge\ngpu_temp 75.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temp"
        assert samples[0].value == 75.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = '# TYPE gpu_temp gauge\ngpu_temp{gpu="0",node="n1"} 82.5\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].labels == {"gpu": "0", "node": "n1"}
        assert samples[0].value == 82.5

    def test_multiple_metrics(self) -> None:
        text = "# TYPE metric_a gauge\n" "metric_a 1.0\n" "# TYPE metric_b gauge\n" "metric_b 2.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert parse_prometheus_text("") == []
