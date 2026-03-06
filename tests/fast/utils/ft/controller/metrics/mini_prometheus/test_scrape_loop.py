"""Tests for ScrapeLoop — target management, client lifecycle, stop edge cases."""
from __future__ import annotations

import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.scrape_loop import ScrapeLoop
from miles.utils.ft.models._metrics import MetricSample


class _FakeStore:
    """Records ingest_samples calls."""

    def __init__(self) -> None:
        self.ingested: list[tuple[str, list[MetricSample]]] = []

    def ingest_samples(self, target_id: str, samples: list[MetricSample]) -> None:
        self.ingested.append((target_id, samples))


# ===================================================================
# Target management
# ===================================================================


class TestTargetManagement:
    def test_add_target(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)

        loop.add_target("t1", "http://host:9090")

        assert loop.targets == {"t1": "http://host:9090"}

    def test_remove_target(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        loop.add_target("t1", "http://host:9090")

        loop.remove_target("t1")

        assert loop.targets == {}

    def test_remove_nonexistent_target_is_safe(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)

        loop.remove_target("nonexistent")

    def test_targets_property_returns_copy(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        loop.add_target("t1", "http://host:9090")

        targets = loop.targets
        targets["t2"] = "http://other:9090"

        assert "t2" not in loop.targets


# ===================================================================
# scrape_once
# ===================================================================


class TestScrapeOnce:
    @pytest.mark.anyio
    async def test_empty_targets_is_noop(self) -> None:
        store = _FakeStore()
        loop = ScrapeLoop(store=store, scrape_interval_seconds=1.0)

        await loop.scrape_once()

        assert len(store.ingested) == 0


# ===================================================================
# _ensure_client
# ===================================================================


class TestEnsureClient:
    def test_creates_client_when_none(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        assert loop._client is None

        client = loop._ensure_client()

        assert client is not None
        assert not client.is_closed

    def test_reuses_existing_open_client(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        client1 = loop._ensure_client()
        client2 = loop._ensure_client()

        assert client1 is client2

    @pytest.mark.anyio
    async def test_recreates_client_after_close(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        client1 = loop._ensure_client()
        await client1.aclose()
        assert client1.is_closed

        client2 = loop._ensure_client()

        assert client2 is not client1
        assert not client2.is_closed
        await client2.aclose()


# ===================================================================
# stop
# ===================================================================


class TestStop:
    @pytest.mark.anyio
    async def test_stop_sets_running_false(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        loop._running = True

        await loop.stop()

        assert loop._running is False

    @pytest.mark.anyio
    async def test_stop_closes_client(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        client = loop._ensure_client()
        assert not client.is_closed

        await loop.stop()

        assert client.is_closed
        assert loop._client is None

    @pytest.mark.anyio
    async def test_stop_when_client_is_none(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)

        await loop.stop()

    @pytest.mark.anyio
    async def test_stop_when_client_already_closed(self) -> None:
        loop = ScrapeLoop(store=_FakeStore(), scrape_interval_seconds=1.0)
        client = loop._ensure_client()
        await client.aclose()

        await loop.stop()
