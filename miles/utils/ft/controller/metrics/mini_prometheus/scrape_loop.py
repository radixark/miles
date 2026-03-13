from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx
from prometheus_client.parser import text_string_to_metric_families

from miles.utils.ft.agents.types import GaugeSample

logger = logging.getLogger(__name__)


_DEFAULT_MAX_CONSECUTIVE_FAILURES = 30


class ScrapeLoop:
    def __init__(
        self,
        store: _IngestTarget,
        scrape_interval_seconds: float,
        max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        self._store = store
        self._scrape_interval_seconds = scrape_interval_seconds
        self._max_consecutive_failures = max_consecutive_failures
        self._targets: dict[str, str] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._running = False
        self._client: httpx.AsyncClient | None = None

    def add_target(self, target_id: str, address: str) -> None:
        self._targets[target_id] = address
        logger.info("mini_prom: scrape target added target_id=%s, address=%s", target_id, address)

    def remove_target(self, target_id: str) -> None:
        was_present = target_id in self._targets
        self._targets.pop(target_id, None)
        self._consecutive_failures.pop(target_id, None)
        if was_present:
            logger.info("mini_prom: scrape target removed target_id=%s", target_id)
        else:
            logger.debug("mini_prom: remove_target no-op target_id=%s not found", target_id)

    @property
    def targets(self) -> dict[str, str]:
        return dict(self._targets)

    async def scrape_once(self) -> None:
        targets = list(self._targets.items())
        if not targets:
            logger.debug("mini_prom: scrape_once skipped, no targets")
            return

        client = self._ensure_client()
        failed_targets: list[tuple[str, int]] = []

        async def _scrape_target(target_id: str, address: str) -> None:
            try:
                response = await client.get(f"{address}/metrics")
                response.raise_for_status()
                samples = parse_prometheus_text(response.text)
                self._store.ingest_samples(target_id=target_id, samples=samples)
                self._consecutive_failures.pop(target_id, None)
            except Exception:
                count = self._consecutive_failures.get(target_id, 0) + 1
                self._consecutive_failures[target_id] = count
                logger.warning(
                    "mini_prom: scrape failed target=%s, address=%s, consecutive_failures=%d",
                    target_id,
                    address,
                    count,
                    exc_info=True,
                )
                if count >= self._max_consecutive_failures:
                    failed_targets.append((target_id, count))

        await asyncio.gather(*(_scrape_target(target_id, address) for target_id, address in targets))

        if failed_targets:
            details = ", ".join(f"{tid}({count})" for tid, count in failed_targets)
            raise RuntimeError(
                f"Scrape targets exceeded max consecutive failures " f"({self._max_consecutive_failures}): {details}"
            )

    async def start(self) -> None:
        logger.info("mini_prom: scrape loop starting, interval=%.1fs", self._scrape_interval_seconds)
        self._running = True
        try:
            while self._running:
                await self.scrape_once()
                await asyncio.sleep(self._scrape_interval_seconds)
        finally:
            await self._close_client()

    async def stop(self) -> None:
        logger.info("mini_prom: scrape loop stopping")
        self._running = False

    async def _close_client(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client


def parse_prometheus_text(text: str) -> list[GaugeSample]:
    samples: list[GaugeSample] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            samples.append(
                GaugeSample(
                    name=sample.name,
                    labels=dict(sample.labels),
                    value=sample.value,
                )
            )
    return samples


class _IngestTarget(Protocol):
    def ingest_samples(
        self,
        target_id: str,
        samples: list[GaugeSample],
    ) -> None: ...
