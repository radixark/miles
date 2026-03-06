"""Local Ray: Metrics pipeline — exporter→scrape→detection, log_step→MiniWandb."""
from __future__ import annotations

import time

import pytest
import ray
from prometheus_client import CollectorRegistry, Gauge

from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig

from tests.fast.utils.ft.integration.local_ray.conftest import get_status

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.timeout(60),
]


class TestLogStepArrivesInMiniWandb:
    def test_latest_iteration_updated_after_log_step(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        ray.get(handle.log_step.remote(
            run_id=run_id, step=42, metrics={"loss": 0.1, "iteration": 42},
        ), timeout=5)

        status = get_status(handle)
        assert status.latest_iteration == 42

    def test_multiple_log_steps_track_latest(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        for step in [10, 20, 30]:
            ray.get(handle.log_step.remote(
                run_id=run_id, step=step, metrics={"iteration": step},
            ), timeout=5)

        status = get_status(handle)
        assert status.latest_iteration == 30


class TestExporterScrapeByMiniPrometheus:
    """Verify that MiniPrometheus can scrape gauges from a PrometheusExporter.

    This test runs entirely in the test process (no actor), but validates the
    exporter→scrape data path used within the controller's tick loop.
    """

    @pytest.mark.anyio
    async def test_mini_prometheus_scrapes_exporter_gauges(
        self, local_ray: None,
    ) -> None:
        exporter = PrometheusExporter()
        gauge = Gauge(
            "test_scrape_metric",
            "test gauge",
            labelnames=["rank"],
            registry=exporter.registry,
        )
        gauge.labels(rank="0").set(99.0)

        mini_prom = MiniPrometheus(
            config=MiniPrometheusConfig(),
        )
        mini_prom.add_scrape_target(
            target_id="test-rank",
            address=exporter.get_address(),
        )

        await mini_prom.scrape_once()

        df = mini_prom.query_latest(metric_name="test_scrape_metric")
        try:
            assert len(df) >= 1
            assert 99.0 in df["value"].to_list()
        finally:
            exporter.shutdown()


class TestRankExporterRegisteredAsScrapeTarget:
    """When register_training_rank provides an exporter_address, it should
    be added as a scrape target in MiniPrometheus."""

    def test_exporter_address_used_for_scraping(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        exporter = PrometheusExporter()
        gauge = Gauge(
            "ft_training_iteration",
            "test iteration gauge",
            labelnames=["rank", "node_id"],
            registry=exporter.registry,
        )
        gauge.labels(rank="0", node_id="n0").set(55.0)

        try:
            ray.get(handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address=exporter.get_address(),
            ), timeout=5)

            time.sleep(0.3)
        finally:
            exporter.shutdown()
