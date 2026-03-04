from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Annotated, NamedTuple

import typer

if TYPE_CHECKING:
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_prometheus.protocol import (
    MetricStoreProtocol,
    ScrapeTargetManagerProtocol,
)
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.controller.prometheus_client_store import PrometheusClient
from miles.utils.ft.platform.protocols import NotificationProtocol
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob

logger = logging.getLogger(__name__)

app = typer.Typer()


def _build_k8s_ray_components(
    ray_address: str,
    entrypoint: str,
) -> tuple[K8sNodeManager, RayTrainingJob]:
    """Lazily import and construct K8s + Ray platform components."""
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob
    from ray.job_submission import JobSubmissionClient

    node_manager = K8sNodeManager()
    training_job = RayTrainingJob(
        client=JobSubmissionClient(address=ray_address),
        entrypoint=entrypoint,
    )
    return node_manager, training_job


class _MetricComponents(NamedTuple):
    metric_store: MetricStoreProtocol
    scrape_target_manager: ScrapeTargetManagerProtocol | None
    controller_exporter: ControllerExporter


def _build_metric_components(
    backend: str,
    prometheus_url: str,
    controller_exporter_port: int,
) -> _MetricComponents:
    """Build metric store, optional scrape target manager, and controller exporter."""
    controller_exporter = ControllerExporter(port=controller_exporter_port)

    if backend == "mini":
        mini_prom = MiniPrometheus(config=MiniPrometheusConfig())
        mini_prom.add_scrape_target(
            target_id="controller",
            address=controller_exporter.address,
        )
        return _MetricComponents(
            metric_store=mini_prom,
            scrape_target_manager=mini_prom,
            controller_exporter=controller_exporter,
        )

    if backend == "prometheus":
        prom_client = PrometheusClient(url=prometheus_url)
        return _MetricComponents(
            metric_store=prom_client,
            scrape_target_manager=None,
            controller_exporter=controller_exporter,
        )

    raise typer.BadParameter(f"Unknown metric-store-backend: {backend}")


def _build_notifier(platform_mode: str) -> NotificationProtocol | None:
    webhook_url = (os.environ.get("FT_LARK_WEBHOOK_URL") or "").strip()
    if webhook_url:
        from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier

        return LarkWebhookNotifier(webhook_url=webhook_url)

    if platform_mode == "stub":
        return StubNotifier()

    return None


@app.command()
def main(
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
    platform: Annotated[
        str, typer.Option(help="Platform mode: 'stub' or 'k8s-ray'")
    ] = "stub",
    ray_address: Annotated[
        str, typer.Option(help="Ray dashboard address (k8s-ray mode)")
    ] = "http://127.0.0.1:8265",
    entrypoint: Annotated[
        str, typer.Option(help="Training job entrypoint command (k8s-ray mode)")
    ] = "",
    metric_store_backend: Annotated[
        str, typer.Option(help="Metric store backend: 'mini' or 'prometheus'")
    ] = "mini",
    prometheus_url: Annotated[
        str, typer.Option(help="Prometheus server URL (prometheus mode)")
    ] = "http://prometheus:9090",
    controller_exporter_port: Annotated[
        int, typer.Option(help="Controller Prometheus exporter HTTP port")
    ] = 9400,
) -> None:
    """FT Controller entry point."""
    if platform == "stub":
        node_manager = StubNodeManager()
        training_job = StubTrainingJob()
    elif platform == "k8s-ray":
        node_manager, training_job = _build_k8s_ray_components(
            ray_address=ray_address,
            entrypoint=entrypoint,
        )
    else:
        raise typer.BadParameter(f"Unknown platform: {platform}")

    metric_store, scrape_target_manager, controller_exporter = _build_metric_components(
        backend=metric_store_backend,
        prometheus_url=prometheus_url,
        controller_exporter_port=controller_exporter_port,
    )
    mini_wandb = MiniWandb()
    notifier = _build_notifier(platform_mode=platform)

    controller_exporter.start()
    logger.info(
        "launcher_started backend=%s platform=%s exporter_port=%d",
        metric_store_backend, platform, controller_exporter_port,
    )

    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        detectors=[],
        notifier=notifier,
        tick_interval=tick_interval,
        controller_exporter=controller_exporter,
        scrape_target_manager=scrape_target_manager,
    )

    asyncio.run(controller.run())


if __name__ == "__main__":
    app()
