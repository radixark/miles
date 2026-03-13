"""Backend selection: platform components and metric store."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import NullScrapeTargetManager

if TYPE_CHECKING:
    from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob


def build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
    ft_id: str,
    k8s_label_prefix: str,
    k8s_namespace: str = "",
    runtime_env: dict[str, Any] | None = None,
    ray_job_poll_interval_seconds: float = 5.0,
    ray_submit_timeout_seconds: float = 60.0,
    ray_get_status_timeout_seconds: float = 30.0,
    ray_stop_job_timeout_seconds: float = 30.0,
) -> tuple[StubNodeManager | K8sNodeManager, StubMainJob | RayMainJob]:
    if platform == "stub":
        return StubNodeManager(), StubMainJob()

    if platform == "k8s-ray":
        from ray.job_submission import JobSubmissionClient

        from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
        from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob

        namespace = k8s_namespace or os.environ.get("K8S_NAMESPACE", "")
        if not namespace:
            raise RuntimeError("K8s namespace not configured. " "Set --k8s-namespace or the K8S_NAMESPACE env var.")

        node_manager = K8sNodeManager(
            label_prefix=k8s_label_prefix,
            namespace=namespace,
        )
        main_job = RayMainJob(
            client=JobSubmissionClient(address=ray_address),
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            ft_id=ft_id,
            k8s_label_prefix=k8s_label_prefix,
            poll_interval_seconds=ray_job_poll_interval_seconds,
            submit_timeout_seconds=ray_submit_timeout_seconds,
            get_status_timeout_seconds=ray_get_status_timeout_seconds,
            stop_job_timeout_seconds=ray_stop_job_timeout_seconds,
        )
        return node_manager, main_job

    raise ValueError(f"Unknown platform: {platform}")


def build_metric_store(
    config: FtControllerConfig,
    controller_exporter: ControllerExporter,
) -> tuple[MiniPrometheus | PrometheusClient, MiniPrometheus | NullScrapeTargetManager]:
    """Return (metric_store, scrape_target_manager) based on config backend."""
    if config.metric_store_backend == "mini":
        mini_prom = MiniPrometheus(
            config=MiniPrometheusConfig(
                scrape_interval=timedelta(seconds=config.scrape_interval_seconds),
                retention=timedelta(minutes=config.mini_prometheus_retention_minutes),
            )
        )
        mini_prom.add_scrape_target(
            target_id="controller",
            address=controller_exporter.address,
        )
        return mini_prom, mini_prom

    if config.metric_store_backend == "prometheus":
        return PrometheusClient(url=config.prometheus_url), NullScrapeTargetManager()

    raise ValueError(f"Unknown metric-store-backend: {config.metric_store_backend}")
