from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from miles.utils.ft.controller.rank_registry import RankRegistry
from miles.utils.ft.models import FtBaseModel
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob

if TYPE_CHECKING:
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob

logger = logging.getLogger(__name__)


class FtControllerConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_id: str = ""
    k8s_label_suffix: str = ""
    platform: Literal["stub", "k8s-ray"] = "stub"
    ray_address: str = "http://127.0.0.1:8265"
    entrypoint: str = ""
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    metric_store_backend: Literal["mini", "prometheus"] = "mini"
    prometheus_url: str = "http://prometheus:9090"
    controller_exporter_port: int = 0
    tick_interval: float = 30.0


def _build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
    runtime_env: dict[str, Any] | None = None,
    ft_id: str = "",
    k8s_label_suffix: str = "",
) -> tuple[StubNodeManager | K8sNodeManager, StubTrainingJob | RayTrainingJob]:
    if platform == "stub":
        return StubNodeManager(), StubTrainingJob()

    if platform == "k8s-ray":
        from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
        from miles.utils.ft.platform.ray_training_job import RayTrainingJob
        from ray.job_submission import JobSubmissionClient

        node_manager = K8sNodeManager(label_suffix=k8s_label_suffix)
        training_job = RayTrainingJob(
            client=JobSubmissionClient(address=ray_address),
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            ft_id=ft_id,
            k8s_label_suffix=k8s_label_suffix,
        )
        return node_manager, training_job

    raise ValueError(f"Unknown platform: {platform}")


def _build_notifier(platform: str) -> LarkWebhookNotifier | StubNotifier | None:
    webhook_url = (os.environ.get("FT_LARK_WEBHOOK_URL") or "").strip()
    if webhook_url:
        from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier

        return LarkWebhookNotifier(webhook_url=webhook_url)

    if platform == "stub":
        return StubNotifier()

    logger.warning(
        "No notifier configured for platform=%s (FT_LARK_WEBHOOK_URL not set). "
        "Recovery alerts will not be delivered.",
        platform,
    )
    return None


def build_ft_controller(
    config: FtControllerConfig | None = None,
    *,
    start_exporter: bool = True,
    **kwargs: object,
) -> FtController:
    """Build an FtController with all dependent components from config parameters.

    Accepts either an ``FtControllerConfig`` or keyword arguments that
    are forwarded to the config constructor.
    """
    if config is not None and kwargs:
        raise ValueError(
            "Cannot provide both 'config' and keyword arguments to build_ft_controller; "
            "use one or the other"
        )
    if config is None:
        config = FtControllerConfig(**kwargs)  # type: ignore[arg-type]

    ft_id = config.ft_id or uuid4().hex[:8]

    node_manager, training_job = _build_platform_components(
        platform=config.platform,
        ray_address=config.ray_address,
        entrypoint=config.entrypoint,
        runtime_env=config.runtime_env,
        ft_id=ft_id,
        k8s_label_suffix=config.k8s_label_suffix,
    )

    controller_exporter = ControllerExporter(port=config.controller_exporter_port)

    if config.metric_store_backend == "mini":
        mini_prom = MiniPrometheus(config=MiniPrometheusConfig())
        mini_prom.add_scrape_target(
            target_id="controller",
            address=controller_exporter.address,
        )
        metric_store = mini_prom
        scrape_target_manager = mini_prom
    elif config.metric_store_backend == "prometheus":
        metric_store = PrometheusClient(url=config.prometheus_url)
        scrape_target_manager = None
    else:
        raise ValueError(f"Unknown metric-store-backend: {config.metric_store_backend}")

    mini_wandb = MiniWandb()
    rank_registry = RankRegistry(
        mini_wandb=mini_wandb,
        scrape_target_manager=scrape_target_manager,
    )

    notifier = _build_notifier(platform=config.platform)
    detectors = build_detector_chain()

    if start_exporter:
        controller_exporter.start()

    logger.info(
        "build_ft_controller ft_id=%s platform=%s backend=%s exporter_port=%d k8s_label_suffix=%s",
        ft_id, config.platform, config.metric_store_backend,
        config.controller_exporter_port, config.k8s_label_suffix or "(none)",
    )

    return FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        rank_registry=rank_registry,
        notifier=notifier,
        detectors=detectors,
        tick_interval=config.tick_interval,
        controller_exporter=controller_exporter,
    )
