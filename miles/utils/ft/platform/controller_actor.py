from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ray

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.controller.prometheus_client_store import PrometheusClient
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob

if TYPE_CHECKING:
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FtControllerConfig:
    platform: str = "stub"
    ray_address: str = "http://127.0.0.1:8265"
    entrypoint: str = ""
    metric_store_backend: str = "mini"
    prometheus_url: str = "http://prometheus:9090"
    controller_exporter_port: int = 9400
    tick_interval: float = 30.0


def _build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
) -> tuple[StubNodeManager | K8sNodeManager, StubTrainingJob | RayTrainingJob]:
    if platform == "stub":
        return StubNodeManager(), StubTrainingJob()

    if platform == "k8s-ray":
        from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
        from miles.utils.ft.platform.ray_training_job import RayTrainingJob
        from ray.job_submission import JobSubmissionClient

        node_manager = K8sNodeManager()
        training_job = RayTrainingJob(
            client=JobSubmissionClient(address=ray_address),
            entrypoint=entrypoint,
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

    node_manager, training_job = _build_platform_components(
        platform=config.platform,
        ray_address=config.ray_address,
        entrypoint=config.entrypoint,
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
    notifier = _build_notifier(platform=config.platform)
    detectors = build_detector_chain()

    if start_exporter:
        controller_exporter.start()

    logger.info(
        "build_ft_controller platform=%s backend=%s exporter_port=%d",
        config.platform, config.metric_store_backend, config.controller_exporter_port,
    )

    return FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        detectors=detectors,
        tick_interval=config.tick_interval,
        controller_exporter=controller_exporter,
        scrape_target_manager=scrape_target_manager,
    )


class _FtControllerActorCls:
    """Thin wrapper around FtController for use as a Ray Actor.

    Created as a Detached Named Async Actor so that FtMegatronAgent
    can find it via ``ray.get_actor(FT_CONTROLLER_ACTOR_NAME)``.
    FtController remains a plain Python class for testability.
    """

    def __init__(self, config: FtControllerConfig | None = None, **kwargs: object) -> None:
        self._ctrl = build_ft_controller(config=config, **kwargs)

    async def run(self) -> None:
        await self._ctrl.run()

    async def shutdown(self) -> None:
        await self._ctrl.shutdown()

    async def log_step(
        self,
        run_id: str,
        rank: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        await self._ctrl.log_step(
            run_id=run_id,
            rank=rank,
            step=step,
            metrics=metrics,
        )

    async def register_rank(
        self,
        run_id: str,
        rank: int,
        world_size: int,
        node_id: str,
        exporter_address: str,
        pid: int | None = None,
    ) -> None:
        await self._ctrl.register_rank(
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            node_id=node_id,
            exporter_address=exporter_address,
            pid=pid,
        )

    def register_agent(self, node_id: str, agent: object) -> None:
        self._ctrl.register_agent(node_id=node_id, agent=agent)

    def get_status(self) -> dict[str, object]:
        return self._ctrl.get_status()


FtControllerActor = ray.remote(
    num_gpus=0,
    max_restarts=-1,
    max_task_retries=-1,
)(_FtControllerActorCls)
