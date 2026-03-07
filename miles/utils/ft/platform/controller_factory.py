from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.chain import DetectorChainConfig, build_detector_chain
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from miles.utils.ft.controller.recovery.helpers import SlidingWindowThrottle
from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob
from miles.utils.ft.protocols.platform import (
    DiagnosticOrchestratorProtocol,
    NodeManagerProtocol,
    NotificationProtocol,
    TrainingJobProtocol,
)

if TYPE_CHECKING:
    from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.platform.notifiers.webhook_notifier import WebhookNotifier
    from miles.utils.ft.platform.ray_training_job import RayTrainingJob

logger = logging.getLogger(__name__)


class FtControllerConfig(FtBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ft_id: str = ""
    k8s_label_prefix: str = ""
    platform: Literal["stub", "k8s-ray"] = "stub"
    ray_address: str = "http://127.0.0.1:8265"
    entrypoint: str = ""
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    metric_store_backend: Literal["mini", "prometheus"] = "mini"
    prometheus_url: str = "http://prometheus:9090"
    controller_exporter_port: int = 0
    tick_interval: float = 30.0
    scrape_interval_seconds: float = 10.0
    detector_config: DetectorChainConfig = Field(default_factory=DetectorChainConfig)


_NOTIFIER_SENTINEL: object = object()


def build_ft_controller(
    config: FtControllerConfig | None = None,
    *,
    start_exporter: bool = True,
    node_manager_override: NodeManagerProtocol | None = None,
    training_job_override: TrainingJobProtocol | None = None,
    notifier_override: NotificationProtocol | None | object = _NOTIFIER_SENTINEL,
    detectors_override: list[BaseFaultDetector] | None = None,
    diagnostic_orchestrator_override: DiagnosticOrchestratorProtocol | None = None,
    recovery_cooldown_override: SlidingWindowThrottle | None = None,
    registration_grace_ticks_override: int | None = None,
    **kwargs: object,
) -> FtController:
    """Build an FtController with all dependent components from config parameters.

    Accepts either an ``FtControllerConfig`` or keyword arguments that
    are forwarded to the config constructor.

    Optional ``*_override`` parameters allow tests to inject fake
    dependencies while still using the real ``FtControllerActor`` wrapper.
    """
    if config is not None and kwargs:
        raise ValueError(
            "Cannot provide both 'config' and keyword arguments to build_ft_controller; "
            "use one or the other"
        )

    _has_nm = node_manager_override is not None
    _has_tj = training_job_override is not None
    if _has_nm != _has_tj:
        raise ValueError(
            "node_manager_override and training_job_override must be provided together"
        )

    if config is None:
        config = FtControllerConfig(**kwargs)  # type: ignore[arg-type]

    ft_id = config.ft_id or uuid4().hex[:8]

    if node_manager_override is not None and training_job_override is not None:
        node_manager: NodeManagerProtocol = node_manager_override
        training_job: TrainingJobProtocol = training_job_override
    else:
        node_manager, training_job = _build_platform_components(
            platform=config.platform,
            ray_address=config.ray_address,
            entrypoint=config.entrypoint,
            runtime_env=config.runtime_env,
            ft_id=ft_id,
            k8s_label_prefix=config.k8s_label_prefix,
        )

    controller_exporter = ControllerExporter(port=config.controller_exporter_port)
    if start_exporter:
        controller_exporter.start()

    metric_store, scrape_target_manager = _build_metric_store(config, controller_exporter)

    mini_wandb = MiniWandb()

    if notifier_override is not _NOTIFIER_SENTINEL:
        notifier: NotificationProtocol | None = notifier_override  # type: ignore[assignment]
    else:
        notifier = _build_notifier(platform=config.platform)

    detectors = (
        detectors_override
        if detectors_override is not None
        else build_detector_chain(config=config.detector_config)
    )

    logger.info(
        "build_ft_controller ft_id=%s platform=%s backend=%s exporter_port=%d k8s_label_prefix=%s",
        ft_id, config.platform, config.metric_store_backend,
        config.controller_exporter_port, config.k8s_label_prefix or "(none)",
    )

    create_kwargs: dict[str, Any] = dict(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        scrape_target_manager=scrape_target_manager,
        notifier=notifier,
        detectors=detectors,
        tick_interval=config.tick_interval,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator_override,
    )
    if recovery_cooldown_override is not None:
        create_kwargs["recovery_cooldown"] = recovery_cooldown_override
    if registration_grace_ticks_override is not None:
        create_kwargs["registration_grace_ticks"] = registration_grace_ticks_override

    return FtController.create(**create_kwargs)


def _build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
    runtime_env: dict[str, Any] | None = None,
    ft_id: str = "",
    k8s_label_prefix: str = "",
) -> tuple[StubNodeManager | K8sNodeManager, StubTrainingJob | RayTrainingJob]:
    if platform == "stub":
        return StubNodeManager(), StubTrainingJob()

    if platform == "k8s-ray":
        from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
        from miles.utils.ft.platform.ray_training_job import RayTrainingJob
        from ray.job_submission import JobSubmissionClient

        node_manager = K8sNodeManager(label_prefix=k8s_label_prefix)
        training_job = RayTrainingJob(
            client=JobSubmissionClient(address=ray_address),
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            ft_id=ft_id,
            k8s_label_prefix=k8s_label_prefix,
        )
        return node_manager, training_job

    raise ValueError(f"Unknown platform: {platform}")


_NOTIFIER_PLATFORM_CLASSES: dict[str, str] = {
    "lark": "LarkWebhookNotifier",
    "slack": "SlackWebhookNotifier",
    "discord": "DiscordWebhookNotifier",
}


def _build_notifier(platform: str) -> WebhookNotifier | StubNotifier | None:
    notify_platform, webhook_url = _resolve_notify_config()
    if webhook_url:
        cls = _get_notifier_class(notify_platform)
        return cls(webhook_url=webhook_url)

    if platform == "stub":
        return StubNotifier()

    logger.warning(
        "No notifier configured for platform=%s "
        "(MILES_FT_NOTIFY_WEBHOOK_URL / MILES_FT_LARK_WEBHOOK_URL not set). "
        "Recovery alerts will not be delivered.",
        platform,
    )
    return None


def _resolve_notify_config() -> tuple[str, str]:
    """Return (platform, webhook_url) from environment variables.

    Priority: MILES_FT_NOTIFY_PLATFORM + MILES_FT_NOTIFY_WEBHOOK_URL.
    Fallback: MILES_FT_LARK_WEBHOOK_URL implies platform=lark.
    """
    webhook_url = (os.environ.get("MILES_FT_NOTIFY_WEBHOOK_URL") or "").strip()
    notify_platform = (os.environ.get("MILES_FT_NOTIFY_PLATFORM") or "").strip().lower()

    if webhook_url and notify_platform:
        return notify_platform, webhook_url

    if webhook_url and not notify_platform:
        return "lark", webhook_url

    legacy_url = (os.environ.get("MILES_FT_LARK_WEBHOOK_URL") or "").strip()
    if legacy_url:
        return "lark", legacy_url

    return notify_platform or "lark", ""


def _get_notifier_class(notify_platform: str) -> type[WebhookNotifier]:
    from miles.utils.ft.platform.notifiers.discord_notifier import DiscordWebhookNotifier
    from miles.utils.ft.platform.notifiers.lark_notifier import LarkWebhookNotifier
    from miles.utils.ft.platform.notifiers.slack_notifier import SlackWebhookNotifier

    registry: dict[str, type[WebhookNotifier]] = {
        "lark": LarkWebhookNotifier,
        "slack": SlackWebhookNotifier,
        "discord": DiscordWebhookNotifier,
    }
    cls = registry.get(notify_platform)
    if cls is None:
        raise ValueError(
            f"Unknown notify platform: {notify_platform!r}. "
            f"Supported: {sorted(registry)}"
        )
    return cls


def _build_metric_store(
    config: FtControllerConfig,
    controller_exporter: ControllerExporter,
) -> tuple[MiniPrometheus | PrometheusClient, MiniPrometheus | None]:
    """Return (metric_store, scrape_target_manager) based on config backend."""
    if config.metric_store_backend == "mini":
        mini_prom = MiniPrometheus(config=MiniPrometheusConfig(
            scrape_interval=timedelta(seconds=config.scrape_interval_seconds),
        ))
        mini_prom.add_scrape_target(
            target_id="controller",
            address=controller_exporter.address,
        )
        return mini_prom, mini_prom

    if config.metric_store_backend == "prometheus":
        return PrometheusClient(url=config.prometheus_url), None

    raise ValueError(f"Unknown metric-store-backend: {config.metric_store_backend}")
