"""Composition root for the FtController and its platform dependencies.

Assembles the controller with all concrete implementations (K8sNodeManager,
RayMainJob, notifiers, metric stores, detectors). Analogous to
node_agent.py for the node agent side.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.notifiers.factory import build_notifier
from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.factory import FtControllerBundle, create_ft_controller
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStore, NullScrapeTargetManager
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

if TYPE_CHECKING:
    from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob

logger = logging.getLogger(__name__)

_NOTIFIER_SENTINEL: object = object()


def _rollout_num_cells_to_ids(num_cells: int) -> list[str] | None:
    if num_cells == 0:
        return None
    if num_cells == 1:
        return ["default"]
    return [str(i) for i in range(num_cells)]


def build_ft_controller(
    config: FtControllerConfig | None = None,
    *,
    start_exporter: bool = True,
    node_manager_override: NodeManagerProtocol | None = None,
    main_job_override: MainJobProtocol | None = None,
    notifier_override: NotifierProtocol | None | object = _NOTIFIER_SENTINEL,
    detectors_override: list[BaseFaultDetector] | None = None,
    diagnostic_orchestrator_override: DiagnosticOrchestratorProtocol | None = None,
    recovery_cooldown_override: SlidingWindowThrottle | None = None,
    registration_grace_ticks_override: int | None = None,
    max_simultaneous_bad_nodes_override: int | None = None,
    recovery_timeout_seconds_override: int | None = None,
    monitoring_timeout_seconds_override: int | None = None,
    monitoring_success_iterations_override: int | None = None,
    rollout_alive_threshold_seconds_override: float | None = None,
    rollout_monitoring_alive_duration_seconds_override: float | None = None,
    **kwargs: object,
) -> FtControllerBundle:
    """Build an FtController with all dependent components from config parameters.

    Accepts either an ``FtControllerConfig`` or keyword arguments that
    are forwarded to the config constructor.

    Optional ``*_override`` parameters allow tests to inject fake
    dependencies while still using the real ``FtControllerActor`` wrapper.
    """
    if config is not None and kwargs:
        raise ValueError(
            "Cannot provide both 'config' and keyword arguments to build_ft_controller; " "use one or the other"
        )

    _has_nm = node_manager_override is not None
    _has_tj = main_job_override is not None
    if _has_nm != _has_tj:
        raise ValueError("node_manager_override and main_job_override must be provided together")

    if config is None:
        config = FtControllerConfig(**kwargs)  # type: ignore[arg-type]

    ft_id = config.ft_id or uuid4().hex[:8]

    if node_manager_override is not None and main_job_override is not None:
        node_manager: NodeManagerProtocol = node_manager_override
        main_job: MainJobProtocol = main_job_override
    else:
        node_manager, main_job = _build_platform_components(
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

    time_series_store, scrape_target_manager = _build_metric_store(config, controller_exporter)

    metric_store = MetricStore(
        time_series_store=time_series_store,
        mini_wandb=MiniWandb(),
    )

    if notifier_override is not _NOTIFIER_SENTINEL:
        notifier: NotifierProtocol | None = notifier_override  # type: ignore[assignment]
    else:
        notifier = build_notifier(
            platform=config.platform,
            notify_webhook_url=config.notify_webhook_url,
            notify_platform=config.notify_platform,
        )

    detectors = (
        detectors_override if detectors_override is not None else build_detector_chain(config=config.detector_config)
    )

    logger.info(
        "build_ft_controller ft_id=%s platform=%s backend=%s exporter_port=%d k8s_label_prefix=%s",
        ft_id,
        config.platform,
        config.metric_store_backend,
        config.controller_exporter_port,
        config.k8s_label_prefix or "(none)",
    )

    rollout_cell_ids = _rollout_num_cells_to_ids(config.rollout_num_cells)

    create_kwargs: dict[str, Any] = dict(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        scrape_target_manager=scrape_target_manager,
        notifier=notifier,
        detectors=detectors,
        tick_interval=config.tick_interval,
        rollout_cell_ids=rollout_cell_ids,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator_override,
    )
    if recovery_cooldown_override is not None:
        create_kwargs["recovery_cooldown"] = recovery_cooldown_override
    if registration_grace_ticks_override is not None:
        create_kwargs["registration_grace_ticks"] = registration_grace_ticks_override
    if max_simultaneous_bad_nodes_override is not None:
        create_kwargs["max_simultaneous_bad_nodes"] = max_simultaneous_bad_nodes_override
    if recovery_timeout_seconds_override is not None:
        create_kwargs["recovery_timeout_seconds"] = recovery_timeout_seconds_override
    if monitoring_timeout_seconds_override is not None:
        create_kwargs["monitoring_timeout_seconds"] = monitoring_timeout_seconds_override
    if monitoring_success_iterations_override is not None:
        create_kwargs["monitoring_success_iterations"] = monitoring_success_iterations_override
    if rollout_alive_threshold_seconds_override is not None:
        create_kwargs["rollout_alive_threshold_seconds"] = rollout_alive_threshold_seconds_override
    if rollout_monitoring_alive_duration_seconds_override is not None:
        create_kwargs["rollout_monitoring_alive_duration_seconds"] = rollout_monitoring_alive_duration_seconds_override

    return create_ft_controller(**create_kwargs)


def launch_ft_controller_actor(
    config: FtControllerConfig,
    actor_name: str,
) -> Any:
    """Create and return a named FtControllerActor with builder injection."""
    from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
    from miles.utils.ft.factories.scheduling import get_cpu_only_scheduling_options

    options_kwargs = get_cpu_only_scheduling_options()
    options_kwargs["name"] = actor_name

    return FtControllerActor.options(**options_kwargs).remote(
        builder=build_ft_controller,
        config=config,
    )


def _build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
    runtime_env: dict[str, Any] | None = None,
    ft_id: str = "",
    k8s_label_prefix: str = "",
) -> tuple[StubNodeManager | K8sNodeManager, StubMainJob | RayMainJob]:
    if platform == "stub":
        return StubNodeManager(), StubMainJob()

    if platform == "k8s-ray":
        from ray.job_submission import JobSubmissionClient

        from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
        from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob

        namespace = os.environ.get("K8S_NAMESPACE", "")
        if not namespace:
            raise RuntimeError("K8S_NAMESPACE env var not set. " "Configure Kubernetes Downward API in pod spec.")

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
        )
        return node_manager, main_job

    raise ValueError(f"Unknown platform: {platform}")


def _build_metric_store(
    config: FtControllerConfig,
    controller_exporter: ControllerExporter,
) -> tuple[MiniPrometheus | PrometheusClient, MiniPrometheus | NullScrapeTargetManager]:
    """Return (metric_store, scrape_target_manager) based on config backend."""
    if config.metric_store_backend == "mini":
        mini_prom = MiniPrometheus(
            config=MiniPrometheusConfig(
                scrape_interval=timedelta(seconds=config.scrape_interval_seconds),
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
