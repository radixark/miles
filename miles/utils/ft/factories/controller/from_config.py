"""Config-driven builder: create FtController from FtControllerConfig."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.notifiers.factory import build_notifier
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtControllerBundle
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStore
from miles.utils.ft.factories.controller.backends import build_metric_store, build_platform_components
from miles.utils.ft.factories.controller.wiring import assemble_ft_controller
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle

logger = logging.getLogger(__name__)

_NOTIFIER_SENTINEL: object = object()

_SIMPLE_OVERRIDES = {
    "registration_grace_ticks_override": "registration_grace_ticks",
    "max_simultaneous_bad_nodes_override": "max_simultaneous_bad_nodes",
    "recovery_timeout_seconds_override": "recovery_timeout_seconds",
    "monitoring_timeout_seconds_override": "monitoring_timeout_seconds",
    "monitoring_success_iterations_override": "monitoring_success_iterations",
    "rollout_alive_threshold_seconds_override": "rollout_alive_threshold_seconds",
    "rollout_monitoring_alive_duration_seconds_override": "rollout_monitoring_alive_duration_seconds",
}


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
        node_manager, main_job = build_platform_components(
            platform=config.platform,
            ray_address=config.ray_address,
            entrypoint=config.entrypoint,
            runtime_env=config.runtime_env,
            ft_id=ft_id,
            k8s_label_prefix=config.k8s_label_prefix,
            k8s_namespace=config.k8s_namespace,
            ray_job_poll_interval_seconds=config.ray_job_poll_interval_seconds,
            ray_submit_timeout_seconds=config.ray_submit_timeout_seconds,
            ray_get_status_timeout_seconds=config.ray_get_status_timeout_seconds,
            ray_stop_job_timeout_seconds=config.ray_stop_job_timeout_seconds,
        )

    controller_exporter = ControllerExporter(port=config.controller_exporter_port)
    if start_exporter:
        controller_exporter.start()

    time_series_store, scrape_target_manager = build_metric_store(config, controller_exporter)

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
            notify_timeout_seconds=config.notify_timeout_seconds,
            notify_max_retries=config.notify_max_retries,
            notify_initial_backoff_seconds=config.notify_initial_backoff_seconds,
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

    assemble_kwargs: dict[str, Any] = dict(
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
        recovery_cooldown_window_minutes=config.recovery_cooldown_window_minutes,
        recovery_cooldown_max_count=config.recovery_cooldown_max_count,
        registration_grace_ticks=config.registration_grace_ticks,
        max_simultaneous_bad_nodes=config.max_simultaneous_bad_nodes,
        monitoring_success_iterations=config.monitoring_success_iterations,
        monitoring_timeout_seconds=config.monitoring_timeout_seconds,
        recovery_timeout_seconds=config.recovery_timeout_seconds,
        rollout_alive_threshold_seconds=config.rollout_alive_threshold_seconds,
        rollout_monitoring_alive_duration_seconds=config.rollout_monitoring_alive_duration_seconds,
    )

    if recovery_cooldown_override is not None:
        assemble_kwargs["recovery_cooldown_window_minutes"] = recovery_cooldown_override.window_minutes
        assemble_kwargs["recovery_cooldown_max_count"] = recovery_cooldown_override.max_count

    all_overrides = locals()
    for param, kwarg in _SIMPLE_OVERRIDES.items():
        if (val := all_overrides[param]) is not None:
            assemble_kwargs[kwarg] = val

    return assemble_ft_controller(**assemble_kwargs)


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
