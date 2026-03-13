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
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStore
from miles.utils.ft.factories.controller.backends import build_metric_store, build_platform_components
from miles.utils.ft.factories.controller.wiring import assemble_ft_controller

logger = logging.getLogger(__name__)

_NOTIFIER_SENTINEL: object = object()


def _rollout_num_cells_to_ids(num_cells: int) -> list[str] | None:
    if num_cells == 0:
        logger.debug("wiring: rollout_num_cells=0, no rollout cell IDs")
        return None
    if num_cells == 1:
        logger.debug("wiring: rollout_num_cells=1, using ['default']")
        return ["default"]
    logger.debug("wiring: rollout_num_cells=%d, using numeric IDs", num_cells)
    return [str(i) for i in range(num_cells)]


def build_ft_controller(
    config: FtControllerConfig,
    *,
    start_exporter: bool = True,
    runtime_config_override: ControllerRuntimeConfig | None = None,
    node_manager_override: NodeManagerProtocol | None = None,
    main_job_override: MainJobProtocol | None = None,
    notifier_override: NotifierProtocol | None | object = _NOTIFIER_SENTINEL,
    detectors_override: list[BaseFaultDetector] | None = None,
    diagnostic_orchestrator_override: DiagnosticOrchestratorProtocol | None = None,
) -> FtControllerBundle:
    """Build an FtController with all dependent components from config.

    Optional ``*_override`` parameters allow tests to inject fake
    dependencies while still using the real ``FtControllerActor`` wrapper.
    """
    _has_nm = node_manager_override is not None
    _has_tj = main_job_override is not None
    if _has_nm != _has_tj:
        raise ValueError("node_manager_override and main_job_override must be provided together")

    runtime_config = runtime_config_override or config.to_runtime_config()
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

    return assemble_ft_controller(
        runtime_config,
        node_manager,
        main_job,
        metric_store,
        scrape_target_manager=scrape_target_manager,
        notifier=notifier,
        detectors=detectors,
        rollout_cell_ids=rollout_cell_ids,
        controller_exporter=controller_exporter,
        diagnostic_orchestrator=diagnostic_orchestrator_override,
    )


def launch_ft_controller_actor(
    config: FtControllerConfig,
    actor_name: str,
) -> Any:
    """Create and return a named FtControllerActor with builder injection."""
    from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
    from miles.utils.ft.factories.scheduling import get_cpu_only_scheduling_options

    logger.info("wiring: launch_ft_controller_actor actor_name=%s", actor_name)
    options_kwargs = get_cpu_only_scheduling_options()
    options_kwargs["name"] = actor_name

    return FtControllerActor.options(**options_kwargs).remote(
        builder=build_ft_controller,
        config=config,
    )
