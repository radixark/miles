"""Composition root for the FtController and its platform dependencies.

Assembles the controller with all concrete implementations (K8sNodeManager,
RayMainJob, notifiers, metric stores, detectors). Analogous to
node_agent.py for the node agent side.

Previously the assembly was split across two factory layers
(controller/factory.py and factories/controller.py). Now unified here.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.notifiers.factory import build_notifier
from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtController, FtControllerBundle
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.metrics.prometheus_api.client import PrometheusClient
from miles.utils.ft.controller.node_agents import NodeAgentRegistry
from miles.utils.ft.controller.state_machines.main import (
    MainContext,
    NormalSt,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.main.models import MainState
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_TIMEOUT_SECONDS
from miles.utils.ft.controller.state_machines.restart.models import (
    MonitoringIterationProgressConfig,
    MonitoringSustainedAliveConfig,
)
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt
from miles.utils.ft.controller.subsystem_hub import RestartMode, SubsystemConfig, SubsystemHub, SubsystemRuntime, SubsystemSpec, TrainingRankRoster
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStore, NullScrapeTargetManager, ScrapeTargetManagerProtocol
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine

if TYPE_CHECKING:
    from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager
    from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob

logger = logging.getLogger(__name__)

_NOTIFIER_SENTINEL: object = object()


# ---------------------------------------------------------------------------
# Core assembly (from components)
# ---------------------------------------------------------------------------


def assemble_ft_controller(
    node_manager: NodeManagerProtocol,
    main_job: MainJobProtocol,
    metric_store: MetricStore,
    *,
    rollout_cell_ids: list[str] | None = None,
    scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
    notifier: NotifierProtocol | None = None,
    detectors: list[BaseFaultDetector] | None = None,
    tick_interval: float = 30.0,
    controller_exporter: ControllerExporter | None = None,
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol | None = None,
    recovery_cooldown_window_minutes: float = 30.0,
    recovery_cooldown_max_count: int = 3,
    registration_grace_ticks: int = 5,
    max_simultaneous_bad_nodes: int = 3,
    monitoring_success_iterations: int = 10,
    monitoring_timeout_seconds: int = 600,
    recovery_timeout_seconds: int = RECOVERY_TIMEOUT_SECONDS,
    rollout_alive_threshold_seconds: float | None = None,
    rollout_monitoring_alive_duration_seconds: float | None = None,
) -> FtControllerBundle:
    """Assemble an FtController from explicit component objects.

    This is the core assembly function used by both ``build_ft_controller``
    (config-driven) and tests (which inject fakes directly).
    """
    from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
    from miles.utils.ft.controller.detectors.chain import build_rollout_detectors, build_shared_hw_detectors
    from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors
    from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

    node_agent_registry = NodeAgentRegistry()
    training_rank_roster_box: Box[TrainingRankRoster | None] = Box(None)

    subsystem_hub = SubsystemHub(
        training_rank_roster_box=training_rank_roster_box,
    )

    def _get_active_training_nodes() -> set[str]:
        roster = training_rank_roster_box.value
        return set(roster.rank_placement.values()) if roster is not None else set()

    resolved_orchestrator: DiagnosticOrchestratorProtocol = diagnostic_orchestrator or DiagnosticOrchestrator(
        node_agent_registry=node_agent_registry,
        pipeline=list(build_all_cluster_executors().values()),
    )

    resolved_exporter = controller_exporter or NullControllerExporter()
    duration_cb = resolved_exporter.observe_recovery_duration
    resolved_detectors = detectors or []

    monitoring_config = MonitoringIterationProgressConfig(
        success_iterations=monitoring_success_iterations,
        timeout_seconds=monitoring_timeout_seconds,
    )

    _cooldown_window_minutes = recovery_cooldown_window_minutes
    _cooldown_max_count = recovery_cooldown_max_count

    def _make_cooldown() -> SlidingWindowThrottle:
        return SlidingWindowThrottle(
            window_minutes=_cooldown_window_minutes,
            max_count=_cooldown_max_count,
        )

    # --- Training SubsystemSpec ---
    training_spec = SubsystemSpec(
        config=SubsystemConfig(
            restart_mode=RestartMode.MAIN_JOB,
            detectors=resolved_detectors,
            monitoring_config=monitoring_config,
        ),
        runtime=SubsystemRuntime(
            actuator=TrainingSubsystemActuator(main_job=main_job),
            cooldown=_make_cooldown(),
            get_active_node_ids=_get_active_training_nodes,
        ),
    )
    subsystem_specs: dict[str, SubsystemSpec] = {"training": training_spec}

    # --- Rollout SubsystemSpecs ---
    rollout_alive_dur = rollout_monitoring_alive_duration_seconds if rollout_monitoring_alive_duration_seconds is not None else 180
    rollout_det_kwargs: dict[str, float] = {}
    if rollout_alive_threshold_seconds is not None:
        rollout_det_kwargs["alive_threshold_seconds"] = rollout_alive_threshold_seconds

    for cell_id in (rollout_cell_ids or []):
        name = f"rollout_{cell_id}"
        _cid = cell_id  # capture for closure
        subsystem_specs[name] = SubsystemSpec(
            config=SubsystemConfig(
                restart_mode=RestartMode.SUBSYSTEM,
                detectors=build_shared_hw_detectors() + build_rollout_detectors(cell_id=cell_id, **rollout_det_kwargs),
                monitoring_config=MonitoringSustainedAliveConfig(alive_duration_seconds=rollout_alive_dur),
            ),
            runtime=SubsystemRuntime(
                actuator=RayRolloutActuator(
                    get_handle=lambda: subsystem_hub.rollout_manager_handle,
                    cell_id=cell_id,
                ),
                cooldown=_make_cooldown(),
                get_active_node_ids=lambda _c=_cid: subsystem_hub.get_rollout_node_ids(_c),
            ),
        )

    # --- Create Main SM (includes all subsystems from the start) ---
    initial_subsystem_states = {name: DetectingAnomalySt() for name in subsystem_specs}
    main_stepper = create_main_stepper()
    controller_sm: StateMachine[MainState, MainContext] = StateMachine(
        initial_state=NormalSt(subsystems=initial_subsystem_states),
        stepper=main_stepper,
    )

    # --- Create TickLoop (no FtController dependency) ---
    tick_loop = TickLoop(
        state_machine=controller_sm,
        registration_grace_ticks=registration_grace_ticks,
    )

    # --- Create FtController ---
    instance = FtController(
        main_job=main_job,
        state_machine=controller_sm,
        subsystem_hub=subsystem_hub,
        metric_store=metric_store,
        node_agent_registry=node_agent_registry,
        tick_interval=tick_interval,
        tick_loop=tick_loop,
        notifier=notifier,
        node_manager=node_manager,
        diagnostic_orchestrator=resolved_orchestrator,
        recovery_timeout_seconds=recovery_timeout_seconds,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        subsystem_specs=subsystem_specs,
        rank_pids_provider=lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := training_rank_roster_box.value) is not None else {}),
        training_rank_roster_box=training_rank_roster_box,
        on_recovery_duration=duration_cb,
        scrape_target_manager=scrape_target_manager,
        controller_exporter=controller_exporter,
        registration_grace_ticks=registration_grace_ticks,
    )

    return FtControllerBundle(controller=instance, subsystem_hub=subsystem_hub)


# ---------------------------------------------------------------------------
# Config-driven builder (from FtControllerConfig)
# ---------------------------------------------------------------------------


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
    )
    if recovery_cooldown_override is not None:
        assemble_kwargs["recovery_cooldown_window_minutes"] = recovery_cooldown_override.window_minutes
        assemble_kwargs["recovery_cooldown_max_count"] = recovery_cooldown_override.max_count
    if registration_grace_ticks_override is not None:
        assemble_kwargs["registration_grace_ticks"] = registration_grace_ticks_override
    if max_simultaneous_bad_nodes_override is not None:
        assemble_kwargs["max_simultaneous_bad_nodes"] = max_simultaneous_bad_nodes_override
    if recovery_timeout_seconds_override is not None:
        assemble_kwargs["recovery_timeout_seconds"] = recovery_timeout_seconds_override
    if monitoring_timeout_seconds_override is not None:
        assemble_kwargs["monitoring_timeout_seconds"] = monitoring_timeout_seconds_override
    if monitoring_success_iterations_override is not None:
        assemble_kwargs["monitoring_success_iterations"] = monitoring_success_iterations_override
    if rollout_alive_threshold_seconds_override is not None:
        assemble_kwargs["rollout_alive_threshold_seconds"] = rollout_alive_threshold_seconds_override
    if rollout_monitoring_alive_duration_seconds_override is not None:
        assemble_kwargs["rollout_monitoring_alive_duration_seconds"] = rollout_monitoring_alive_duration_seconds_override

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_platform_components(
    platform: str,
    ray_address: str,
    entrypoint: str,
    ft_id: str,
    k8s_label_prefix: str,
    runtime_env: dict[str, Any] | None = None,
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
