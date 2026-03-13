"""Core assembly: wire component objects into an FtControllerBundle."""

from __future__ import annotations

from collections.abc import Callable

from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtController, FtControllerBundle
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.node_agents import NodeAgentRegistry
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.controller.state_machines.main import MainContext, NormalSt, create_main_stepper
from miles.utils.ft.controller.state_machines.main.models import MainState
from miles.utils.ft.controller.state_machines.restart.models import (
    MonitoringIterationProgressConfig,
    MonitoringRunningAfterDelayConfig,
)
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt
from miles.utils.ft.controller.subsystem_hub import (
    RestartMode,
    SubsystemConfig,
    SubsystemHub,
    SubsystemRuntime,
    SubsystemSpec,
    TrainingRankRoster,
)
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol, MetricStore, ScrapeTargetManagerProtocol
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine


def assemble_ft_controller(
    runtime_config: ControllerRuntimeConfig,
    node_manager: NodeManagerProtocol,
    main_job: MainJobProtocol,
    metric_store: MetricStore,
    *,
    rollout_cell_ids: list[str] | None = None,
    scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
    notifier: NotifierProtocol | None = None,
    detectors: list[BaseFaultDetector] | None = None,
    controller_exporter: ControllerExporter | None = None,
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol | None = None,
) -> FtControllerBundle:
    """Assemble an FtController from explicit component objects.

    This is the core assembly function used by both ``build_ft_controller``
    (config-driven) and tests (which inject fakes directly).
    """
    from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors
    from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

    node_agent_registry = NodeAgentRegistry()
    training_rank_roster_box: Box[TrainingRankRoster | None] = Box(None)

    subsystem_hub = SubsystemHub(
        training_rank_roster_box=training_rank_roster_box,
    )

    def _get_active_training_nodes() -> frozenset[str]:
        roster = training_rank_roster_box.value
        return frozenset(roster.rank_placement.values()) if roster is not None else frozenset()

    resolved_orchestrator: DiagnosticOrchestratorProtocol = diagnostic_orchestrator or DiagnosticOrchestrator(
        node_agent_registry=node_agent_registry,
        pipeline=list(build_all_cluster_executors().values()),
    )

    resolved_exporter = controller_exporter or NullControllerExporter()
    duration_cb = resolved_exporter.observe_recovery_duration
    resolved_detectors = detectors or []

    monitoring_config = MonitoringIterationProgressConfig(
        success_iterations=runtime_config.monitoring_success_iterations,
        timeout_seconds=runtime_config.monitoring_timeout_seconds,
    )

    def _make_cooldown() -> SlidingWindowThrottle:
        return SlidingWindowThrottle(
            window_minutes=runtime_config.recovery_cooldown_window_minutes,
            max_count=runtime_config.recovery_cooldown_max_count,
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
    rollout_specs = _build_rollout_subsystem_specs(
        rollout_cell_ids=rollout_cell_ids,
        rollout_alive_threshold_seconds=runtime_config.rollout_alive_threshold_seconds,
        rollout_monitoring_alive_duration_seconds=runtime_config.rollout_monitoring_alive_duration_seconds,
        monitoring_timeout_seconds=runtime_config.monitoring_timeout_seconds,
        subsystem_hub=subsystem_hub,
        make_cooldown=_make_cooldown,
    )
    subsystem_specs.update(rollout_specs)

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
        registration_grace_ticks=runtime_config.registration_grace_ticks,
    )

    # --- Create FtController ---
    instance = FtController(
        runtime_config=runtime_config,
        main_job=main_job,
        state_machine=controller_sm,
        subsystem_hub=subsystem_hub,
        metric_store=metric_store,
        node_agent_registry=node_agent_registry,
        tick_loop=tick_loop,
        notifier=notifier,
        node_manager=node_manager,
        diagnostic_orchestrator=resolved_orchestrator,
        subsystem_specs=subsystem_specs,
        rank_pids_provider=lambda node_id: (
            r.get_rank_pids_for_node(node_id) if (r := training_rank_roster_box.value) is not None else {}
        ),
        training_rank_roster_box=training_rank_roster_box,
        on_recovery_duration=duration_cb,
        scrape_target_manager=scrape_target_manager,
        controller_exporter=controller_exporter,
    )

    return FtControllerBundle(controller=instance, subsystem_hub=subsystem_hub)


def _build_rollout_subsystem_specs(
    *,
    rollout_cell_ids: list[str] | None,
    rollout_alive_threshold_seconds: float | None,
    rollout_monitoring_alive_duration_seconds: float | None,
    monitoring_timeout_seconds: int,
    subsystem_hub: SubsystemHub,
    make_cooldown: Callable[[], SlidingWindowThrottle],
) -> dict[str, SubsystemSpec]:
    """Build SubsystemSpecs for each rollout cell."""
    from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
    from miles.utils.ft.controller.detectors.chain import build_rollout_detectors, build_shared_hw_detectors

    rollout_alive_dur = (
        rollout_monitoring_alive_duration_seconds if rollout_monitoring_alive_duration_seconds is not None else 180
    )
    rollout_det_kwargs: dict[str, float] = {}
    if rollout_alive_threshold_seconds is not None:
        rollout_det_kwargs["alive_threshold_seconds"] = rollout_alive_threshold_seconds

    specs: dict[str, SubsystemSpec] = {}
    for cell_id in rollout_cell_ids or []:
        name = f"rollout_{cell_id}"
        _cid = cell_id  # capture for closure
        specs[name] = SubsystemSpec(
            config=SubsystemConfig(
                restart_mode=RestartMode.SUBSYSTEM,
                detectors=build_shared_hw_detectors() + build_rollout_detectors(cell_id=cell_id, **rollout_det_kwargs),
                monitoring_config=MonitoringRunningAfterDelayConfig(
                    alive_duration_seconds=rollout_alive_dur,
                    timeout_seconds=monitoring_timeout_seconds,
                ),
            ),
            runtime=SubsystemRuntime(
                actuator=RayRolloutActuator(
                    get_handle=lambda: subsystem_hub.rollout_manager_handle,
                    cell_id=cell_id,
                ),
                cooldown=make_cooldown(),
                get_active_node_ids=lambda _c=_cid: subsystem_hub.get_rollout_node_ids(_c),
            ),
        )

    return specs
