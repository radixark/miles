from __future__ import annotations

import logging
from typing import NamedTuple

from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.node_agents import NodeAgentRegistry
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.subsystem_hub import RestartMode, SubsystemConfig, SubsystemHub, TrainingRankRoster
from miles.utils.ft.controller.state_machines.main import (
    MainContext,
    NormalSt,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.main.models import MainState
from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomalySt
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_TIMEOUT_SECONDS
from miles.utils.ft.controller.state_machines.restart.models import (
    MonitoringIterationProgressConfig,
    MonitoringSustainedAliveConfig,
)
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import (
    DiagnosticOrchestratorProtocol,
    MetricStore,
    ScrapeTargetManagerProtocol,
)
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


class FtControllerBundle(NamedTuple):
    controller: FtController
    subsystem_hub: SubsystemHub


def create_ft_controller(
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
    recovery_cooldown: SlidingWindowThrottle | None = None,
    registration_grace_ticks: int = 5,
    max_simultaneous_bad_nodes: int = 3,
    monitoring_success_iterations: int = 10,
    monitoring_timeout_seconds: int = 600,
    recovery_timeout_seconds: int = RECOVERY_TIMEOUT_SECONDS,
    rollout_alive_threshold_seconds: float | None = None,
    rollout_monitoring_alive_duration_seconds: float | None = None,
) -> FtControllerBundle:
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
    cooldown = recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3)
    resolved_detectors = detectors or []

    monitoring_config = MonitoringIterationProgressConfig(
        success_iterations=monitoring_success_iterations,
        timeout_seconds=monitoring_timeout_seconds,
    )

    # --- Training SubsystemConfig ---
    training_config = SubsystemConfig(
        actuator=TrainingSubsystemActuator(main_job=main_job),
        restart_mode=RestartMode.MAIN_JOB,
        detectors=resolved_detectors,
        monitoring_config=monitoring_config,
        get_active_node_ids=_get_active_training_nodes,
    )
    subsystem_configs: dict[str, SubsystemConfig] = {"training": training_config}

    # --- Rollout SubsystemConfigs ---
    rollout_alive_dur = rollout_monitoring_alive_duration_seconds if rollout_monitoring_alive_duration_seconds is not None else 180
    rollout_det_kwargs: dict[str, float] = {}
    if rollout_alive_threshold_seconds is not None:
        rollout_det_kwargs["alive_threshold_seconds"] = rollout_alive_threshold_seconds

    for cell_id in (rollout_cell_ids or []):
        name = f"rollout_{cell_id}"
        _cid = cell_id  # capture for closure
        subsystem_configs[name] = SubsystemConfig(
            actuator=RayRolloutActuator(
                get_handle=lambda: subsystem_hub.rollout_manager_handle,
                cell_id=cell_id,
            ),
            restart_mode=RestartMode.SUBSYSTEM,
            detectors=build_shared_hw_detectors() + build_rollout_detectors(cell_id=cell_id, **rollout_det_kwargs),
            monitoring_config=MonitoringSustainedAliveConfig(alive_duration_seconds=rollout_alive_dur),
            get_active_node_ids=lambda _c=_cid: subsystem_hub.get_rollout_node_ids(_c),
        )

    # --- Create Main SM (includes all subsystems from the start) ---
    initial_subsystem_states = {name: DetectingAnomalySt() for name in subsystem_configs}
    main_stepper = create_main_stepper()
    controller_sm: StateMachine[MainState, MainContext] = StateMachine(
        initial_state=NormalSt(subsystems=initial_subsystem_states),
        stepper=main_stepper,
    )

    # --- Create FtController ---
    instance = FtController(
        main_job=main_job,
        state_machine=controller_sm,
        subsystem_hub=subsystem_hub,
        metric_store=metric_store,
        node_agent_registry=node_agent_registry,
        tick_interval=tick_interval,
        tick_loop=None,  # type: ignore[arg-type]  # set below
        notifier=notifier,
        scrape_target_manager=scrape_target_manager,
        controller_exporter=controller_exporter,
    )

    # --- Create TickLoop ---
    tick_loop = TickLoop(
        state_machine=controller_sm,
        training_rank_roster_box=training_rank_roster_box,
        node_agent_registry=node_agent_registry,
        main_job=main_job,
        metric_store=metric_store,
        notifier=notifier,
        node_manager=node_manager,
        cooldown=cooldown,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        diagnostic_orchestrator=resolved_orchestrator,
        recovery_timeout_seconds=recovery_timeout_seconds,
        subsystem_configs=subsystem_configs,
        on_new_run=instance._activate_run,
        rank_pids_provider=lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := training_rank_roster_box.value) is not None else {}),
        on_recovery_duration=duration_cb,
        controller_exporter=controller_exporter,
        registration_grace_ticks=registration_grace_ticks,
    )
    instance._tick_loop = tick_loop

    return FtControllerBundle(controller=instance, subsystem_hub=subsystem_hub)
