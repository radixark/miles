from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.ft.adapters.impl.training_actuator import TrainingSubsystemActuator
from miles.utils.ft.adapters.types import MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.controller import (
    ControllerContext,
    NormalState,
    create_controller_stepper,
)
from miles.utils.ft.controller.state_machines.controller.models import ControllerState
from miles.utils.ft.controller.state_machines.main import DetectingAnomaly, MainContext, MainState, create_main_stepper
from miles.utils.ft.controller.state_machines.recovery import RECOVERY_TIMEOUT_SECONDS
from miles.utils.ft.controller.subsystem import MonitoringConfig, SubsystemEntry
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import (
    DiagnosticOrchestratorProtocol,
    MetricQueryProtocol,
    MetricStoreProtocol,
    ScrapeTargetManagerProtocol,
)
from miles.utils.ft.utils.sliding_window import SlidingWindowThrottle
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


@dataclass
class PlatformDeps:
    """Bundles platform-level dependencies shared across action handlers."""

    node_manager: NodeManagerProtocol
    main_job: MainJobProtocol
    metric_store: MetricQueryProtocol
    mini_wandb: MiniWandb
    notifier: NotifierProtocol | None
    diagnostic_orchestrator: DiagnosticOrchestratorProtocol
    controller_exporter: ControllerExporter | None
    on_new_run: Callable[[str], None] | None = field(default=None)
    rank_pids_provider: Callable[[str], dict[int, int]] | None = field(default=None)


def create_ft_controller(
    node_manager: NodeManagerProtocol,
    main_job: MainJobProtocol,
    metric_store: MetricStoreProtocol,
    mini_wandb: MiniWandb,
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
) -> FtController:
    from miles.utils.ft.controller.diagnostics.executors import build_all_cluster_executors
    from miles.utils.ft.controller.diagnostics.orchestrator import DiagnosticOrchestrator

    agents: dict[str, object] = {}
    training_rank_roster = TrainingRankRoster(scrape_target_manager=scrape_target_manager)

    resolved_orchestrator: DiagnosticOrchestratorProtocol = diagnostic_orchestrator or DiagnosticOrchestrator(
        agents=agents,
        pipeline=list(build_all_cluster_executors().values()),
    )

    resolved_exporter = controller_exporter or NullControllerExporter()
    duration_cb = resolved_exporter.observe_recovery_duration
    cooldown = recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3)
    resolved_detectors = detectors or []

    monitoring_config = MonitoringConfig(
        mode="iteration_progress",
        success_iterations=monitoring_success_iterations,
        timeout_seconds=monitoring_timeout_seconds,
    )

    # --- Create SubsystemEntry ---
    main_stepper = create_main_stepper()

    def _make_training_sub_sm() -> StateMachine[MainState, MainContext]:
        return StateMachine(
            initial_state=DetectingAnomaly(),
            stepper=main_stepper,
        )

    training_entry = SubsystemEntry(
        name="training",
        state_machine=_make_training_sub_sm(),
        actuator=TrainingSubsystemActuator(main_job=main_job),
        detectors=resolved_detectors,
        monitoring_config=monitoring_config,
        get_active_node_ids=lambda: set(training_rank_roster.rank_placement.values()),
    )

    # --- Create Controller SM ---
    controller_stepper = create_controller_stepper()
    initial_subsystems = {"training": training_entry}
    controller_sm: StateMachine[ControllerState, ControllerContext] = StateMachine(
        initial_state=NormalState(subsystems=initial_subsystems),
        stepper=controller_stepper,
    )

    # --- create_fresh_subsystems callback ---
    def create_fresh_subsystems() -> dict[str, SubsystemEntry]:
        return {
            "training": SubsystemEntry(
                name="training",
                state_machine=_make_training_sub_sm(),
                actuator=TrainingSubsystemActuator(main_job=main_job),
                detectors=resolved_detectors,
                monitoring_config=monitoring_config,
                get_active_node_ids=lambda: set(training_rank_roster.rank_placement.values()),
            ),
        }

    # --- Create FtController ---
    instance = FtController(
        main_job=main_job,
        state_machine=controller_sm,
        training_rank_roster=training_rank_roster,
        mini_wandb=mini_wandb,
        scrape_target_manager=scrape_target_manager,
        agents=agents,  # type: ignore[arg-type]
        tick_interval=tick_interval,
        tick_loop=None,  # type: ignore[arg-type]  # set below
        notifier=notifier,
        metric_store=metric_store,
        controller_exporter=controller_exporter,
    )

    # --- Create TickLoop ---
    tick_loop = TickLoop(
        state_machine=controller_sm,
        training_rank_roster=training_rank_roster,
        agents=agents,  # type: ignore[arg-type]
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        node_manager=node_manager,
        cooldown=cooldown,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        diagnostic_orchestrator=resolved_orchestrator,
        recovery_timeout_seconds=recovery_timeout_seconds,
        create_fresh_subsystems=create_fresh_subsystems,
        on_new_run=instance._activate_run,
        rank_pids_provider=lambda node_id: instance._training_rank_roster.get_rank_pids_for_node(node_id),
        on_recovery_duration=duration_cb,
        controller_exporter=controller_exporter,
        registration_grace_ticks=registration_grace_ticks,
    )
    instance._tick_loop = tick_loop

    platform_deps = PlatformDeps(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        diagnostic_orchestrator=resolved_orchestrator,
        controller_exporter=controller_exporter,
        on_new_run=instance._activate_run,
        rank_pids_provider=lambda node_id: instance._training_rank_roster.get_rank_pids_for_node(node_id),
    )

    return instance
