from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.ft.adapters.types import (
    NodeManagerProtocol,
    NotifierProtocol,
    MainJobProtocol,
)
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.rank_roster import RankRoster
from miles.utils.ft.controller.state_machines.main import (
    DetectingAnomaly,
    MainContext,
    MainState,
    create_main_stepper,
)
from miles.utils.ft.controller.state_machines.recovery import (
    RECOVERY_TIMEOUT_SECONDS,
    create_recovery_stepper,
)
from miles.utils.ft.controller.state_machines.restart import RestartContext, create_restart_stepper
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
    rank_roster = RankRoster(scrape_target_manager=scrape_target_manager)

    resolved_orchestrator: DiagnosticOrchestratorProtocol = diagnostic_orchestrator or DiagnosticOrchestrator(
        agents=agents,
        pipeline=list(build_all_cluster_executors().values()),
    )

    resolved_exporter = controller_exporter or NullControllerExporter()
    duration_cb = resolved_exporter.observe_recovery_duration
    cooldown = recovery_cooldown or SlidingWindowThrottle(window_minutes=30.0, max_count=3)

    restart_stepper = create_restart_stepper()
    recovery_stepper = create_recovery_stepper()
    main_stepper = create_main_stepper()

    state_machine: StateMachine[MainState, MainContext] = StateMachine(
        initial_state=DetectingAnomaly(),
        stepper=main_stepper,
    )

    tick_loop = TickLoop(
        state_machine=state_machine,
        rank_roster=rank_roster,
        agents=agents,  # type: ignore[arg-type]
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        detectors=detectors or [],
        notifier=notifier,
        cooldown=cooldown,
        recovery_stepper=recovery_stepper,
        on_recovery_duration=duration_cb,
        max_simultaneous_bad_nodes=max_simultaneous_bad_nodes,
        diagnostic_orchestrator=resolved_orchestrator,
        restart_stepper=restart_stepper,
        restart_context=None,
        recovery_timeout_seconds=recovery_timeout_seconds,
        controller_exporter=controller_exporter,
        registration_grace_ticks=registration_grace_ticks,
    )

    platform_deps = PlatformDeps(
        node_manager=node_manager,
        main_job=main_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=notifier,
        diagnostic_orchestrator=resolved_orchestrator,
        controller_exporter=controller_exporter,
        on_new_run=None,
    )

    instance = FtController(
        main_job=main_job,
        state_machine=state_machine,
        rank_roster=rank_roster,
        mini_wandb=mini_wandb,
        scrape_target_manager=scrape_target_manager,
        agents=agents,  # type: ignore[arg-type]
        tick_interval=tick_interval,
        tick_loop=tick_loop,
        notifier=notifier,
        metric_store=metric_store,
        controller_exporter=controller_exporter,
    )

    restart_context = RestartContext(
        node_manager=node_manager,
        main_job=main_job,
        mini_wandb=mini_wandb,
        notifier=notifier,
        on_new_run=instance._activate_run,
        monitoring_success_iterations=monitoring_success_iterations,
        monitoring_timeout_seconds=monitoring_timeout_seconds,
        node_metadata=instance._node_metadata,
    )
    tick_loop._restart_context = restart_context

    platform_deps.on_new_run = instance._activate_run
    platform_deps.rank_pids_provider = lambda node_id: instance._rank_roster.get_rank_pids_for_node(node_id)

    return instance
