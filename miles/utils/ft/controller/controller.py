from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import NamedTuple

from miles.utils.ft.adapters.types import MainJobProtocol, NodeAgentProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.node_agents import NodeAgentRegistry
from miles.utils.ft.controller.runtime_config import ControllerRuntimeConfig
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState
from miles.utils.ft.controller.status import build_controller_status
from miles.utils.ft.controller.subsystem_hub import SubsystemHub, SubsystemSpec, TrainingRankRoster
from miles.utils.ft.controller.tick_loop import TickDeps, TickLoop
from miles.utils.ft.controller.types import (
    ControllerStatus,
    DiagnosticOrchestratorProtocol,
    MetricStore,
    NullScrapeTargetManager,
    ScrapeTargetManagerProtocol,
)
from miles.utils.ft.utils.box import Box
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


class FtControllerBundle(NamedTuple):
    controller: FtController
    subsystem_hub: SubsystemHub


class FtController:
    def __init__(
        self,
        *,
        runtime_config: ControllerRuntimeConfig,
        main_job: MainJobProtocol,
        state_machine: StateMachine[MainState, MainContext],
        subsystem_hub: SubsystemHub,
        metric_store: MetricStore,
        node_agent_registry: NodeAgentRegistry,
        tick_loop: TickLoop,
        notifier: NotifierProtocol | None,
        node_manager: NodeManagerProtocol,
        diagnostic_orchestrator: DiagnosticOrchestratorProtocol,
        subsystem_specs: dict[str, SubsystemSpec],
        rank_pids_provider: Callable[[str], dict[int, int]],
        training_rank_roster_box: Box[TrainingRankRoster | None],
        on_recovery_duration: Callable[[float], None] | None = None,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
        controller_exporter: ControllerExporter | None = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._main_job = main_job
        self._state_machine = state_machine
        self._subsystem_hub = subsystem_hub
        self._metric_store = metric_store
        self._node_agent_registry = node_agent_registry
        self._tick_loop = tick_loop
        self._notifier = notifier
        self._node_manager = node_manager
        self._diagnostic_orchestrator = diagnostic_orchestrator
        self._subsystem_specs = subsystem_specs
        self._rank_pids_provider = rank_pids_provider
        self._training_rank_roster_box = training_rank_roster_box
        self._on_recovery_duration = on_recovery_duration
        self._scrape_target_manager: ScrapeTargetManagerProtocol = scrape_target_manager or NullScrapeTargetManager()
        self._controller_exporter: ControllerExporter = controller_exporter or NullControllerExporter()

        self._shutting_down: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def metric_store(self) -> MetricStore:
        return self._metric_store

    @property
    def node_metadata(self) -> dict[str, dict[str, str]]:
        return self._node_agent_registry.all_metadata

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._scrape_target_manager.add_scrape_target(
            target_id=target_id,
            address=address,
        )

    def register_node_agent(
        self,
        node_id: str,
        agent: NodeAgentProtocol,
        exporter_address: str = "",
        node_metadata: dict[str, str] | None = None,
    ) -> None:
        self._node_agent_registry.register(node_id=node_id, agent=agent, metadata=node_metadata)
        if exporter_address:
            self.add_scrape_target(target_id=node_id, address=exporter_address)
        logger.info(
            "agent_registered node_id=%s exporter=%s metadata_keys=%s",
            node_id,
            exporter_address,
            sorted(node_metadata) if node_metadata else "(none)",
        )

    async def submit_initial_job(self) -> str:
        run_id = await self._main_job.start()
        self._activate_run(run_id)
        return run_id

    async def run(self) -> None:
        logger.info("controller_start tick_interval=%s", self._runtime_config.tick_interval)
        scrape_task = await start_metric_store_task(self._metric_store.time_series_store)
        try:
            while not self._shutting_down:
                await self._tick()
                if not self._shutting_down:
                    await asyncio.sleep(self._runtime_config.tick_interval)
        finally:
            await self._stop_services(scrape_task)
        logger.info("controller_stopped")

    async def shutdown(self) -> None:
        logger.info("controller_shutdown_requested")
        self._shutting_down = True

    def get_status(self) -> ControllerStatus:
        return build_controller_status(
            controller_state_machine=self._state_machine,
            mini_wandb=self._metric_store.mini_wandb,
            training_rank_roster=self._subsystem_hub.training_rank_roster_box.value,
            tick_count=self._tick_loop.tick_count,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _activate_run(self, run_id: str) -> None:
        """Create a fresh TrainingRankRoster for the new run and switch MiniWandb."""
        old_roster = self._subsystem_hub.training_rank_roster_box.value
        self._subsystem_hub.training_rank_roster_box.value = TrainingRankRoster(
            run_id=run_id,
            scrape_target_manager=self._scrape_target_manager,
        )
        if old_roster is not None:
            old_roster.cleanup()
        self._metric_store.mini_wandb.set_active_run_id(run_id)
        logger.info("run_activated run_id=%s", run_id)

    def _build_tick_deps(self) -> TickDeps:
        return TickDeps(
            main_job=self._main_job,
            metric_store=self._metric_store,
            notifier=self._notifier,
            node_manager=self._node_manager,
            diagnostic_orchestrator=self._diagnostic_orchestrator,
            recovery_timeout_seconds=self._runtime_config.recovery_timeout_seconds,
            max_simultaneous_bad_nodes=self._runtime_config.max_simultaneous_bad_nodes,
            subsystem_specs=self._subsystem_specs,
            on_main_job_new_run=self._activate_run,
            rank_pids_provider=self._rank_pids_provider,
            on_recovery_duration=self._on_recovery_duration,
            controller_exporter=self._controller_exporter,
            registration_grace_ticks=self._runtime_config.registration_grace_ticks,
            training_rank_roster_box=self._training_rank_roster_box,
            node_agent_registry=self._node_agent_registry,
        )

    async def _tick(self) -> None:
        await self._tick_loop.tick(self._build_tick_deps())

    async def _stop_services(self, scrape_task: asyncio.Task[None]) -> None:
        await stop_metric_store_task(self._metric_store.time_series_store, scrape_task)
        self._controller_exporter.stop()
        if self._notifier is not None:
            try:
                await self._notifier.aclose()
            except Exception:
                logger.warning("notifier_aclose_failed", exc_info=True)
