from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.adapters.types import MainJobProtocol, NodeAgentProtocol, NotifierProtocol
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.state_machines.main.models import MainContext, MainState
from miles.utils.ft.controller.status import build_controller_status
from miles.utils.ft.controller.subsystem_hub import SubsystemHub
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.types import ControllerStatus, MetricStoreProtocol, ScrapeTargetManagerProtocol
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


class FtController:
    def __init__(
        self,
        *,
        main_job: MainJobProtocol,
        state_machine: StateMachine[MainState, MainContext],
        subsystem_hub: SubsystemHub,
        mini_wandb: MiniWandb,
        agents: dict[str, NodeAgentProtocol],
        tick_interval: float,
        tick_loop: TickLoop,
        notifier: NotifierProtocol | None,
        metric_store: MetricStoreProtocol,
        scrape_target_manager: ScrapeTargetManagerProtocol | None = None,
        controller_exporter: ControllerExporter | None = None,
    ) -> None:
        self._main_job = main_job
        self._state_machine = state_machine
        self._subsystem_hub = subsystem_hub
        self._mini_wandb = mini_wandb
        self._agents = agents
        self._tick_interval = tick_interval
        self._tick_loop = tick_loop
        self._notifier = notifier
        self._metric_store = metric_store
        self._scrape_target_manager = scrape_target_manager
        self._controller_exporter: ControllerExporter = controller_exporter or NullControllerExporter()

        self._node_metadata: dict[str, dict[str, str]] = {}
        self._shutting_down: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def training_rank_roster(self) -> TrainingRankRoster:
        return self._subsystem_hub.training_rank_roster

    @property
    def mini_wandb(self) -> MiniWandb:
        return self._mini_wandb

    @property
    def node_metadata(self) -> dict[str, dict[str, str]]:
        return self._node_metadata

    def add_scrape_target(self, target_id: str, address: str) -> None:
        if self._scrape_target_manager is not None:
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
        self._agents[node_id] = agent
        if node_metadata:
            self._node_metadata[node_id] = node_metadata
        if exporter_address:
            self.add_scrape_target(target_id=node_id, address=exporter_address)
        logger.info(
            "agent_registered node_id=%s exporter=%s metadata_keys=%s",
            node_id,
            exporter_address,
            sorted(node_metadata) if node_metadata else "(none)",
        )

    async def submit_initial_job(self) -> str:
        run_id = await self._main_job.submit_job()
        self._activate_run(run_id)
        return run_id

    async def run(self) -> None:
        logger.info("controller_start tick_interval=%s", self._tick_interval)
        scrape_task = await start_metric_store_task(self._metric_store)
        try:
            while not self._shutting_down:
                await self._tick()
                if not self._shutting_down:
                    await asyncio.sleep(self._tick_interval)
        finally:
            await self._stop_services(scrape_task)
        logger.info("controller_stopped")

    async def shutdown(self) -> None:
        logger.info("controller_shutdown_requested")
        self._shutting_down = True

    def get_status(self) -> ControllerStatus:
        return build_controller_status(
            controller_state_machine=self._state_machine,
            mini_wandb=self._mini_wandb,
            training_rank_roster=self._subsystem_hub.training_rank_roster_box.value,
            tick_count=self._tick_count,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _tick_count(self) -> int:
        return self._tick_loop.tick_count

    def _activate_run(self, run_id: str) -> None:
        """Create a fresh TrainingRankRoster for the new run and switch MiniWandb."""
        old_roster = self._subsystem_hub.training_rank_roster_box.value
        if old_roster is not None:
            old_roster.cleanup()
        self._subsystem_hub.training_rank_roster_box.value = TrainingRankRoster(
            run_id=run_id,
            scrape_target_manager=self._scrape_target_manager,
        )
        self._mini_wandb.set_active_run_id(run_id)
        logger.info("run_activated run_id=%s", run_id)

    async def _tick(self) -> None:
        await self._tick_loop.tick()

    async def _stop_services(self, scrape_task: asyncio.Task[None] | None) -> None:
        await stop_metric_store_task(self._metric_store, scrape_task)
        self._controller_exporter.stop()
        if self._notifier is not None:
            try:
                await self._notifier.aclose()
            except Exception:
                logger.warning("notifier_aclose_failed", exc_info=True)
