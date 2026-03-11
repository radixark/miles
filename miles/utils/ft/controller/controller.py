from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from miles.utils.ft.adapters.types import MainJobProtocol, NodeAgentProtocol, NotifierProtocol
from miles.utils.ft.controller.metrics.exporter import ControllerExporter, NullControllerExporter
from miles.utils.ft.controller.metrics.lifecycle import start_metric_store_task, stop_metric_store_task
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.training_rank_roster import TrainingRankRoster
from miles.utils.ft.controller.state_machines.main.context import MainContext
from miles.utils.ft.controller.state_machines.main.models import MainState, NormalState
from miles.utils.ft.controller.state_machines.subsystem.models import SubsystemContext, SubsystemState
from miles.utils.ft.controller.status import build_controller_status
from miles.utils.ft.controller.subsystem import MonitoringSustainedAliveConfig, SubsystemEntry
from miles.utils.ft.controller.tick_loop import TickLoop
from miles.utils.ft.controller.types import ControllerStatus, MetricStoreProtocol, ScrapeTargetManagerProtocol
from miles.utils.ft.utils.state_machine import StateMachine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RolloutSubsystemConfig:
    cell_id: str
    rm_handle: object
    get_active_node_ids: Callable[[], set[str]]


def _build_rollout_subsystem_entry(*, name: str, config: _RolloutSubsystemConfig) -> SubsystemEntry:
    from miles.utils.ft.adapters.impl.ray.rollout_actuator import RayRolloutActuator
    from miles.utils.ft.controller.detectors.chain import build_rollout_detectors, build_shared_hw_detectors
    from miles.utils.ft.controller.state_machines.subsystem import DetectingAnomaly, create_subsystem_stepper

    return SubsystemEntry(
        name=name,
        state_machine=StateMachine(
            initial_state=DetectingAnomaly(),
            stepper=create_subsystem_stepper(),
        ),
        actuator=RayRolloutActuator(rm_handle=config.rm_handle, cell_id=config.cell_id),
        has_level1_restart=True,
        detectors=build_shared_hw_detectors() + build_rollout_detectors(cell_id=config.cell_id),
        monitoring_config=MonitoringSustainedAliveConfig(alive_duration_seconds=180),
        get_active_node_ids=config.get_active_node_ids,
    )


class FtController:
    def __init__(
        self,
        *,
        main_job: MainJobProtocol,
        state_machine: StateMachine[MainState, MainContext],
        training_rank_roster: TrainingRankRoster,
        mini_wandb: MiniWandb,
        scrape_target_manager: ScrapeTargetManagerProtocol | None,
        agents: dict[str, NodeAgentProtocol],
        tick_interval: float,
        tick_loop: TickLoop,
        notifier: NotifierProtocol | None,
        metric_store: MetricStoreProtocol,
        controller_exporter: ControllerExporter | None = None,
    ) -> None:
        self._main_job = main_job
        self._state_machine = state_machine
        self._training_rank_roster = training_rank_roster
        self._mini_wandb = mini_wandb
        self._scrape_target_manager = scrape_target_manager
        self._agents = agents
        self._tick_interval = tick_interval
        self._tick_loop = tick_loop
        self._notifier = notifier
        self._metric_store = metric_store
        self._controller_exporter: ControllerExporter = controller_exporter or NullControllerExporter()

        self._roster_cell: list[TrainingRankRoster] | None = None
        self._rollout_configs: list[_RolloutSubsystemConfig] = []
        self._node_metadata: dict[str, dict[str, str]] = {}
        self._shutting_down: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def training_rank_roster(self) -> TrainingRankRoster:
        return self._training_rank_roster

    @property
    def mini_wandb(self) -> MiniWandb:
        return self._mini_wandb

    @property
    def _tick_count(self) -> int:
        return self._tick_loop.tick_count

    @property
    def node_metadata(self) -> dict[str, dict[str, str]]:
        return self._node_metadata

    @property
    def _training_state_machine(self) -> StateMachine[SubsystemState, SubsystemContext]:
        state = self._state_machine.state
        if not isinstance(state, NormalState):
            raise RuntimeError(f"Expected NormalState, got {type(state).__name__}")
        training = state.subsystems.get("training")
        if training is None:
            raise RuntimeError("No 'training' subsystem in NormalState")
        return training.state_machine

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
        if exporter_address and self._scrape_target_manager is not None:
            self._scrape_target_manager.add_scrape_target(
                target_id=node_id,
                address=exporter_address,
            )
        logger.info(
            "agent_registered node_id=%s exporter=%s metadata_keys=%s",
            node_id,
            exporter_address,
            sorted(node_metadata) if node_metadata else "(none)",
        )

    def register_rollout_subsystems(
        self,
        *,
        rm_handle: object,
        ft_rollout_agent: object,
    ) -> None:
        cell_ids: list[str] = ft_rollout_agent.get_cell_ids()  # type: ignore[union-attr]
        state = self._state_machine.state
        if not isinstance(state, NormalState):
            raise RuntimeError(
                f"Cannot register rollout subsystems in {type(state).__name__}, expected NormalState"
            )

        new_subsystems = dict(state.subsystems)
        for cell_id in cell_ids:
            cell_agent = ft_rollout_agent.get_cell_agent(cell_id)  # type: ignore[union-attr]
            config = _RolloutSubsystemConfig(
                cell_id=cell_id,
                rm_handle=rm_handle,
                get_active_node_ids=cell_agent.get_node_ids,
            )
            self._rollout_configs.append(config)
            name = f"rollout_{cell_id}"
            entry = _build_rollout_subsystem_entry(name=name, config=config)
            new_subsystems[name] = entry

        self._state_machine.state = NormalState(subsystems=new_subsystems)
        logger.info(
            "rollout_subsystems_registered cell_ids=%s total_subsystems=%d",
            cell_ids,
            len(new_subsystems),
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
            training_rank_roster=self._training_rank_roster,
            tick_count=self._tick_count,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _activate_run(self, run_id: str) -> None:
        """Create a fresh TrainingRankRoster for the new run and switch MiniWandb."""
        self._training_rank_roster.cleanup()
        self._training_rank_roster = TrainingRankRoster(
            run_id=run_id,
            scrape_target_manager=self._scrape_target_manager,
        )
        self._tick_loop.training_rank_roster = self._training_rank_roster
        if self._roster_cell is not None:
            self._roster_cell[0] = self._training_rank_roster
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
