from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import ray

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
from miles.utils.ft.adapters.impl.ray.node_agent_actor import FtNodeAgentActor
from miles.utils.ft.adapters.types import ft_controller_actor_name, ft_node_agent_actor_name
from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.utils.metric_names import XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL
from miles.utils.ft.controller.types import ControllerMode, ControllerStatus
from miles.utils.ft.factories.controller.from_config import build_ft_controller
from miles.utils.ft.factories.node_agent import build_node_agent
from tests.fast.utils.ft.testbed.config import TestbedConfig
from tests.fast.utils.ft.testbed.ray.actor_group import TestbedRayTrainGroup
from tests.fast.utils.ft.testbed.ray.rollout import TestbedRolloutManager
from tests.fast.utils.ft.testbed.utils.ft.adapters.impl.k8s_node_manager import TestbedNodeManager
from tests.fast.utils.ft.testbed.utils.ft.adapters.impl.notifiers.webhook_notifier import TestbedNotifier
from tests.fast.utils.ft.testbed.utils.ft.adapters.impl.ray.main_job import TestbedMainJob
from tests.fast.utils.ft.testbed.utils.ft.agents.collectors.collector import TestbedCollector
from tests.fast.utils.ft.utils.diagnostic_fakes import StubDiagnostic

if TYPE_CHECKING:
    from tests.fast.utils.ft.integration.conftest import RayNodeInfo

logger = logging.getLogger(__name__)


@dataclass
class _NodeMapping:
    """Maps logical node IDs to Ray node IDs."""
    training: dict[str, str]
    rollout: dict[str, str]
    spare: dict[str, str]


class MilesTestbed:
    """High-fidelity miles FT system test environment.

    Only fakes leaf dependencies (training computation, sglang, K8s, collectors).
    All intermediate FT logic (controller, state machines, detectors, metrics pipeline)
    runs real production code.
    """

    def __init__(
        self,
        config: TestbedConfig,
        ft_id: str,
        controller: ray.actor.ActorHandle,
        train_group: TestbedRayTrainGroup,
        node_manager: TestbedNodeManager,
        notifier: TestbedNotifier | None,
        collector_states: dict[str, ray.actor.ActorHandle],
        rollout_manager: ray.actor.ActorHandle | None,
        cleanup_handles: list[ray.actor.ActorHandle],
        cleanup_names: list[str],
    ) -> None:
        self._config = config
        self._ft_id = ft_id
        self._controller = controller
        self._train_group = train_group
        self._node_manager = node_manager
        self._notifier = notifier
        self._collector_states = collector_states
        self._rollout_manager = rollout_manager
        self._cleanup_handles = cleanup_handles
        self._cleanup_names = cleanup_names

    @classmethod
    async def launch(
        cls,
        config: TestbedConfig,
        ray_nodes: list[RayNodeInfo],
    ) -> MilesTestbed:
        ft_id = uuid4().hex[:8]
        cleanup_handles: list[ray.actor.ActorHandle] = []
        cleanup_names: list[str] = []

        # Step 1: Build logical → Ray node mapping
        node_mapping = _build_node_mapping(
            config=config,
            ray_nodes=ray_nodes,
        )

        # Step 2: Create state actors for node manager, notifier
        node_manager = TestbedNodeManager.create()
        cleanup_handles.append(node_manager.state_actor)

        notifier: TestbedNotifier | None = None
        if config.notifier_override is None:
            notifier = TestbedNotifier.create()
            cleanup_handles.append(notifier.state_actor)
        resolved_notifier = config.notifier_override if config.notifier_override is not None else notifier

        # Step 3: Prepare collector state actors (before node agents)
        collector_states: dict[str, ray.actor.ActorHandle] = {}
        all_node_configs = list(config.training_nodes) + list(config.rollout_nodes)
        all_node_mapping = {**node_mapping.training, **node_mapping.rollout}

        collectors_by_node: dict[str, TestbedCollector] = {}
        for node_config in all_node_configs:
            collector, collector_state = TestbedCollector.create(node_id=node_config.node_id)
            collector_states[node_config.node_id] = collector_state
            collectors_by_node[node_config.node_id] = collector
            cleanup_handles.append(collector_state)

        # Step 4: Create TestbedRayTrainGroup + TestbedMainJob
        train_group = TestbedRayTrainGroup(
            training_nodes=config.training_nodes,
            node_mapping=node_mapping.training,
            ft_id=ft_id,
            step_interval=config.step_interval,
        )
        cleanup_handles.append(train_group._store)
        main_job = TestbedMainJob(train_group=train_group)

        # Step 5: Create FtControllerActor with real build_ft_controller
        detectors = config.detectors if config.detectors is not None else build_detector_chain()

        controller_kwargs: dict[str, object] = dict(
            builder=build_ft_controller,
            config=FtControllerConfig(
                platform="stub",
                tick_interval=config.tick_interval,
                ft_id=ft_id,
                scrape_interval_seconds=config.scrape_interval_seconds,
                rollout_num_cells=config.rollout_num_cells,
            ),
            runtime_config_override=config.build_runtime_config(),
            main_job_override=main_job,
            node_manager_override=node_manager,
            notifier_override=resolved_notifier,
            detectors_override=detectors,
            diagnostic_orchestrator_override=config.diagnostic_orchestrator_override,
            start_exporter=True,
        )

        controller_name = ft_controller_actor_name(ft_id)
        controller = FtControllerActor.options(name=controller_name).remote(**controller_kwargs)
        cleanup_names.append(controller_name)

        # Step 6: Fire-and-forget submit_and_run (starts main job + tick loop)
        controller.submit_and_run.remote()
        await _poll_for_run_id(controller)

        # Step 7: Deploy FtNodeAgentActor per node (AFTER controller exists,
        #         because node agent start() registers with controller)
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        for node_config in all_node_configs:
            node_id = node_config.node_id
            ray_node_id = all_node_mapping[node_id]

            diagnostics = [
                StubDiagnostic(
                    passed=node_config.diagnostic_pass,
                    diagnostic_type=dt,
                )
                for dt in ["gpu", "nccl_simple", "nccl_pairwise"]
            ]

            agent_name = ft_node_agent_actor_name(ft_id, node_id)
            node_agent = FtNodeAgentActor.options(
                name=agent_name,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray_node_id,
                    soft=False,
                ),
            ).remote(
                builder=build_node_agent,
                node_id=node_id,
                ft_id=ft_id,
                collect_interval_seconds=0.3,
                collectors_override=[collectors_by_node[node_id]],
                diagnostics_override=diagnostics,
            )
            cleanup_names.append(agent_name)

            ray.get(node_agent.start.remote(), timeout=15)
            logger.info("Node agent deployed: %s → ray_node %s", node_id, ray_node_id)

        # Step 8: If rollout configured, create TestbedRolloutManager
        rollout_manager: ray.actor.ActorHandle | None = None
        if config.rollout_num_cells > 0 and config.rollout_nodes:
            rollout_cell_ids = _rollout_num_cells_to_ids(config.rollout_num_cells)

            rollout_node_mapping: dict[str, str] = {}
            rollout_node_ids_for_cells: dict[str, set[str]] = {}
            for i, cell_id in enumerate(rollout_cell_ids):
                rollout_node_idx = i % len(config.rollout_nodes)
                rollout_node = config.rollout_nodes[rollout_node_idx]
                ray_rollout_node_id = node_mapping.rollout[rollout_node.node_id]
                rollout_node_mapping[cell_id] = ray_rollout_node_id
                rollout_node_ids_for_cells[cell_id] = {rollout_node.node_id}

            rollout_manager = TestbedRolloutManager.remote(
                ft_id=ft_id,
                cell_ids=rollout_cell_ids,
                rollout_node_mapping=rollout_node_mapping,
            )
            cleanup_handles.append(rollout_manager)

            ray.get(rollout_manager.init_ft_agent.remote(), timeout=15)

            for cell_id, node_ids in rollout_node_ids_for_cells.items():
                ray.get(
                    controller.set_rollout_node_ids.remote(cell_id, node_ids),
                    timeout=5,
                )

        # Step 9: Wait for training to stabilize
        testbed = cls(
            config=config,
            ft_id=ft_id,
            controller=controller,
            train_group=train_group,
            node_manager=node_manager,
            notifier=notifier,
            collector_states=collector_states,
            rollout_manager=rollout_manager,
            cleanup_handles=cleanup_handles,
            cleanup_names=cleanup_names,
        )
        if config.initial_stable_iterations > 0:
            await testbed.wait_for_training_stable(
                n_iterations=config.initial_stable_iterations,
                timeout=30.0,
            )
        return testbed

    async def shutdown(self) -> None:
        logger.info("MilesTestbed.shutdown: ft_id=%s", self._ft_id)
        try:
            ray.get(self._controller.shutdown.remote(), timeout=10)
        except Exception:
            logger.debug("shutdown: controller shutdown failed", exc_info=True)

        await self._train_group.kill_all()

        for name in self._cleanup_names:
            try:
                handle = ray.get_actor(name)
                ray.kill(handle, no_restart=True)
            except Exception:
                logger.debug("shutdown: failed to kill actor %s", name, exc_info=True)

        for handle in self._cleanup_handles:
            try:
                ray.kill(handle, no_restart=True)
            except Exception:
                logger.debug("shutdown: failed to kill handle", exc_info=True)

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    async def get_status(self) -> ControllerStatus:
        return ray.get(self._controller.get_status.remote(), timeout=5)

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    async def kill_training_on_node(self, node_id: str) -> None:
        logger.info("MilesTestbed.kill_training_on_node: node_id=%s", node_id)
        await self._train_group.kill_on_node(node_id)

    async def crash_training(self) -> None:
        logger.info("MilesTestbed.crash_training: injecting fault")
        await self._train_group.kill_all()

    async def inject_gpu_xid(self, node_id: str, count: float = 1.0) -> None:
        state = self._collector_states.get(node_id)
        if state is None:
            raise KeyError(f"No collector state for node {node_id}")
        labels = {"node_id": node_id}
        await state.set_metrics.remote([
            GaugeSample(
                name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
                labels=labels,
                value=count,
            ),
        ])

    async def kill_sglang_cell(self, cell_id: str) -> None:
        if self._rollout_manager is None:
            raise RuntimeError("No rollout manager configured")
        await self._rollout_manager.kill_engine.remote(cell_id)

    async def inject_collector_metrics(
        self,
        node_id: str,
        metrics: list[GaugeSample],
    ) -> None:
        state = self._collector_states.get(node_id)
        if state is None:
            raise KeyError(f"No collector state for node {node_id}")
        await state.set_metrics.remote(metrics)

    async def inject_hang(self) -> None:
        logger.info("MilesTestbed.inject_hang")
        await self._train_group.set_hung(hung=True)

    async def clear_hang(self) -> None:
        logger.info("MilesTestbed.clear_hang")
        await self._train_group.set_hung(hung=False)

    async def inject_nan_loss(self) -> None:
        logger.info("MilesTestbed.inject_nan_loss")
        await self._train_group.set_custom_log_metrics({"loss": float("nan")})

    async def clear_nan_loss(self) -> None:
        logger.info("MilesTestbed.clear_nan_loss")
        await self._train_group.set_custom_log_metrics({})

    async def inject_custom_metrics(self, metrics: dict[str, float]) -> None:
        await self._train_group.set_custom_log_metrics(metrics)

    async def clear_collector_metrics(self, node_id: str) -> None:
        state = self._collector_states.get(node_id)
        if state is None:
            raise KeyError(f"No collector state for node {node_id}")
        await state.set_metrics.remote([])

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    async def wait_for_training_stable(
        self,
        n_iterations: int = 3,
        timeout: float = 60.0,
    ) -> None:
        """Wait until iteration metrics are advancing on the controller."""
        deadline = time.monotonic() + timeout
        baseline: int | None = None

        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                current = status.latest_iteration
                if current is not None and current > 0:
                    if baseline is None:
                        baseline = current
                    elif current >= baseline + n_iterations:
                        return
            except Exception:
                logger.debug("wait_for_training_stable: status check failed", exc_info=True)
            await asyncio.sleep(0.3)

        raise TimeoutError(
            f"Training not stable after {timeout}s (baseline={baseline})"
        )

    async def wait_for_subsystem_state(
        self,
        name: str,
        state: str,
        timeout: float = 120.0,
    ) -> ControllerStatus:
        """Wait until a subsystem reaches the specified state name."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                current_state = status.subsystem_states.get(name, "")
                if state in current_state:
                    return status
            except Exception:
                logger.debug("wait_for_subsystem_state: status check failed", exc_info=True)
            await asyncio.sleep(0.3)

        last_status = await self.get_status()
        raise TimeoutError(
            f"Subsystem '{name}' did not reach state '{state}' within {timeout}s. "
            f"Current states: {last_status.subsystem_states}"
        )

    async def wait_for_recovery_phase(
        self,
        phase: str,
        timeout: float = 120.0,
    ) -> ControllerStatus:
        """Wait until recovery.phase contains the given string."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                if status.recovery is not None and phase in status.recovery.phase:
                    return status
            except Exception:
                logger.debug("wait_for_recovery_phase: status check failed", exc_info=True)
            await asyncio.sleep(0.3)
        raise TimeoutError(f"Recovery phase did not reach '{phase}' within {timeout}s")

    async def wait_for_mode(
        self,
        mode: ControllerMode,
        timeout: float = 60.0,
    ) -> ControllerStatus:
        """Wait until controller mode matches."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                if status.mode == mode:
                    return status
            except Exception:
                logger.debug("wait_for_mode: status check failed", exc_info=True)
            await asyncio.sleep(0.3)
        raise TimeoutError(f"Mode did not reach {mode} within {timeout}s")

    async def wait_for_mode_transition(
        self,
        target_mode: ControllerMode,
        timeout: float = 120.0,
    ) -> ControllerStatus:
        """Wait for mode to leave target, then return when it reaches target again."""
        deadline = time.monotonic() + timeout
        last_logged_mode: ControllerMode | None = None

        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                if status.mode != last_logged_mode:
                    logger.info("wait_for_mode_transition: phase=leave current=%s target=%s", status.mode, target_mode)
                    last_logged_mode = status.mode
                if status.mode != target_mode:
                    break
            except Exception:
                logger.debug("wait_for_mode_transition: status check failed", exc_info=True)
            await asyncio.sleep(0.3)

        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                if status.mode != last_logged_mode:
                    logger.info(
                        "wait_for_mode_transition: phase=return current=%s target=%s subsystems=%s",
                        status.mode, target_mode, status.subsystem_states,
                    )
                    last_logged_mode = status.mode
                if status.mode == target_mode:
                    return status
            except Exception:
                logger.debug("wait_for_mode_transition: status check failed", exc_info=True)
            await asyncio.sleep(0.3)

        last_status = await self.get_status()
        raise TimeoutError(
            f"Mode transition to {target_mode} did not complete within {timeout}s. "
            f"Last mode={last_status.mode} subsystems={last_status.subsystem_states}"
        )

    async def wait_for_all_subsystems_detecting(
        self,
        timeout: float = 120.0,
    ) -> ControllerStatus:
        """Wait until all subsystems are in DetectingAnomalySt."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = await self.get_status()
                if status.subsystem_states and all(
                    s == "DetectingAnomalySt" for s in status.subsystem_states.values()
                ):
                    return status
            except Exception:
                logger.debug("wait_for_all_subsystems_detecting: check failed", exc_info=True)
            await asyncio.sleep(0.3)
        last_status = await self.get_status()
        raise TimeoutError(
            f"Not all subsystems reached DetectingAnomalySt within {timeout}s "
            f"(states={last_status.subsystem_states})"
        )

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    @property
    def node_manager(self) -> TestbedNodeManager:
        return self._node_manager

    @property
    def notifications(self) -> list[tuple[str, str, str]]:
        if self._notifier is None:
            raise RuntimeError("No TestbedNotifier; notifier_override was used")
        return self._notifier.calls

    @property
    def ft_id(self) -> str:
        return self._ft_id

    @property
    def controller(self) -> ray.actor.ActorHandle:
        return self._controller

    @property
    def train_group(self) -> TestbedRayTrainGroup:
        return self._train_group

    @property
    def collector_states(self) -> dict[str, ray.actor.ActorHandle]:
        return self._collector_states

    @property
    def node_agents(self) -> dict[str, str]:
        """node_id → actor_name mapping for direct actor access."""
        from miles.utils.ft.adapters.types import ft_node_agent_actor_name
        return {
            node_config.node_id: ft_node_agent_actor_name(self._ft_id, node_config.node_id)
            for node_config in (
                list(self._config.training_nodes) + list(self._config.rollout_nodes)
            )
        }


def _build_node_mapping(
    config: TestbedConfig,
    ray_nodes: list[RayNodeInfo],
) -> _NodeMapping:
    """Assign logical node IDs to Ray nodes."""
    available = list(ray_nodes)
    needed = (
        len(config.training_nodes)
        + len(config.rollout_nodes)
        + len(config.spare_nodes)
    )
    if len(available) < needed:
        raise ValueError(
            f"Need {needed} Ray nodes but only {len(available)} available"
        )

    idx = 0
    training_map: dict[str, str] = {}
    for node in config.training_nodes:
        training_map[node.node_id] = available[idx].ray_node_id
        idx += 1

    rollout_map: dict[str, str] = {}
    for node in config.rollout_nodes:
        rollout_map[node.node_id] = available[idx].ray_node_id
        idx += 1

    spare_map: dict[str, str] = {}
    for spare_id in config.spare_nodes:
        spare_map[spare_id] = available[idx].ray_node_id
        idx += 1

    return _NodeMapping(
        training=training_map,
        rollout=rollout_map,
        spare=spare_map,
    )


def _rollout_num_cells_to_ids(num_cells: int) -> list[str]:
    if num_cells == 1:
        return ["default"]
    return [str(i) for i in range(num_cells)]


async def _poll_for_run_id(
    controller: ray.actor.ActorHandle,
    timeout: float = 15.0,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = ray.get(controller.get_status.remote(), timeout=5)
        if status.active_run_id is not None:
            return status.active_run_id
        await asyncio.sleep(0.3)
    raise TimeoutError("active_run_id not set within timeout")
