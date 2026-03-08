"""Remote-controlled training job and training worker for local_ray integration tests.

Provides a TrainingJobProtocol implementation whose state lives in a separate
Ray actor, so the test driver can mutate it (crash, hang, recover) while the
FtController actor reads it during its tick loop.

TrainingWorkerActor simulates a training process hosting a real
FtTrainingRankAgent — it self-registers with the controller, advances
iteration metrics, and reacts to crash/recovery state transitions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from unittest.mock import patch
from uuid import uuid4

import ray

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent
from miles.utils.ft.models.metrics import MetricSample
from miles.utils.ft.platform.ray_wrappers.controller_client import RayControllerClient
from miles.utils.ft.protocols.platform import JobStatus

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0, num_gpus=0)
class TrainingStateActor:
    """Holds mutable state shared between the test driver and RemoteControlledTrainingJob.

    Methods are synchronous (not async) since state updates are trivial.
    """

    def __init__(self) -> None:
        self._status: str = JobStatus.RUNNING.value
        self._run_id: str = uuid4().hex[:8]
        self._submit_count: int = 0
        self._stop_called: bool = False
        self._excluded_node_ids: list[str] = []
        self._hung: bool = False
        self._custom_log_metrics: dict[str, float] = {}

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def get_run_id(self) -> str:
        return self._run_id

    def submit(self, excluded_node_ids: list[str] | None = None) -> str:
        self._submit_count += 1
        self._run_id = uuid4().hex[:8]
        self._status = JobStatus.RUNNING.value
        self._stop_called = False
        self._hung = False
        self._custom_log_metrics = {}
        self._excluded_node_ids = excluded_node_ids or []
        return self._run_id

    def stop(self) -> None:
        self._stop_called = True
        self._status = JobStatus.STOPPED.value

    def get_submit_count(self) -> int:
        return self._submit_count

    def get_stop_called(self) -> bool:
        return self._stop_called

    def get_excluded_node_ids(self) -> list[str]:
        return self._excluded_node_ids

    def set_hung(self, hung: bool) -> None:
        self._hung = hung

    def get_hung(self) -> bool:
        return self._hung

    def set_custom_log_metrics(self, metrics: dict[str, float]) -> None:
        self._custom_log_metrics = metrics

    def get_custom_log_metrics(self) -> dict[str, float]:
        return self._custom_log_metrics


class RemoteControlledTrainingJob:
    """TrainingJobProtocol that delegates to a TrainingStateActor.

    Instances are serialized into the FtControllerActor via cloudpickle.
    All state queries go through Ray RPCs to the shared TrainingStateActor,
    so the test driver can control the training job externally.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    async def get_training_status(self) -> JobStatus:
        status_str: str = await self._state.get_status.remote()
        return JobStatus(status_str)

    async def stop_training(self, timeout_seconds: int = 300) -> None:
        await self._state.stop.remote()

    async def submit_training(
        self,
        excluded_node_ids: list[str] | None = None,
    ) -> str:
        run_id: str = await self._state.submit.remote(excluded_node_ids)
        return run_id


@ray.remote(num_cpus=0, num_gpus=0)
class TrainingWorkerActor:
    """Simulates a training process hosting a real FtTrainingRankAgent.

    Watches TrainingStateActor for status changes:
    - RUNNING: creates agent (if needed) and advances iterations
    - FAILED/STOPPED: shuts down agent and pauses

    After recovery (new submit), detects the new run_id and re-creates
    the agent so it re-registers with the controller under the new run.
    """

    def __init__(
        self,
        state_actor: ray.actor.ActorHandle,
        ft_id: str = "",
        rank: int = 0,
        world_size: int = 1,
        node_id: str = "sim-node-0",
        step_interval: float = 0.1,
    ) -> None:
        self._state_actor = state_actor
        self._ft_id = ft_id
        self._rank = rank
        self._world_size = world_size
        self._node_id = node_id
        self._step_interval = step_interval

        self._agent: FtTrainingRankAgent | None = None
        self._controller_client: RayControllerClient | None = None
        self._current_run_id: str = ""
        self._iteration: int = 0
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        run_id: str = ray.get(self._state_actor.get_run_id.remote())
        self._create_agent(run_id)
        self._loop_task = asyncio.get_event_loop().create_task(self._run_loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self._shutdown_agent()

    def get_iteration(self) -> int:
        return self._iteration

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_agent(self, run_id: str) -> None:
        self._shutdown_agent()
        self._current_run_id = run_id

        os.environ["MILES_FT_ID"] = self._ft_id
        os.environ["MILES_FT_TRAINING_RUN_ID"] = run_id

        self._controller_client = RayControllerClient(ft_id=self._ft_id)

        with patch("socket.gethostname", return_value=self._node_id):
            self._agent = FtTrainingRankAgent(
                rank=self._rank,
                world_size=self._world_size,
                controller_client=self._controller_client,
            )

        self._iteration = 0

    def _shutdown_agent(self) -> None:
        if self._agent is not None:
            self._agent.shutdown()
            self._agent = None

    async def _run_loop(self) -> None:
        while True:
            status_str: str = await self._state_actor.get_status.remote()
            status = JobStatus(status_str)

            if status == JobStatus.RUNNING:
                hung: bool = await self._state_actor.get_hung.remote()
                if hung:
                    await asyncio.sleep(self._step_interval)
                    continue

                run_id: str = await self._state_actor.get_run_id.remote()
                if self._agent is None or run_id != self._current_run_id:
                    self._create_agent(run_id)

                self._iteration += 1
                self._agent.step()

                if self._controller_client is not None:
                    try:
                        custom: dict[str, float] = await self._state_actor.get_custom_log_metrics.remote()
                        metrics: dict[str, float] = {"iteration": float(self._iteration)}
                        metrics.update(custom)
                        self._controller_client.log_step(
                            run_id=self._current_run_id,
                            step=self._iteration,
                            metrics=metrics,
                        )
                    except Exception:
                        logger.debug("log_step failed", exc_info=True)

            elif status in (JobStatus.FAILED, JobStatus.STOPPED):
                if self._agent is not None:
                    self._shutdown_agent()

            await asyncio.sleep(self._step_interval)


@ray.remote(num_cpus=0, num_gpus=0)
class CollectorStateActor:
    """Shared metric state for RemoteControlledCollector.

    Tests call set_metrics() to inject hardware metrics (GPU unavailable,
    XID errors, NIC down, etc.) that node agents will scrape.
    """

    def __init__(self) -> None:
        self._metrics: list[MetricSample] = []

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def get_metrics(self) -> list[MetricSample]:
        return self._metrics


class RemoteControlledCollector(BaseCollector):
    """Collector that reads metrics from a CollectorStateActor.

    Allows tests to dynamically inject/update node-level hardware
    metrics at runtime without real hardware.
    """

    def __init__(
        self,
        state_actor: ray.actor.ActorHandle,
        collect_interval: float = 1.0,
    ) -> None:
        self.collect_interval = collect_interval
        self._state = state_actor

    def _collect_sync(self) -> list[MetricSample]:
        return ray.get(self._state.get_metrics.remote())


@ray.remote(num_cpus=0, num_gpus=0)
class NotifierStateActor:
    """Stores notification records across Ray actor boundaries.

    The controller actor runs in a separate process, so a plain FakeNotifier's
    calls list is inaccessible from the test driver. This actor holds the
    records so both sides can access them via Ray RPCs.
    """

    def __init__(self) -> None:
        self._calls: list[tuple[str, str, str]] = []

    def record(self, title: str, content: str, severity: str) -> None:
        self._calls.append((title, content, severity))

    def get_calls(self) -> list[tuple[str, str, str]]:
        return list(self._calls)

    def clear(self) -> None:
        self._calls.clear()


class RemoteControlledNotifier:
    """NotificationProtocol that delegates to a NotifierStateActor.

    Serialized into FtControllerActor via cloudpickle. All send() calls
    are forwarded to the shared actor so the test driver can read them back.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    async def send(self, title: str, content: str, severity: str) -> None:
        await self._state.record.remote(title, content, severity)

    async def aclose(self) -> None:
        pass
