import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass

from tests.utils.deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import LaunchOutcome, SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import BaseSoakActionForm, SoakActionEvidence, SoakActionRequest
from tests.utils.soak.core.utils import note_launch_outcome
from tests.utils.soak.core.views import SoakActionRecord, latest_observation
from tests.utils.soak.deploy.guard.launch_guard import HotRestartLaunchGuard, HotRestartLaunchSpec
from tests.utils.soak.deploy.guard.target_check import assert_workloads_unchanged
from tests.utils.soak.deploy.session import LauncherChain
from tests.utils.soak.deploy.types import (
    DEPLOYMENT_TARGET_KIND,
    DeploymentTarget,
    HotRestartDetails,
    HotRestartTakeOverEvidence,
)
from tests.utils.soak.deploy.utils import compute_hot_restart_config
from tests.utils.soak.recipes.gsm8k import Gsm8kLaunchSpec, launch

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0


def saved_iteration_after(action: SoakActionRecord) -> int:
    assert action.applied is not None, f"Action {action.requested.request.request_id} never took effect"
    before = action.requested.request.target
    after = _take_over_evidence(action).after
    return max(
        before.saved_iteration if before.saved_iteration is not None else -1,
        after.saved_iteration if after.saved_iteration is not None else -1,
    )


@dataclass(frozen=True, kw_only=True)
class HotRestartForm(BaseSoakActionForm):
    launch_spec: Gsm8kLaunchSpec
    event_log: EventLog
    chain: LauncherChain

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def harms_target(self) -> bool:
        return False

    def maybe_create_request(
        self,
        *,
        target: DeploymentTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        if not target.ready or target.finished_rollout_id is None or target.saved_iteration is None:
            return None
        return SoakActionRequest(target=target, form_name=self.name, details=HotRestartDetails())

    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool:
        if action.applied is None:
            return False
        before = action.requested.request.target
        after = _take_over_evidence(action).after
        saved = saved_iteration_after(action)
        finished = before.finished_rollout_id if before.finished_rollout_id is not None else -1
        return any(
            observation.timestamp > action.applied.timestamp
            and not observation.errors
            and target.ready
            and target.namespace == before.namespace
            and target.release == before.release
            and target.workload_uids == after.workload_uids
            and target.workload_stamps == after.workload_stamps
            and (saved_iteration := target.saved_iteration) is not None
            and saved_iteration > saved
            and (finished_rollout_id := target.finished_rollout_id) is not None
            and finished_rollout_id > finished
            for observation in events
            if isinstance(observation, SoakObservationEvent)
            for target in read_deployment_targets(observation)
        )

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        target = request.target
        assert request.form_name == self.name
        assert target.namespace == self.launch_spec.config.namespace
        config = compute_hot_restart_config(self.launch_spec.config, installed_release=target.release)
        await asyncio.to_thread(assert_workloads_unchanged, target)
        spec = HotRestartLaunchSpec(
            config=config,
            train_args=self.launch_spec.train_args,
            fully_async=self.launch_spec.fully_async,
            target=target,
        )
        launcher = asyncio.create_task(self._launch(request=request, spec=spec))
        self.chain.put_nowait(launcher)
        try:
            async with asyncio.timeout(TAKE_OVER_TIMEOUT_SECONDS):
                await self._wait_for_take_over(request=request, launcher=launcher, report_applied=report_applied)
            await launcher
        finally:
            if not launcher.done():
                launcher.cancel()
            await asyncio.gather(launcher, return_exceptions=True)

    async def _launch(self, *, request: SoakActionRequest, spec: HotRestartLaunchSpec) -> LaunchOutcome:
        return await note_launch_outcome(
            event_log=self.event_log,
            request_id=request.request_id,
            launching=launch(spec, guard=HotRestartLaunchGuard(target=spec.target)),
        )

    async def _wait_for_take_over(
        self,
        *,
        request: SoakActionRequest,
        launcher: asyncio.Task[LaunchOutcome],
        report_applied: Callable[[SoakActionEvidence], None],
    ) -> None:
        target = request.target
        while True:
            await asyncio.sleep(TAKE_OVER_POLL_INTERVAL_SECONDS)
            after = _latest_target(self.event_log.events, target=target)
            if after is not None and all(
                (stamp := after.workload_stamps.get(one)) is not None and stamp != target.workload_stamps.get(one)
                for one in compute_hot_restart_workloads(target.release)
            ):
                report_applied(HotRestartTakeOverEvidence(after=after))
                return
            assert not launcher.done(), (
                f"Hot restart launcher {request.request_id} returned {launcher.result()} before restamping "
                f"{sorted(compute_hot_restart_workloads(target.release))}"
            )


def read_deployment_targets(observation: SoakObservationEvent) -> list[DeploymentTarget]:
    return [target for target in observation.targets or [] if target.kind == DEPLOYMENT_TARGET_KIND]


def _take_over_evidence(action: SoakActionRecord) -> HotRestartTakeOverEvidence:
    assert action.applied is not None and isinstance(action.applied.evidence, HotRestartTakeOverEvidence)
    return action.applied.evidence


def _latest_target(events: list[SoakEvent], *, target: DeploymentTarget) -> DeploymentTarget | None:
    if (observation := latest_observation(events)) is None:
        return None
    return next(
        (
            one
            for one in read_deployment_targets(observation)
            if one.ready and one.release == target.release and one.namespace == target.namespace
        ),
        None,
    )
