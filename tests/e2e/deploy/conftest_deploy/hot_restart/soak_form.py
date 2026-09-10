import asyncio
import random
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads
from tests.e2e.deploy.conftest_deploy.hot_restart.deployment_target import validate_deployment_target
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import REPLACED_LAUNCH_EXIT_CODE, compute_hot_restart_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import (
    HOT_RESTART_FORM_NAME,
    TAKE_OVER_POLL_INTERVAL_SECONDS,
    TAKE_OVER_TIMEOUT_SECONDS,
    restamped_replaced_workloads,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.guarded_launcher import HotRestartLaunchSpec
from tests.utils.soak.action import SoakActionForm
from tests.utils.soak.fault_forms import BaseFaultForm
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec, launch
from tests.utils.soak.state import (
    EventLog,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakDeploymentTarget,
    SoakLauncherExitedEvent,
    SoakObservation,
)

SESSION_TIMEOUT_SECONDS: float = 6 * 3600


class SoakActionFormHotRestart(BaseFaultForm, SoakActionForm):
    def __init__(
        self,
        *,
        launch_spec: Gsm8kLaunchSpec,
        event_log: EventLog,
        log_dir: Path,
        max_allowed_rollout_id: int,
        poll_interval_seconds: float = TAKE_OVER_POLL_INTERVAL_SECONDS,
        timeout_seconds: float = TAKE_OVER_TIMEOUT_SECONDS,
    ) -> None:
        self._launch_spec = launch_spec
        self._event_log = event_log
        self._log_dir = log_dir
        self._max_allowed_rollout_id = max_allowed_rollout_id
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def harms_cell(self) -> bool:
        return False

    def is_within_injection_window(self) -> bool:
        observation = next(
            (event for event in reversed(self._event_log.events) if isinstance(event, SoakObservation)), None
        )
        if observation is None or len(observation.deployments) != 1:
            return False
        progress = observation.deployments[0].finished_rollout_id
        return progress is None or progress < self._max_allowed_rollout_id

    async def execute(self, request: SoakActionRequest) -> None:
        target = request.target
        assert isinstance(target, SoakDeploymentTarget), "Hot restart requires a deployment target"
        assert request.form_name == self.name
        assert target.namespace == self._launch_spec.config.namespace
        config = compute_hot_restart_config(self._launch_spec.config, installed_release=target.release)
        await validate_deployment_target(target)
        spec = HotRestartLaunchSpec(
            config=config,
            train_args=self._launch_spec.train_args,
            fully_async=self._launch_spec.fully_async,
            target=target,
            guard_directory=self._log_dir / f"guard-{request.request_id}",
        )
        log_path = self._log_dir / f"launcher-{request.request_id}.log"
        launcher = asyncio.create_task(self._launch(request=request, spec=spec, log_path=log_path))
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._wait_for_take_over(request=request, launcher=launcher)
            result = await launcher
            assert result in (
                0,
                REPLACED_LAUNCH_EXIT_CODE,
            ), f"Hot restart launcher {request.request_id} exited {result}; see {log_path}"
        finally:
            if not launcher.done():
                launcher.cancel()
            await asyncio.gather(launcher, return_exceptions=True)

    def inject(self, cell: dict, rng: random.Random) -> None:
        raise AssertionError("Hot restart must run through the async soak runner")

    async def _launch(self, *, request: SoakActionRequest, spec: Gsm8kLaunchSpec, log_path: Path) -> int:
        result = await launch(
            spec,
            log_path=log_path,
            timeout_seconds=SESSION_TIMEOUT_SECONDS,
            module_name="tests.e2e.deploy.conftest_deploy.hot_restart.guarded_launcher",
        )
        self._event_log.note_launcher_exited(
            SoakLauncherExitedEvent(request_id=request.request_id, returncode=result, log_path=log_path)
        )
        return result

    async def _wait_for_take_over(self, *, request: SoakActionRequest, launcher: asyncio.Task[int]) -> None:
        target = request.target
        assert isinstance(target, SoakDeploymentTarget)
        while True:
            await asyncio.sleep(self._poll_interval_seconds)
            events = self._event_log.events
            observation = next((event for event in reversed(events) if isinstance(event, SoakObservation)), None)
            after = (
                next(
                    (
                        one
                        for one in observation.deployments
                        if one.release == target.release and one.namespace == target.namespace
                    ),
                    None,
                )
                if observation is not None
                else None
            )
            if after is not None and restamped_replaced_workloads(
                before=target.workload_stamps,
                after=after.workload_stamps,
                workloads=compute_hot_restart_workloads(target.release),
            ):
                requests = [
                    event.request.request_id
                    for event in events
                    if isinstance(event, SoakActionRequestedEvent)
                    and isinstance(event.request.target, SoakDeploymentTarget)
                ]
                record = HotRestartRecord(
                    index=requests.index(request.request_id),
                    saved_iteration_at_trigger=target.saved_iteration,
                    frozen_rollout_id=-1 if target.finished_rollout_id is None else target.finished_rollout_id,
                )
                self._event_log.note_action_applied(
                    SoakActionAppliedEvent(
                        request_id=request.request_id,
                        evidence={"record": record.model_dump(mode="json"), "after": after.model_dump(mode="json")},
                    )
                )
                return
            assert not launcher.done(), (
                f"Hot restart launcher {request.request_id} exited {launcher.result()} before restamping "
                f"{sorted(compute_hot_restart_workloads(target.release))}"
            )
