import asyncio
import random
from pathlib import Path

import pytest
from tests.fast.e2e.deploy.hot_restart.cluster_facts import ORCHESTRATOR, ROLLOUT_EXECUTOR
from tests.fast.utils.soak.deploy.deploy_fakes import (
    _deployment_observation,
    _deployment_target,
    _FakeLauncher,
    _hot_restart_request,
    _HotRestartHarness,
    _landed_take_over,
    _requested_take_over,
    _restamped,
)
from tests.fast.utils.soak.soak_fakes import _at, _wait_until
from tests.utils.soak.core.events import LaunchOutcome, SoakEvent
from tests.utils.soak.core.types import SoakActionRequest
from tests.utils.soak.core.utils import REPLACED_LAUNCH_EXIT_CODE
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.deploy.actions import hot_restart as hot_restart_module
from tests.utils.soak.deploy.actions.hot_restart import saved_iteration_after
from tests.utils.soak.deploy.guard.launch_guard import HotRestartLaunchGuard, HotRestartLaunchSpec
from tests.utils.soak.deploy.types import DeploymentTarget, HotRestartDetails, HotRestartTakeOverEvidence
from tests.utils.soak.deploy.utils import HOT_RESTART_ARG

from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import RunExitedError
from miles.utils.workers.cell_operations.base import StaleFaultTargetError


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.fixture
def launcher() -> _FakeLauncher:
    return _FakeLauncher(error=RunExitedError(REPLACED_LAUNCH_EXIT_CODE))


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher: _FakeLauncher) -> _HotRestartHarness:
    return _HotRestartHarness(tmp_path, monkeypatch, launcher=launcher)


class TestHotRestartFormRequest:
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param(_deployment_target(ready=False), id="not_ready"),
            pytest.param(_deployment_target(saved_iteration=None), id="no_checkpoint"),
            pytest.param(_deployment_target(finished_rollout_id=None), id="no_finished_rollout"),
        ],
    )
    def test_a_target_failing_a_gate_is_declined(self, harness: _HotRestartHarness, target: DeploymentTarget) -> None:
        """A take-over needs a ready release that has saved and finished at least one rollout."""
        form = harness.form()

        request = form.maybe_create_request(
            target=target, observation=_deployment_observation([target], at=_at(0)), events=[], rng=random.Random(0)
        )

        assert request is None

    def test_a_ready_checkpointed_target_is_requested_as_observed(self, harness: _HotRestartHarness) -> None:
        """The request pins the exact observed generation the guard later compares against."""
        target = _deployment_target(saved_iteration=0, finished_rollout_id=0)

        request = harness.form().maybe_create_request(
            target=target, observation=_deployment_observation([target], at=_at(0)), events=[], rng=random.Random(0)
        )

        assert (request.form_name, request.target, request.details) == ("hot_restart", target, HotRestartDetails())

    def test_a_take_over_leaves_the_target_running(self, harness: _HotRestartHarness) -> None:
        """A hot restart is not a harm, so the scheduler must not treat the release as down."""
        assert not harness.form().harms_target


class TestHotRestartFormExecuteRefusals:
    async def test_a_target_in_another_namespace_launches_nothing(self, harness: _HotRestartHarness) -> None:
        """The form only takes over the namespace its launch spec installs into."""
        with pytest.raises(AssertionError):
            await harness.start(_hot_restart_request(_deployment_target(namespace="other")))

        assert harness.launcher.calls == [] and harness.chain.empty()

    async def test_a_spec_for_another_release_launches_nothing(self, harness: _HotRestartHarness) -> None:
        """A relaunch under another run id would install beside the watched run."""
        with pytest.raises(AssertionError, match="already up"):
            await harness.start(_hot_restart_request(_deployment_target()), run_id="other")

        assert harness.launcher.calls == [] and harness.chain.empty()

    async def test_stale_workloads_launch_nothing(self, harness: _HotRestartHarness) -> None:
        """A generation changed since observation must be refused before any launcher starts."""
        harness.stale = StaleFaultTargetError("changed")
        target = _deployment_target()

        with pytest.raises(StaleFaultTargetError):
            await harness.start(_hot_restart_request(target))

        assert harness.checked == [target]
        assert harness.launcher.calls == [] and harness.chain.empty()


class TestHotRestartFormExecuteTakeOver:
    async def test_the_launch_carries_the_hot_restart_config_and_a_guard_on_the_target(
        self, harness: _HotRestartHarness
    ) -> None:
        """The relaunch reuses this run's args, flags hot restart, and guards writes against the observed target."""
        target = _deployment_target()
        task = harness.start(_hot_restart_request(target))
        await _wait_until(lambda: harness.launcher.calls)

        [(spec, guard)] = harness.launcher.calls
        assert isinstance(spec, HotRestartLaunchSpec)
        assert spec.config.hot_restart == HOT_RESTART_ARG
        assert (spec.train_args, spec.fully_async, spec.target) == ("--save /ckpt ", True, target)
        assert isinstance(guard, HotRestartLaunchGuard) and guard.target == target
        assert harness.chain.qsize() == 1

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_it_is_applied_only_once_every_hot_restart_workload_is_restamped(
        self, harness: _HotRestartHarness
    ) -> None:
        """Half a rolled upgrade, an unready release or another release is not a landed take-over."""
        target = _deployment_target()
        task = harness.start(_hot_restart_request(target))
        await _wait_until(lambda: harness.launcher.calls)

        harness.observe(_restamped(target, "t1", names=(ORCHESTRATOR,)))
        await _settle()
        harness.observe(_restamped(target, "t1", ready=False))
        await _settle()
        harness.observe(_restamped(target, "t1", release="miles-run-other-all"))
        await _settle()
        assert harness.reported == []

        after = _restamped(target, "t1")
        harness.observe(after)
        await _wait_until(lambda: harness.reported)
        assert harness.reported == [HotRestartTakeOverEvidence(after=after)]

        harness.launcher.finish.set()
        await task
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.REPLACED)]

    async def test_a_stamp_equal_to_the_observed_one_is_not_a_restamp(self, harness: _HotRestartHarness) -> None:
        """A second take-over must rewrite the stamps the first one left, not merely carry some stamp."""
        target = _deployment_target(stamps={ORCHESTRATOR: "t1", ROLLOUT_EXECUTOR: "t1"})
        task = harness.start(_hot_restart_request(target))
        await _wait_until(lambda: harness.launcher.calls)

        harness.observe(_restamped(target, "t2", names=(ROLLOUT_EXECUTOR,)))
        await _settle()
        assert harness.reported == []

        harness.observe(_restamped(target, "t2"))
        await _wait_until(lambda: harness.reported)
        harness.launcher.finish.set()
        await task

    async def test_a_launcher_that_finishes_before_the_restamp_fails_without_applied(
        self, harness: _HotRestartHarness, launcher: _FakeLauncher
    ) -> None:
        """A relaunch that returned without rolling the workloads never took the run over."""
        launcher.error = None
        launcher.finish.set()

        with pytest.raises(AssertionError, match="before restamping"):
            await harness.start(_hot_restart_request(_deployment_target()))

        assert harness.reported == []
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.FINISHED)]

    async def test_a_launcher_that_fails_before_the_restamp_is_raised_without_applied(
        self, harness: _HotRestartHarness, launcher: _FakeLauncher
    ) -> None:
        """A refused upgrade leaves the old script running, which must surface as the failure it is."""
        launcher.error = RuntimeError("helm upgrade failed")
        launcher.finish.set()

        with pytest.raises(RuntimeError, match="helm upgrade failed"):
            await harness.start(_hot_restart_request(_deployment_target()))

        assert harness.reported == []
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.FAILED)]

    async def test_a_take_over_that_never_lands_times_out_and_cancels_its_launcher(
        self, harness: _HotRestartHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a restamp there is no evidence the relaunch installed, and the launcher must not leak."""
        monkeypatch.setattr(hot_restart_module, "TAKE_OVER_TIMEOUT_SECONDS", 0.2)

        with pytest.raises(TimeoutError):
            await harness.start(_hot_restart_request(_deployment_target()))

        assert harness.reported == []
        assert harness.launcher.cancelled
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.FAILED)]

    async def test_cancelling_the_action_cancels_its_launcher(self, harness: _HotRestartHarness) -> None:
        """A soak shutting down mid take-over must not leave a relaunch running."""
        task = harness.start(_hot_restart_request(_deployment_target()))
        await _wait_until(lambda: harness.launcher.calls)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert harness.launcher.cancelled
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.FAILED)]

    async def test_a_launcher_failing_after_the_landing_is_raised_after_applied(
        self, harness: _HotRestartHarness, launcher: _FakeLauncher
    ) -> None:
        """The last relaunch carries the run's verdict, so its failure must not be lost after applied."""
        launcher.error = RuntimeError("metric checker failed")
        target = _deployment_target()
        task = harness.start(_hot_restart_request(target))
        await _wait_until(lambda: harness.launcher.calls)

        harness.observe(_restamped(target, "t1"))
        await _wait_until(lambda: harness.reported)
        launcher.finish.set()

        with pytest.raises(RuntimeError, match="metric checker failed"):
            await task
        assert len(harness.reported) == 1
        assert harness.launch_outcomes() == [("req-1", LaunchOutcome.FAILED)]

    def test_a_failed_exit_code_after_the_landing_reaches_the_caller_without_stopping_the_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-SIGTERM launcher exit must surface to the awaiting runner rather than escape the event loop."""
        launcher = _FakeLauncher(error=RunExitedError(1))
        harness = _HotRestartHarness(tmp_path, monkeypatch, launcher=launcher)
        target = _deployment_target()

        async def _scenario() -> BaseException | None:
            task = harness.start(_hot_restart_request(target))
            await _wait_until(lambda: launcher.calls)
            harness.observe(_restamped(target, "t1"))
            await _wait_until(lambda: harness.reported)
            launcher.finish.set()
            try:
                await task
            except BaseException as error:
                return error
            return None

        caught = asyncio.run(_scenario())

        assert isinstance(caught, RuntimeError)
        assert isinstance(caught.__cause__, RunExitedError) and caught.__cause__.exit_code == 1


def _recovery_events(
    *,
    before: DeploymentTarget,
    after: DeploymentTarget,
    later: DeploymentTarget,
    later_at: float = 3,
    errors: dict[str, str] | None = None,
    applied: bool = True,
) -> tuple[SoakActionRequest, list[SoakEvent]]:
    request = _hot_restart_request(before)
    events: list[SoakEvent] = [_requested_take_over(request, at=_at(0))]
    if applied:
        events.append(_landed_take_over(request, after=after, at=_at(2)))
    events.append(_deployment_observation([later], at=_at(later_at), errors=errors))
    return request, events


def _is_recovered(harness: _HotRestartHarness, events: list[SoakEvent]) -> bool:
    [action] = project_actions(events).values()
    return harness.form().is_recovered(action=action, events=events)


class TestHotRestartFormIsRecovered:
    _BEFORE = _deployment_target(saved_iteration=3, finished_rollout_id=5)
    _AFTER = _restamped(_BEFORE, "t1", saved_iteration=4)

    def test_the_new_generation_saving_and_finishing_more_is_recovered(self, harness: _HotRestartHarness) -> None:
        """Recovery needs the landed generation to checkpoint past both sides and finish a later rollout."""
        later = self._AFTER.model_copy(update={"saved_iteration": 5, "finished_rollout_id": 6})
        _, events = _recovery_events(before=self._BEFORE, after=self._AFTER, later=later)

        assert _is_recovered(harness, events)

    @pytest.mark.parametrize(
        ("update", "later_at", "errors"),
        [
            pytest.param({"saved_iteration": 4}, 3, None, id="saved_not_past_after"),
            pytest.param({"finished_rollout_id": 5}, 3, None, id="rollout_not_advanced"),
            pytest.param({"saved_iteration": None}, 3, None, id="no_saved"),
            pytest.param({"ready": False}, 3, None, id="not_ready"),
            pytest.param(
                {"workload_stamps": _restamped(_BEFORE, "t2").workload_stamps}, 3, None, id="another_take_over"
            ),
            pytest.param({"workload_uids": {ORCHESTRATOR: "uid-new"}}, 3, None, id="recreated"),
            pytest.param({"namespace": "other"}, 3, None, id="other_namespace"),
            pytest.param({}, 2, None, id="at_applied_time"),
            pytest.param({}, 3, {"progress": "boom"}, id="observation_errors"),
        ],
    )
    def test_a_later_observation_that_proves_less_is_not_recovered(
        self, harness: _HotRestartHarness, update: dict, later_at: float, errors: dict[str, str] | None
    ) -> None:
        """Any missing progress, mismatch with the landed generation, or unreliable read is no recovery."""
        progressed = self._AFTER.model_copy(update={"saved_iteration": 5, "finished_rollout_id": 6})
        later = progressed.model_copy(update=update)
        _, events = _recovery_events(
            before=self._BEFORE, after=self._AFTER, later=later, later_at=later_at, errors=errors
        )

        assert not _is_recovered(harness, events)

    def test_an_action_that_never_landed_is_not_recovered(self, harness: _HotRestartHarness) -> None:
        """Progress without a landed take-over is ordinary training, not recovery from one."""
        later = self._AFTER.model_copy(update={"saved_iteration": 9, "finished_rollout_id": 9})
        _, events = _recovery_events(before=self._BEFORE, after=self._AFTER, later=later, applied=False)

        assert not _is_recovered(harness, events)


class TestSavedIterationAfter:
    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            pytest.param(3, 4, 4, id="after_newer"),
            pytest.param(5, 4, 5, id="before_newer"),
            pytest.param(None, None, -1, id="never_saved"),
            pytest.param(None, 2, 2, id="first_save_after"),
        ],
    )
    def test_it_is_the_newest_checkpoint_either_side_of_the_landing(
        self, before: int | None, after: int | None, expected: int
    ) -> None:
        """The next take-over must save past whichever checkpoint was already there when this one landed."""
        before_target = _deployment_target(saved_iteration=before)
        after_target = _restamped(before_target, "t1", saved_iteration=after)
        _, events = _recovery_events(before=before_target, after=after_target, later=after_target)
        [action] = project_actions(events).values()

        assert saved_iteration_after(action) == expected

    def test_an_unapplied_action_is_refused(self) -> None:
        """Only a landed take-over has an after side to read."""
        target = _deployment_target()
        _, events = _recovery_events(before=target, after=target, later=target, applied=False)
        [action] = project_actions(events).values()

        with pytest.raises(AssertionError, match="never took effect"):
            saved_iteration_after(action)
