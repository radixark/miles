import asyncio
import contextlib
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import (
    _cell_target,
    _FakeForm,
    _flatten_errors,
    _healthy_observer,
    _make_runner,
    _now,
    _observation,
    _pod_evidence,
    _RecordingTeardown,
    _request,
    _runner_config,
    _ScriptedObserver,
    _ScriptedScheduler,
    _ScriptedSutFeed,
    _step_end,
    _wait_until,
)
from tests.utils.soak.core.config import SoakTailConfig, SoakTimeoutConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakActionAppliedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakCollectionClosedEvent,
    SoakEvent,
    SoakObservationEvent,
    read_events,
)
from tests.utils.soak.core.runner import SoakRunner, _assert_within_tail_budget
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest


async def _hang() -> None:
    await asyncio.Event().wait()


class TestSoakRunnerLifecycle:
    async def test_a_finished_run_closes_admission_observes_once_more_tears_down_then_archives(
        self, tmp_path: Path
    ) -> None:
        """After training ends the runner closes admission, takes a final observation, tears down and closes."""
        observer = _healthy_observer()
        runner = _make_runner(tmp_path, observer=observer)
        teardown = _RecordingTeardown(runner.event_log)

        await runner.run(_wait_until(lambda: observer.calls >= 3), teardown=teardown)

        kinds = [event.kind for event in runner.event_log.events]
        assert kinds[-3:] == ["admission_closed", "observation", "collection_closed"]
        assert kinds[:-3] == ["observation"] * len(kinds[:-3]) and len(kinds) >= 6
        assert teardown.closed_when_called == [False]
        assert read_events(runner.event_log.path) == runner.event_log.events

    async def test_a_training_failure_propagates_after_teardown_and_archive(self, tmp_path: Path) -> None:
        """A crashed run is never reported as success yet still releases resources and closes evidence."""
        observer = _healthy_observer()
        runner = _make_runner(tmp_path, observer=observer)
        teardown = _RecordingTeardown(runner.event_log)

        async def crash() -> None:
            await _wait_until(lambda: observer.calls >= 1)
            raise RuntimeError("training crashed")

        with pytest.raises(BaseException) as excinfo:
            await runner.run(crash(), teardown=teardown)

        assert [type(error) for error in _flatten_errors(excinfo.value)] == [RuntimeError]
        assert teardown.calls == 1
        assert isinstance(runner.event_log.events[-1], SoakCollectionClosedEvent)

    async def test_a_run_exceeding_its_deadline_times_out_and_still_closes(self, tmp_path: Path) -> None:
        """A hung training is bounded by the run timeout and its evidence is still closed."""
        config = _runner_config(timeouts=SoakTimeoutConfig(run_seconds=0.05))
        runner = _make_runner(tmp_path, observer=_healthy_observer(), config=config)
        teardown = _RecordingTeardown(runner.event_log)

        with pytest.raises(BaseException) as excinfo:
            await runner.run(_hang(), teardown=teardown)

        assert [type(error) for error in _flatten_errors(excinfo.value)] == [TimeoutError]
        assert teardown.calls == 1
        assert isinstance(runner.event_log.events[-1], SoakCollectionClosedEvent)

    async def test_a_cancelled_run_still_tears_down_and_closes(self, tmp_path: Path) -> None:
        """Cancelling the soak cannot leak the run or leave evidence open."""
        observer = _healthy_observer()
        runner = _make_runner(tmp_path, observer=observer)
        teardown = _RecordingTeardown(runner.event_log)

        task = asyncio.create_task(runner.run(_hang(), teardown=teardown))
        await _wait_until(lambda: observer.calls >= 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert teardown.calls == 1
        assert isinstance(runner.event_log.events[-1], SoakCollectionClosedEvent)

    async def test_a_hung_observer_is_recorded_as_a_failed_observation(self, tmp_path: Path) -> None:
        """An observation exceeding its bound becomes a target-less observation carrying the timeout."""
        config = _runner_config(timeouts=SoakTimeoutConfig(observation_seconds=0.01, final_observation_seconds=0.01))
        observer = _ScriptedObserver(lambda index: _observation([_cell_target()], at=_now()), hang_from_call=0)
        runner = _make_runner(tmp_path, observer=observer, config=config)

        await runner.run(_wait_until(lambda: observer.calls >= 2), teardown=_RecordingTeardown(runner.event_log))

        observations = [event for event in runner.event_log.events if isinstance(event, SoakObservationEvent)]
        assert len(observations) >= 2
        assert all(one.targets is None and "observation" in one.errors for one in observations)

    async def test_an_action_still_running_at_the_end_fails_the_soak(self, tmp_path: Path) -> None:
        """A cancelled in-flight action is recorded as not returned and the soak is refused."""
        form = _FakeForm(execute=lambda request, report_applied: _hang())
        scheduler = _ScriptedScheduler([_request(_cell_target())])
        runner = _make_runner(tmp_path, observer=_healthy_observer(), forms={"actor": [form]}, scheduler=scheduler)

        with pytest.raises(AssertionError, match="did not finish"):
            await runner.run(_wait_until(lambda: bool(form.executed)), teardown=_RecordingTeardown(runner.event_log))

        [result] = [event for event in runner.event_log.events if isinstance(event, SoakActionResultEvent)]
        assert result.returned is False
        assert "CancelledError" in result.error

    async def test_an_unrecovered_action_fails_the_soak(self, tmp_path: Path) -> None:
        """A finished action whose target never recovered cannot end in a passing soak."""
        form = _FakeForm(recovered=False)
        scheduler = _ScriptedScheduler([_request(_cell_target())])
        runner = _make_runner(tmp_path, observer=_healthy_observer(), forms={"actor": [form]}, scheduler=scheduler)

        with pytest.raises(AssertionError, match="did not recover"):
            await runner.run(
                _wait_until(lambda: any(isinstance(e, SoakActionResultEvent) for e in runner.event_log.events)),
                teardown=_RecordingTeardown(runner.event_log),
            )

    async def test_a_failed_final_check_still_tears_down_and_closes_the_evidence(self, tmp_path: Path) -> None:
        """Rejecting an unrecovered action must not leak the run or leave its evidence unarchived."""
        form = _FakeForm(recovered=False)
        scheduler = _ScriptedScheduler([_request(_cell_target())])
        runner = _make_runner(tmp_path, observer=_healthy_observer(), forms={"actor": [form]}, scheduler=scheduler)
        teardown = _RecordingTeardown(runner.event_log)

        with pytest.raises(AssertionError, match="did not recover"):
            await runner.run(
                _wait_until(lambda: any(isinstance(e, SoakActionResultEvent) for e in runner.event_log.events)),
                teardown=teardown,
            )

        assert teardown.calls == 1
        assert isinstance(runner.event_log.events[-1], SoakCollectionClosedEvent)

    def test_a_recovery_tail_without_live_training_events_is_refused(self, tmp_path: Path) -> None:
        """Closing admission depends on training progress, so a tail needs the training event feed."""
        with pytest.raises(ValueError, match="live training events"):
            SoakRunner(
                observer=_healthy_observer(),
                scheduler=_ScriptedScheduler(),
                forms={},
                event_log=EventLog(tmp_path / "events.jsonl"),
                config=_runner_config(tail=SoakTailConfig(close_after_rollout_id=1)),
            )


class TestSoakRunnerTail:
    async def test_admission_closes_once_at_the_threshold_and_observation_continues(self, tmp_path: Path) -> None:
        """The first step at the closing rollout closes admission; later steps are still recorded."""
        feed = _ScriptedSutFeed([[_step_end(rollout_id, at=_now())] for rollout_id in range(4)])
        observer = _healthy_observer()
        runner = _make_runner(
            tmp_path,
            observer=observer,
            config=_runner_config(tail=SoakTailConfig(close_after_rollout_id=1)),
            sut_events=feed,
        )

        await runner.run(_wait_until(lambda: observer.calls >= 4), teardown=_RecordingTeardown(runner.event_log))

        events = runner.event_log.events
        closures = [index for index, event in enumerate(events) if isinstance(event, SoakAdmissionClosedEvent)]
        assert len(closures) == 1
        before = events[closures[0] - 1]
        assert isinstance(before, SoakObservationEvent)
        assert [step.rollout_id for step in before.new_sut_events] == [1]
        later_rollouts = [
            step.rollout_id
            for event in events[closures[0] :]
            if isinstance(event, SoakObservationEvent)
            for step in event.new_sut_events
        ]
        assert later_rollouts == [2, 3]

    async def test_steps_below_the_threshold_leave_admission_open_until_training_ends(self, tmp_path: Path) -> None:
        """Admission only closes early for a step at or past the closing rollout."""
        feed = _ScriptedSutFeed([[_step_end(rollout_id, at=_now())] for rollout_id in range(3)])
        observer = _healthy_observer()
        runner = _make_runner(
            tmp_path,
            observer=observer,
            config=_runner_config(tail=SoakTailConfig(close_after_rollout_id=5)),
            sut_events=feed,
        )

        await runner.run(_wait_until(lambda: observer.calls >= 3), teardown=_RecordingTeardown(runner.event_log))

        kinds = [event.kind for event in runner.event_log.events]
        assert kinds.index("admission_closed") == len(kinds) - 3

    async def test_a_tail_exceeding_its_budget_fails_the_soak(self, tmp_path: Path) -> None:
        """Recovery that does not finish within the tail budget times out instead of running on."""
        feed = _ScriptedSutFeed([[_step_end(0, at=_now())]])
        runner = _make_runner(
            tmp_path,
            observer=_healthy_observer(),
            config=_runner_config(
                tail=SoakTailConfig(close_after_rollout_id=0), timeouts=SoakTimeoutConfig(tail_seconds=0.001)
            ),
            sut_events=feed,
        )
        teardown = _RecordingTeardown(runner.event_log)

        with pytest.raises(BaseException) as excinfo:
            await runner.run(_hang(), teardown=teardown)

        [error] = _flatten_errors(excinfo.value)
        assert isinstance(error, TimeoutError)
        assert "tail exceeded" in str(error)
        assert teardown.calls == 1


class TestAssertWithinTailBudget:
    def test_an_open_admission_has_no_tail_budget(self) -> None:
        """Before closure the run is not in its tail, however long it has been running."""
        _assert_within_tail_budget([_observation(None, at=_now() - timedelta(days=1))], tail_seconds=1.0)

    def test_a_tail_older_than_its_budget_raises(self) -> None:
        """A closure older than the budget is a timed-out tail."""
        closed = SoakAdmissionClosedEvent(timestamp=_now() - timedelta(seconds=10))

        with pytest.raises(TimeoutError, match="tail exceeded"):
            _assert_within_tail_budget([closed], tail_seconds=5.0)

    def test_a_tail_within_its_budget_passes(self) -> None:
        """A recent closure is still inside the tail budget."""
        _assert_within_tail_budget([SoakAdmissionClosedEvent(timestamp=_now())], tail_seconds=60.0)


class TestSoakRunnerActions:
    async def _run_one(self, tmp_path: Path, form: _FakeForm) -> list[SoakEvent]:
        request = _request(_cell_target())
        runner = _make_runner(
            tmp_path, observer=_healthy_observer(), forms={"actor": [form]}, scheduler=_ScriptedScheduler([request])
        )
        with contextlib.suppress(AssertionError):
            await runner.run(
                _wait_until(lambda: any(isinstance(e, SoakActionResultEvent) for e in runner.event_log.events)),
                teardown=_RecordingTeardown(runner.event_log),
            )
        return runner.event_log.events

    async def test_a_request_is_recorded_before_its_effect_and_its_result(self, tmp_path: Path) -> None:
        """The request is logged first, then the reported effect, then a returned result."""
        form = _FakeForm()

        events = await self._run_one(tmp_path, form)

        kinds = [event.kind for event in events if event.kind.startswith("action_")]
        assert kinds == ["action_requested", "action_applied", "action_result"]
        [request] = form.executed
        requested, applied, result = (event for event in events if event.kind.startswith("action_"))
        assert requested.request == request
        assert applied.request_id == result.request_id == request.request_id
        assert result.returned is True and result.error is None

    async def test_a_form_failing_before_its_effect_records_no_effect_and_a_failed_result(
        self, tmp_path: Path
    ) -> None:
        """An exception before any effect is a failed action, never an applied one."""

        async def refuse(request: SoakActionRequest, report_applied: Callable[[SoakActionEvidence], None]) -> None:
            raise RuntimeError("refused")

        events = await self._run_one(tmp_path, _FakeForm(execute=refuse))

        assert not any(isinstance(event, SoakActionAppliedEvent) for event in events)
        [result] = [event for event in events if isinstance(event, SoakActionResultEvent)]
        assert result.returned is False
        assert "refused" in result.error

    async def test_a_form_failing_after_its_effect_keeps_the_effect_and_fails_the_result(self, tmp_path: Path) -> None:
        """The effect already happened, so it stays recorded next to the failed result."""

        async def apply_then_fail(
            request: SoakActionRequest, report_applied: Callable[[SoakActionEvidence], None]
        ) -> None:
            report_applied(_pod_evidence())
            raise RuntimeError("late failure")

        events = await self._run_one(tmp_path, _FakeForm(execute=apply_then_fail))

        kinds = [event.kind for event in events if event.kind.startswith("action_")]
        assert kinds == ["action_requested", "action_applied", "action_result"]
        assert events[[event.kind for event in events].index("action_result")].returned is False

    async def test_reporting_an_effect_twice_fails_the_action(self, tmp_path: Path) -> None:
        """One request has exactly one applied effect."""

        async def apply_twice(
            request: SoakActionRequest, report_applied: Callable[[SoakActionEvidence], None]
        ) -> None:
            report_applied(_pod_evidence())
            report_applied(_pod_evidence())

        events = await self._run_one(tmp_path, _FakeForm(execute=apply_twice))

        assert sum(isinstance(event, SoakActionAppliedEvent) for event in events) == 1
        [result] = [event for event in events if isinstance(event, SoakActionResultEvent)]
        assert result.returned is False
        assert "twice" in result.error
