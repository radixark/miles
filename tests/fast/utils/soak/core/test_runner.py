import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import (
    _cell_target,
    _flatten_errors,
    _healthy_observer,
    _make_runner,
    _now,
    _observation,
    _RecordingTeardown,
    _runner_config,
    _ScriptedObserver,
    _ScriptedSutFeed,
    _step_end,
    _wait_until,
)
from tests.utils.soak.core.config import SoakTailConfig, SoakTimeoutConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakAdmissionClosedEvent,
    SoakCollectionClosedEvent,
    SoakObservationEvent,
    read_events,
)
from tests.utils.soak.core.runner import SoakRunner, _assert_within_tail_budget


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

    def test_a_recovery_tail_without_live_training_events_is_refused(self, tmp_path: Path) -> None:
        """Closing admission depends on training progress, so a tail needs the training event feed."""
        with pytest.raises(ValueError, match="live training events"):
            SoakRunner(
                observer=_healthy_observer(),
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
