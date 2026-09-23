import asyncio
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
    _wait_until,
)
from tests.utils.soak.core.config import SoakTimeoutConfig
from tests.utils.soak.core.events import SoakCollectionClosedEvent, SoakObservationEvent, read_events


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
