from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import (
    _cell_target,
    _FakeForm,
    _now,
    _observation,
    _RecordingProcesses,
    _ScriptedObserver,
    _step_end,
    _wait_until,
    _write_sut_lines,
)
from tests.utils.soak.core import teardown as teardown_module
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.entrypoint import run_soak
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakCollectionClosedEvent,
    SoakEvidenceArchivedEvent,
    SoakObservationEvent,
    SoakRunContextEvent,
    SoakTeardownEvent,
    read_events,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend

_SUBMISSION_ID = "miles-soak-entry"


class _SoakWorld:
    def __init__(self, tmp_path: Path, *, last_step_rollout: int = 100, ready_until_call: int = 10**6) -> None:
        self.dump_dir = tmp_path / "dump"
        self.events_dir = self.dump_dir / EVENTS_DIRNAME
        self.events_dir.mkdir(parents=True)
        self.config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.RAY, run_id="260926-120000-000", ray_submission_id=_SUBMISSION_ID
        )
        self.event_log = EventLog(tmp_path / "evidence" / "events.jsonl")
        self.form = _FakeForm()
        self._last_step_rollout = last_step_rollout
        self._ready_until_call = ready_until_call
        self.observer = _ScriptedObserver(self._observe)

    def _observe(self, index: int) -> SoakObservationEvent:
        if index <= self._last_step_rollout:
            _write_sut_lines(self.events_dir / "trainer_controller_actor.jsonl", [_step_end(index, at=_now())])
        return _observation([_cell_target(ready=index < self._ready_until_call)], at=_now())

    async def run(self, *, expected_count: int = 1) -> None:
        await run_soak(
            config=self.config,
            dump_dir=self.dump_dir,
            sut_run=_wait_until(lambda: self.observer.calls >= 4),
            runner_config=SoakRunnerConfig(
                seed=0,
                poll_interval_seconds=0.001,
                tail=SoakTailConfig(close_after_rollout_id=1),
                target_configs={"actor": SoakTargetConfig(expected_count=expected_count, mean_interval_seconds=1e9)},
            ),
            forms={"actor": [self.form]},
            event_log=self.event_log,
            observer=self.observer,
            evidence_dir=self.dump_dir.parent / "stop-output",
        )


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> _RecordingProcesses:
    fake = _RecordingProcesses()
    monkeypatch.setattr(teardown_module, "run_process", fake)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    return fake


class TestRunSoak:
    async def test_a_complete_run_records_its_context_tears_down_its_job_and_archives_training_events(
        self, tmp_path: Path, processes: _RecordingProcesses
    ) -> None:
        """The real entrypoint wires context, live training events, teardown and archive into one closed log."""
        world = _SoakWorld(tmp_path)

        await world.run()

        events = world.event_log.events
        context = events[0]
        assert isinstance(context, SoakRunContextEvent)
        assert context.context.base_url == "http://localhost:18080"
        assert context.context.form_names == {"actor": ["fake"]}
        assert context.context.train_config == world.config
        assert context.sources == {"training_events": world.events_dir}
        assert processes.calls == [["ray", "job", "stop", "--address", "http://127.0.0.1:8265", _SUBMISSION_ID]]
        [teardown] = [event for event in events if isinstance(event, SoakTeardownEvent)]
        assert (teardown.resource, teardown.returned) == (f"ray-job:{_SUBMISSION_ID}", True)
        archived = events[-2]
        assert isinstance(archived, SoakEvidenceArchivedEvent)
        assert (archived.sources["training_events"] / "trainer_controller_actor.jsonl").is_file()
        assert isinstance(events[-1], SoakCollectionClosedEvent)
        assert read_events(world.event_log.path) == events

    async def test_live_training_events_reach_the_observations_through_the_real_feed(
        self, tmp_path: Path, processes: _RecordingProcesses
    ) -> None:
        """Steps written by training appear exactly once across the recorded observations."""
        world = _SoakWorld(tmp_path)

        await world.run()

        rollouts = [
            step.rollout_id
            for event in world.event_log.events
            if isinstance(event, SoakObservationEvent)
            for step in event.new_sut_events
        ]
        assert rollouts == list(range(len(rollouts)))
        assert len(rollouts) >= 4

    async def test_a_run_without_progress_after_admission_closed_is_rejected(
        self, tmp_path: Path, processes: _RecordingProcesses
    ) -> None:
        """Training that stops advancing at the closing rollout cannot prove recovery, yet evidence is closed."""
        world = _SoakWorld(tmp_path, last_step_rollout=1)

        with pytest.raises(AssertionError, match="Soak tail has no successful training progress"):
            await world.run()
        assert isinstance(world.event_log.events[-1], SoakCollectionClosedEvent)
        assert processes.calls
