import logging
from collections.abc import Callable
from pathlib import Path


from miles.utils.external_utils.command_utils.helm_backend.naming import _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _state_file(tmp_path):
    return _orchestrator_state_path(tmp_path, "260101-000000-000001")


def _observed(phase: str, *, startup_failure: str | None = None) -> observer.ObservedPod:
    return observer.ObservedPod(phase=phase, startup_failure=startup_failure)


class TestComputeRunOutcome:
    def test_keeps_waiting_while_the_run_is_healthy(self, tmp_path):
        """A running orchestrator has written started and its pod is Running."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(path),
                observed=_observed("Running"),
                missing_polls=0,
                dead_polls=0,
                failing_polls=0,
            )
            is None
        )

    def test_passes_the_orchestrator_exit_code_through(self, tmp_path):
        """A failed experiment has to fail the caller's shell, not just be visible in the logs."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=7)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=_observed("Running"),
            missing_polls=0,
            dead_polls=0,
            failing_polls=0,
        )

        assert outcome.exit_code == 7

    def test_prefers_the_state_file_over_the_pod_phase(self, tmp_path):
        """The wrapper stays alive after writing, so the pod's phase says nothing about the outcome."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=0)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=_observed("Failed"),
            missing_polls=0,
            dead_polls=0,
            failing_polls=0,
        )

        assert outcome.exit_code == 0

    def test_ends_the_run_when_the_pod_died_without_a_verdict(self, tmp_path):
        """A crash before the wrapper ran would otherwise leave the launcher polling forever."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=_observed("Failed"),
            missing_polls=0,
            dead_polls=observer._DEAD_POD_POLLS,
            failing_polls=0,
        )

        assert outcome.exit_code == 1
        assert "without writing an exit code" in outcome.reason

    def test_a_succeeded_pod_ends_a_run_without_a_verdict(self, tmp_path: Path) -> None:
        """A completed pod without a state-file verdict must not leave the launcher polling forever."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=_observed("Succeeded"),
            missing_polls=0,
            dead_polls=observer._DEAD_POD_POLLS,
            failing_polls=0,
        )

        assert outcome.exit_code == 1
        assert "reached Succeeded without writing an exit code" in outcome.reason

    def test_ends_the_run_when_the_pod_stayed_gone(self, tmp_path):
        """Someone deleting the release mid-run must not hang the launcher either."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=None,
            missing_polls=observer._MISSING_POD_POLLS,
            dead_polls=0,
            failing_polls=0,
        )

        assert outcome.exit_code == 1

    def test_waits_out_a_single_poll_that_found_no_pod(self, tmp_path):
        """A StatefulSet deletes a pod before creating its replacement, which is a handover, not a loss."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(path), observed=None, missing_polls=1, dead_polls=0, failing_polls=0
            )
            is None
        )

    def test_never_reads_a_terminal_state_that_names_no_exit_code_as_a_verdict(self, tmp_path):
        """A file half-written by an older wrapper must not read as the success it never reported."""
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "exited", "exit_code": null, "timestamp": 0.0}')

        assert OrchestratorState.read(path) is None
        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(path),
                observed=_observed("Running"),
                missing_polls=0,
                dead_polls=0,
                failing_polls=0,
            )
            is None
        )

    def test_ignores_a_state_file_whose_shape_it_does_not_know(self, tmp_path):
        """A future or corrupt writer must not crash the launcher out of a run that is still going."""
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "exited", "surprise": 1}')

        assert OrchestratorState.read(path) is None

    def test_waits_out_the_gap_between_install_and_the_first_pod(self, tmp_path):
        """A missing pod before the run has started is scheduling, not failure."""
        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(_state_file(tmp_path)),
            observed=None,
            missing_polls=0,
            dead_polls=0,
            failing_polls=0,
        )

        assert outcome is None

    def test_ends_the_run_when_the_container_cannot_start(self, tmp_path):
        """A wrapper that never runs writes no verdict, and CrashLoopBackOff is neither Failed nor gone."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path),
            observed=_observed("Pending", startup_failure="ImagePullBackOff"),
            missing_polls=0,
            dead_polls=0,
            failing_polls=observer._FAILING_POD_POLLS,
        )

        assert outcome.exit_code == 1
        assert "ImagePullBackOff" in outcome.reason

    def test_waits_out_a_container_that_has_only_just_backed_off(self, tmp_path):
        """One backoff poll can be the first restart of a container the run goes on to survive."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(path),
                observed=_observed("Pending", startup_failure="CrashLoopBackOff"),
                missing_polls=0,
                dead_polls=0,
                failing_polls=1,
            )
            is None
        )

    def test_keeps_waiting_through_a_pod_restart(self, tmp_path):
        """Pending covers image pulls and reschedules, which are normal before training begins."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(path),
                observed=_observed("Pending"),
                missing_polls=0,
                dead_polls=0,
                failing_polls=0,
            )
            is None
        )


def _reader(scripted: tuple[object, ...], afterwards: Path) -> Callable[[], Path]:
    reads = iter(scripted)

    def read() -> Path:
        result = next(reads, afterwards)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, Path)
        return result

    return read


class TestWaitForRun:
    def test_follows_the_replacement_orchestrator_generation(self, tmp_path, monkeypatch):
        """A hot restart's old SIGTERM verdict must not finish the replacement generation."""
        old_path = _orchestrator_state_path(tmp_path, "old")
        new_path = _orchestrator_state_path(tmp_path, "new")
        _write(old_path, OrchestratorStatus.EXITED, exit_code=143)
        _write(new_path, OrchestratorStatus.STARTED)

        def sleep(seconds: float) -> None:
            _write(new_path, OrchestratorStatus.EXITED, exit_code=0)

        monkeypatch.setattr(observer.time, "sleep", sleep)
        outcome = observer.wait_for_run(
            state_file=old_path,
            read_pod=lambda: _observed("Running"),
            read_active_state_file=lambda: new_path,
        )

        assert outcome.exit_code == 0

    def test_dead_pod_grace_follows_a_generation_published_after_two_polls(self, tmp_path, monkeypatch):
        """A replacement published during a dead-pod grace makes the old failure stale."""
        old_path = _orchestrator_state_path(tmp_path, "old")
        new_path = _orchestrator_state_path(tmp_path, "new")
        _write(old_path, OrchestratorStatus.STARTED)
        _write(new_path, OrchestratorStatus.EXITED, exit_code=0)

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(
            state_file=old_path,
            read_pod=lambda: _observed("Failed"),
            read_active_state_file=_reader((old_path, old_path, new_path), afterwards=new_path),
        )

        assert outcome.exit_code == 0

    def test_dead_pod_grace_fails_after_stable_polls_in_the_same_generation(self, tmp_path, monkeypatch):
        """A pod that stays dead in one generation still fails without a state verdict."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)
        phase_reads = []

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=lambda: phase_reads.append(True) or _observed("Failed"),
            read_active_state_file=lambda: path,
        )

        assert outcome.exit_code == 1
        assert len(phase_reads) == observer._DEAD_POD_POLLS

    def test_terminal_state_bypasses_the_dead_pod_grace(self, tmp_path):
        """An explicit state-file verdict is authoritative on the first poll."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=7)
        phase_reads = []

        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=lambda: phase_reads.append(True) or _observed("Failed"),
            read_active_state_file=lambda: path,
        )

        assert outcome.exit_code == 7
        assert phase_reads == [True]

    def test_does_not_accept_a_verdict_without_confirming_the_active_generation(self, tmp_path, monkeypatch):
        """A failed identity lookup cannot turn a replaced orchestrator's verdict into the run verdict."""
        old_path = _orchestrator_state_path(tmp_path, "old")
        new_path = _orchestrator_state_path(tmp_path, "new")
        _write(old_path, OrchestratorStatus.EXITED, exit_code=143)
        _write(new_path, OrchestratorStatus.EXITED, exit_code=0)

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(
            state_file=old_path,
            read_pod=lambda: _observed("Running"),
            read_active_state_file=_reader((RuntimeError("unreachable"),), new_path),
        )

        assert outcome.exit_code == 0

    def test_retries_the_recheck_it_could_not_read_instead_of_taking_the_verdict(self, tmp_path, monkeypatch):
        """Failing the confirming read open is the same as never confirming, and hands back the stale verdict."""
        old_path = _orchestrator_state_path(tmp_path, "old")
        new_path = _orchestrator_state_path(tmp_path, "new")
        _write(old_path, OrchestratorStatus.EXITED, exit_code=143)
        _write(new_path, OrchestratorStatus.EXITED, exit_code=0)

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(
            state_file=old_path,
            read_pod=lambda: _observed("Running"),
            read_active_state_file=_reader((old_path, RuntimeError("unreachable")), new_path),
        )

        assert outcome.exit_code == 0

    def test_polls_until_an_outcome_appears(self, tmp_path, monkeypatch):
        """The launcher blocks like ray submit does, so it must keep looking rather than check once."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                _write(path, OrchestratorStatus.EXITED, exit_code=0)

        monkeypatch.setattr(observer.time, "sleep", sleep)
        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=lambda: _observed("Running"),
            read_active_state_file=lambda: path,
        )

        assert outcome.exit_code == 0
        assert sleeps == [observer._POLL_INTERVAL_SECONDS] * 3

    def test_does_not_end_a_healthy_run_because_the_api_server_stopped_answering(self, tmp_path, monkeypatch):
        """A network blip reads as an unanswerable question, not as the orchestrator pod being gone."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)
        polls = []

        def unreachable_until_run_finishes():
            polls.append(True)
            if len(polls) > observer._MISSING_POD_POLLS + 1:
                _write(path, OrchestratorStatus.EXITED, exit_code=0)
            raise RuntimeError("the api server is unreachable")

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=unreachable_until_run_finishes,
            read_active_state_file=lambda: path,
        )

        assert outcome.exit_code == 0

    def test_reports_why_the_run_ended(self, tmp_path, monkeypatch, caplog):
        """A run that failed without a verdict looks identical to a clean one unless the reason is logged."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        with caplog.at_level(logging.INFO, logger=observer.__name__):
            observer.wait_for_run(
                state_file=path,
                read_pod=lambda: _observed("Failed"),
                read_active_state_file=lambda: path,
            )

        assert "exit code 1" in caplog.text
