import logging


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


class TestComputeRunOutcome:
    def test_keeps_waiting_while_the_run_is_healthy(self, tmp_path):
        """A running orchestrator has written started and its pod is Running."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Running", missing_polls=0) is None
        )

    def test_passes_the_orchestrator_exit_code_through(self, tmp_path):
        """A failed experiment has to fail the caller's shell, not just be visible in the logs."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=7)

        outcome = observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Running", missing_polls=0)

        assert outcome.exit_code == 7

    def test_prefers_the_state_file_over_the_pod_phase(self, tmp_path):
        """The wrapper stays alive after writing, so the pod's phase says nothing about the outcome."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.EXITED, exit_code=0)

        outcome = observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Failed", missing_polls=0)

        assert outcome.exit_code == 0

    def test_ends_the_run_when_the_pod_died_without_a_verdict(self, tmp_path):
        """A crash before the wrapper ran would otherwise leave the launcher polling forever."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Failed", missing_polls=0)

        assert outcome.exit_code == 1
        assert "without writing an exit code" in outcome.reason

    def test_ends_the_run_when_the_pod_stayed_gone(self, tmp_path):
        """Someone deleting the release mid-run must not hang the launcher either."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(path), phase=None, missing_polls=observer._MISSING_POD_POLLS
        )

        assert outcome.exit_code == 1

    def test_waits_out_a_single_poll_that_found_no_pod(self, tmp_path):
        """A StatefulSet deletes a pod before creating its replacement, which is a handover, not a loss."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert observer._compute_run_outcome(state=OrchestratorState.read(path), phase=None, missing_polls=1) is None

    def test_never_reads_a_terminal_state_that_names_no_exit_code_as_a_verdict(self, tmp_path):
        """A file half-written by an older wrapper must not read as the success it never reported."""
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "exited", "exit_code": null, "timestamp": 0.0}')

        assert OrchestratorState.read(path) is None
        assert (
            observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Running", missing_polls=0) is None
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
            state=OrchestratorState.read(_state_file(tmp_path)), phase=None, missing_polls=0
        )

        assert outcome is None

    def test_keeps_waiting_through_a_pod_restart(self, tmp_path):
        """Pending covers image pulls and reschedules, which are normal before training begins."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        assert (
            observer._compute_run_outcome(state=OrchestratorState.read(path), phase="Pending", missing_polls=0) is None
        )


class TestWaitForRun:
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
        outcome = observer.wait_for_run(state_file=path, read_pod_phase=lambda: "Running")

        assert outcome.exit_code == 0
        assert sleeps == [observer._POLL_INTERVAL_SECONDS] * 3

    def test_does_not_end_a_healthy_run_because_the_api_server_stopped_answering(self, tmp_path, monkeypatch):
        """A network blip reads as an unanswerable question, not as the orchestrator pod being gone."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)
        polls = []

        def unreachable_until_the_run_finishes():
            polls.append(True)
            if len(polls) > observer._MISSING_POD_POLLS + 1:
                _write(path, OrchestratorStatus.EXITED, exit_code=0)
            raise RuntimeError("the api server is unreachable")

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        outcome = observer.wait_for_run(state_file=path, read_pod_phase=unreachable_until_the_run_finishes)

        assert outcome.exit_code == 0

    def test_reports_why_the_run_ended(self, tmp_path, monkeypatch, caplog):
        """A run that failed without a verdict looks identical to a clean one unless the reason is logged."""
        path = _state_file(tmp_path)
        _write(path, OrchestratorStatus.STARTED)

        monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)
        with caplog.at_level(logging.INFO, logger=observer.__name__):
            observer.wait_for_run(state_file=path, read_pod_phase=lambda: "Failed")

        assert "exit code 1" in caplog.text
