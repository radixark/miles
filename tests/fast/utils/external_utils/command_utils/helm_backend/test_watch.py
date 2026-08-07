from miles.utils.external_utils.command_utils.helm_backend import run_state, watch


def _exit_file(tmp_path):
    return tmp_path / "orchestrator.exit"


class TestPollOnce:
    def test_keeps_waiting_while_the_run_is_healthy(self, tmp_path):
        """A running orchestrator has written started and its pod is Running."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        assert watch.poll_once(exit_file=path, read_pod_phase=lambda: "Running") is None

    def test_passes_the_orchestrator_exit_code_through(self, tmp_path):
        """A failed experiment has to fail the caller's shell, not just be visible in the logs."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=7)

        outcome = watch.poll_once(exit_file=path, read_pod_phase=lambda: "Running")

        assert outcome.exit_code == 7

    def test_prefers_the_exit_file_over_the_pod_phase(self, tmp_path):
        """The wrapper stays alive after writing, so the pod's phase says nothing about the outcome."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=0)

        outcome = watch.poll_once(exit_file=path, read_pod_phase=lambda: "Failed")

        assert outcome.exit_code == 0

    def test_ends_the_run_when_the_pod_died_without_a_verdict(self, tmp_path):
        """A crash before the wrapper ran would otherwise leave the launcher polling forever."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        outcome = watch.poll_once(exit_file=path, read_pod_phase=lambda: "Failed")

        assert outcome.exit_code == 1
        assert "without writing an exit code" in outcome.reason

    def test_ends_the_run_when_the_pod_disappeared(self, tmp_path):
        """Someone deleting the release mid-run must not hang the launcher either."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        outcome = watch.poll_once(exit_file=path, read_pod_phase=lambda: None)

        assert outcome.exit_code == 1

    def test_waits_out_the_gap_between_install_and_the_first_pod(self, tmp_path):
        """A missing pod before the run has started is scheduling, not failure."""
        assert watch.poll_once(exit_file=_exit_file(tmp_path), read_pod_phase=lambda: None) is None

    def test_keeps_waiting_through_a_pod_restart(self, tmp_path):
        """Pending covers image pulls and reschedules, which are normal before training begins."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)

        assert watch.poll_once(exit_file=path, read_pod_phase=lambda: "Pending") is None


class TestWaitForRun:
    def test_polls_until_an_outcome_appears(self, tmp_path):
        """The launcher blocks like ray submit does, so it must keep looking rather than check once."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                run_state.write_orchestrator_state(path, run_state.STATUS_EXITED, exit_code=0)

        outcome = watch.wait_for_run(
            exit_file=path, read_pod_phase=lambda: "Running", sleep=sleep, log=lambda message: None
        )

        assert outcome.exit_code == 0
        assert len(sleeps) == 3

    def test_reports_why_the_run_ended(self, tmp_path):
        """A run that failed without a verdict looks identical to a clean one unless the reason is printed."""
        path = _exit_file(tmp_path)
        run_state.write_orchestrator_state(path, run_state.STATUS_STARTED)
        logged = []

        watch.wait_for_run(
            exit_file=path,
            read_pod_phase=lambda: "Failed",
            sleep=lambda seconds: None,
            log=logged.append,
        )

        assert "exit code 1" in logged[0]
