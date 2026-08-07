import subprocess
from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend import kube, launcher, observe, run_state, watch


def _write(path: Path, status: str, generation: int, exit_code: int | None = None) -> None:
    run_state.write_orchestrator_state(path, status, exit_code=exit_code, generation=generation)


class TestGenerationSafeReset:
    def test_the_reset_claims_the_next_generation(self, tmp_path):
        """A relaunch has to invalidate the old verdict, or the launcher would return it immediately."""
        exit_file = tmp_path / "orchestrator.exit"
        _write(exit_file, run_state.STATUS_EXITED, generation=3, exit_code=7)

        generation = run_state.reset_for_new_generation(exit_file, 3)

        assert generation == 4
        assert run_state.read_orchestrator_state(exit_file).status == run_state.STATUS_STARTED

    def test_a_wrapper_that_finished_first_keeps_its_verdict(self, tmp_path):
        """A fast run publishes before the launcher resets, and clobbering it would hang the launcher forever."""
        exit_file = tmp_path / "orchestrator.exit"
        _write(exit_file, run_state.STATUS_EXITED, generation=5, exit_code=0)

        generation = run_state.reset_for_new_generation(exit_file, 3)

        assert generation == 5
        assert run_state.read_orchestrator_state(exit_file).status == run_state.STATUS_EXITED

    def test_a_stale_verdict_is_not_mistaken_for_this_launch(self, tmp_path):
        """The previous generation's exit code belongs to a run this launcher never started."""
        exit_file = tmp_path / "orchestrator.exit"
        _write(exit_file, run_state.STATUS_EXITED, generation=2, exit_code=1)

        outcome = watch.poll_once(exit_file=exit_file, read_pod_phase=lambda: "Running", min_generation=3)

        assert outcome is None

    def test_this_generation_verdict_is_accepted(self, tmp_path):
        """Once the wrapper answers for this launch, the launcher must return that exit code."""
        exit_file = tmp_path / "orchestrator.exit"
        _write(exit_file, run_state.STATUS_EXITED, generation=3, exit_code=4)

        outcome = watch.poll_once(exit_file=exit_file, read_pod_phase=lambda: "Running", min_generation=3)

        assert outcome is not None
        assert outcome.exit_code == 4


class TestSharedRootResolution:
    def test_the_run_directory_comes_from_the_infra_values(self, tmp_path):
        """Chart and launcher must resolve one path, or they read and write different files."""
        infra = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": {"runsSubPath": "teamdata"}}}

        assert launcher.resolve_shared_root(infra) == "/mnt/x/teamdata"

    def test_a_disagreeing_override_is_refused(self, tmp_path):
        """Silently preferring one of two answers is exactly how the two halves drift apart."""
        infra = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": {"runsSubPath": "teamdata"}}}

        try:
            launcher.resolve_shared_root(infra, override="/somewhere/else")
        except AssertionError as error:
            assert "shared_root" in str(error)
        else:
            raise AssertionError("a disagreeing override must not be accepted")


class TestLogStreamOrdering:
    def test_the_watcher_runs_while_the_log_streams(self, monkeypatch, tmp_path):
        """kubectl logs --follow never returns on a keep-alive pod, so a serial launcher would never poll."""
        exit_file = tmp_path / "orchestrator.exit"
        _write(exit_file, run_state.STATUS_EXITED, generation=1, exit_code=3)
        started: list[str] = []

        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False

            def wait(self, timeout: float | None = None) -> int:
                while not self.terminated:
                    pass
                return 0

            def poll(self) -> int | None:
                return 0 if self.terminated else None

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.terminated = True

        process = FakeProcess()

        def fake_popen(command, *args, **kwargs):
            started.append(command[0])
            return process

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            kube,
            "release_pods",
            lambda namespace, release: [observe.PodStatus(name="p", phase="Running", ready=True, restarts=0)],
        )
        monkeypatch.setattr(kube, "pod_events", lambda namespace, pods: [])
        monkeypatch.setattr(kube, "pod_phase", lambda namespace, workload: "Running")
        run = launcher.LaunchedRun(release="myrun", namespace="myns", exit_file=exit_file, generation=1)

        exit_code = launcher.follow_until_finished(run, log=lambda message: None)

        assert exit_code == 3
        assert started == ["kubectl"]
        assert process.terminated
