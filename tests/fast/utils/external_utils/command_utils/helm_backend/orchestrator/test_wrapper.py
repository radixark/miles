import json
import os
import signal
import subprocess
import sys

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import state as orchestrator_state
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import wrapper as orchestrator_wrapper

MANIFEST = "/etc/miles-uninstall/uninstall-job.yaml"


def _state(path):
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _no_keep_alive(monkeypatch):
    monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: None)


@pytest.fixture(autouse=True)
def slept(monkeypatch):
    waits = []
    monkeypatch.setattr(orchestrator_wrapper, "sleep", lambda seconds: waits.append(seconds))
    return waits


@pytest.fixture
def kubectl_calls(monkeypatch):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(list(arguments))
        return subprocess.CompletedProcess(args=list(arguments), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(Kubectl, "_run", staticmethod(fake_run))
    return calls


def _refusal(stderr: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


class TestMain:
    def test_publishes_a_successful_exit_code(self, tmp_path):
        """The launcher reads this file to learn the run finished, so a clean run must record zero."""
        state_file = tmp_path / "orchestrator.state"

        code = orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert code == 0
        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 0)

    def test_publishes_a_failing_exit_code(self, tmp_path):
        """A failed experiment has to reach the caller's shell, not just the pod logs."""
        state_file = tmp_path / "orchestrator.state"

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(7)"]
        )

        assert code == 7
        assert _state(state_file)["exit_code"] == 7

    def test_marks_the_run_started_before_the_script_runs(self, tmp_path):
        """Otherwise a launcher polling an absent file cannot tell a slow start from a lost run."""
        state_file = tmp_path / "orchestrator.state"
        observed = tmp_path / "observed"
        script = f"import shutil; shutil.copyfile({str(state_file)!r}, {str(observed)!r})"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert json.loads(observed.read_text())["status"] == "started"
        assert _state(state_file)["status"] == "exited"

    def test_records_a_verdict_when_the_script_cannot_be_launched(self, tmp_path):
        """A wrapper that died silently would leave the launcher polling a started run forever."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", str(tmp_path / "not-an-executable")])

        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 127)

    def test_publishes_a_verdict_when_the_pod_is_asked_to_stop(self, tmp_path):
        """A signal kills python without raising, and a restart would then silently re-run the training."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"

        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 128 + signal.SIGTERM)

    def test_keeps_the_verdict_when_the_pod_is_torn_down_afterwards(self, tmp_path, monkeypatch):
        """A helm uninstall during keep-alive would otherwise overwrite the run's real exit code with 143."""
        state_file = tmp_path / "orchestrator.state"

        def stop_pod():
            os.kill(os.getpid(), signal.SIGTERM)

        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", stop_pod)
        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(
                ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(7)"]
            )

        assert _state(state_file)["exit_code"] == 7

    def test_does_not_run_a_script_whose_verdict_is_already_recorded(self, tmp_path):
        """The pod restarts on its own, and rerunning training that already finished would lose its result."""
        state_file = tmp_path / "orchestrator.state"
        ran = tmp_path / "ran"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"]
        )

        assert code == 4
        assert not ran.exists()

    def test_keeps_an_inherited_verdict_when_the_restarted_pod_is_torn_down(self, tmp_path, monkeypatch):
        """A pod rebuilt after its run finished skips the script, and its teardown must not rewrite the result."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: os.kill(os.getpid(), signal.SIGTERM))
        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert _state(state_file)["exit_code"] == 4

    def test_reruns_a_script_whose_recorded_verdict_carries_no_exit_code(self, tmp_path):
        """A terminal state with no code is a corrupt file, and returning it would exit the pod as a success."""
        state_file = tmp_path / "orchestrator.state"
        state_file.write_text(json.dumps({"status": "exited", "exit_code": None, "timestamp": 0.0}))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(5)"]
        )

        assert code == 5
        assert _state(state_file)["exit_code"] == 5

    def test_reports_a_signalled_script_as_a_shell_exit_code(self, tmp_path):
        """A negative return code is not an exit status, and the launcher passes this straight to its caller."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"

        code = orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert code == 128 + signal.SIGKILL
        assert _state(state_file)["exit_code"] == 128 + signal.SIGKILL

    def test_stays_alive_after_the_script_finishes(self, tmp_path, monkeypatch):
        """The pod's logs die with its process, so exiting would take the run's own output with it."""
        state_file = tmp_path / "orchestrator.state"
        stayed = []
        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: stayed.append(True))

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert stayed == [True]

    def test_requires_a_command(self, tmp_path):
        """A wrapper with nothing to run is a chart bug, and must fail loudly at startup."""
        with pytest.raises(AssertionError):
            orchestrator_wrapper.main(["--state-file", str(tmp_path / "orchestrator.state")])

    def test_ignores_a_started_verdict_of_its_own_first_attempt(self, tmp_path):
        """Nothing else distinguishes a restarted pod from a first start, so the exit file has to."""
        state_file = tmp_path / "orchestrator.state"
        ran = tmp_path / "ran"
        orchestrator_state.OrchestratorState(status=orchestrator_state.OrchestratorStatus.STARTED).write(state_file)

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"]
        )

        assert code == 1
        assert not ran.exists()
        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 1)

    def test_writes_where_the_run_directory_layout_says(self, tmp_path):
        """The chart passes a path derived from the shared root, so both sides agree without extra config."""
        state_file = _orchestrator_state_path(
            RunFiles.run_dir(shared_root=tmp_path, run_id="260101-000000-000"), "260101-000000-000001"
        )

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert state_file.is_file()


class TestUninstallJob:
    def test_creates_the_job_as_soon_as_the_run_has_a_verdict(self, tmp_path, kubectl_calls):
        """The wrapper could die during the grace period, so the job carries the delay and is created at once."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_for_a_failed_run_too(self, tmp_path, kubectl_calls):
        """A failed run holds just as many gpus as a successful one, and its logs stay in the run directory."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            [
                "--state-file",
                str(state_file),
                "--uninstall-manifest",
                MANIFEST,
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_when_the_script_could_not_be_launched(self, tmp_path, kubectl_calls):
        """That verdict ends the run as surely as a failing script, and the release must go the same way."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            [
                "--state-file",
                str(state_file),
                "--uninstall-manifest",
                MANIFEST,
                "--",
                str(tmp_path / "not-an-executable"),
            ]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_never_creates_the_job_when_the_pod_is_asked_to_stop(self, tmp_path, kubectl_calls):
        """A relaunch SIGTERMs the orchestrator it replaces, and uninstalling then would tear down the new run."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"

        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(
                ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", script]
            )

        assert kubectl_calls == []

    def test_creates_the_job_a_restarted_pod_found_no_trace_of(self, tmp_path, kubectl_calls):
        """The wrapper may have written its verdict and then died before creating the job."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 4
        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_for_a_run_a_restart_cut_short(self, tmp_path, kubectl_calls):
        """A pod that restarted mid-training reports a failure, and that failure ends the release too."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_state.OrchestratorState(status=orchestrator_state.OrchestratorStatus.STARTED).write(state_file)

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 1
        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_treats_a_job_someone_already_created_as_success(self, tmp_path, monkeypatch):
        """A second attempt is normal after a restart, and it must not turn a finished run into a failure."""
        state_file = tmp_path / "orchestrator.state"
        refusal = _refusal('Error from server (AlreadyExists): jobs "u" already exists')
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: refusal))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 0

    def test_keeps_the_run_alive_when_every_attempt_is_refused(self, tmp_path, monkeypatch, slept):
        """The run's own outcome is what the launcher waits for; a leaked release is the smaller problem."""
        state_file = tmp_path / "orchestrator.state"
        refusal = _refusal("Error from server: forbidden")
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: refusal))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 0
        assert _state(state_file)["exit_code"] == 0
        assert slept == list(orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS)

    def test_retries_a_create_the_cluster_refused_once(self, tmp_path, monkeypatch, slept):
        """An apiserver that was busy for a moment must not cost the release its only cleaner."""
        state_file = tmp_path / "orchestrator.state"
        answers = [_refusal("Error from server: try again"), subprocess.CompletedProcess(args=[], returncode=0)]
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: answers.pop(0)))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]

    def test_stops_retrying_as_soon_as_the_job_is_there(self, tmp_path, monkeypatch, slept):
        """A retry that finds the job someone else created has nothing left to do."""
        state_file = tmp_path / "orchestrator.state"
        answers = [
            _refusal("Error from server: try again"),
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
        ]
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: answers.pop(0)))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]

    def test_touches_no_cluster_when_no_manifest_was_rendered(self, tmp_path, kubectl_calls):
        """A cluster with no account for it renders no manifest, and the run must not try to uninstall itself."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert kubectl_calls == []
