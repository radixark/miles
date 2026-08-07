import json
import signal
import sys

import pytest

from miles.utils.external_utils.command_utils.helm_backend import orchestrator_wrapper, run_state


def _state(path):
    return json.loads(path.read_text())


class TestMain:
    def test_publishes_a_successful_exit_code(self, tmp_path):
        """The launcher reads this file to learn the run finished, so a clean run must record zero."""
        exit_file = tmp_path / "orchestrator.exit"

        code = orchestrator_wrapper.main(
            ["--exit-file", str(exit_file), "--no-keep-alive", "--", sys.executable, "-c", "pass"]
        )

        assert code == 0
        assert (_state(exit_file)["status"], _state(exit_file)["exit_code"]) == ("exited", 0)

    def test_publishes_a_failing_exit_code(self, tmp_path):
        """A failed experiment has to reach the caller's shell, not just the pod logs."""
        exit_file = tmp_path / "orchestrator.exit"

        code = orchestrator_wrapper.main(
            ["--exit-file", str(exit_file), "--no-keep-alive", "--", sys.executable, "-c", "raise SystemExit(7)"]
        )

        assert code == 7
        assert _state(exit_file)["exit_code"] == 7

    def test_marks_the_run_started_before_the_script_runs(self, tmp_path):
        """Otherwise a launcher polling an absent file cannot tell a slow start from a lost run."""
        exit_file = tmp_path / "orchestrator.exit"
        observed = tmp_path / "observed"
        script = f"import shutil; shutil.copyfile({str(exit_file)!r}, {str(observed)!r})"

        orchestrator_wrapper.main(
            ["--exit-file", str(exit_file), "--no-keep-alive", "--", sys.executable, "-c", script]
        )

        assert json.loads(observed.read_text())["status"] == "started"
        assert _state(exit_file)["status"] == "exited"

    def test_records_a_verdict_when_the_script_cannot_be_launched(self, tmp_path):
        """A wrapper that died silently would leave the launcher polling a started run forever."""
        exit_file = tmp_path / "orchestrator.exit"

        with pytest.raises(OSError):
            orchestrator_wrapper.main(
                ["--exit-file", str(exit_file), "--no-keep-alive", "--", str(tmp_path / "not-an-executable")]
            )

        assert (_state(exit_file)["status"], _state(exit_file)["exit_code"]) == ("exited", 1)

    def test_publishes_a_verdict_when_the_pod_is_asked_to_stop(self, tmp_path):
        """A signal kills python without raising, and a restart would then silently re-run the training."""
        exit_file = tmp_path / "orchestrator.exit"
        script = "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"

        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(
                ["--exit-file", str(exit_file), "--no-keep-alive", "--", sys.executable, "-c", script]
            )

        assert _state(exit_file) == {
            "status": "exited",
            "exit_code": 128 + signal.SIGTERM,
            "timestamp": _state(exit_file)["timestamp"],
            "generation": 1,
        }

    def test_requires_a_command(self, tmp_path):
        """A wrapper with nothing to run is a chart bug, and must fail loudly at startup."""
        with pytest.raises(AssertionError):
            orchestrator_wrapper.main(["--exit-file", str(tmp_path / "orchestrator.exit")])

    def test_writes_where_the_run_directory_layout_says(self, tmp_path):
        """The chart passes a path derived from the shared root, so both sides agree without extra config."""
        exit_file = run_state.orchestrator_exit_path(run_state.run_dir(tmp_path, "260101-000000-000"))

        orchestrator_wrapper.main(
            ["--exit-file", str(exit_file), "--no-keep-alive", "--", sys.executable, "-c", "pass"]
        )

        assert exit_file.is_file()
