from pathlib import Path
from subprocess import CompletedProcess

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import diagnosis


class TestCollectDiagnosis:
    def test_collection_preserves_outputs_and_reports_a_failed_pod_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Collection keeps command output and reports a failed describe without losing other artifacts."""
        kubectl_commands: list[list[str]] = []

        def fake_kubectl_process(
            command: list[str], *, capture_output: bool, check: bool, input: str | None
        ) -> CompletedProcess[str]:
            kubectl_commands.append(command)
            return CompletedProcess(
                command,
                returncode=0,
                stdout='{"items":[{"metadata":{"name":"trainer-0","uid":"trainer-uid"}}]}',
                stderr="",
            )

        def fake_run_process(command: list[str], *, capture_output: bool, check: bool) -> CompletedProcess[str]:
            if command[1:3] == ["describe", "pod"]:
                return CompletedProcess(command, returncode=1, stdout="describe stdout\n", stderr="describe stderr\n")
            artifact = "events" if command[1:3] == ["get", "events"] else "logs"
            return CompletedProcess(
                command,
                returncode=0,
                stdout=f"{artifact} stdout\n",
                stderr=f"{artifact} stderr\n",
            )

        monkeypatch.setattr(command_wrapper, "run_process", fake_kubectl_process)
        monkeypatch.setattr(diagnosis, "run_process", fake_run_process)

        result = diagnosis.collect_diagnosis(namespace="training", output_dir=tmp_path, selector="run=selected")

        assert kubectl_commands == [
            [
                "kubectl",
                "get",
                "pods",
                "--namespace",
                "training",
                "--output",
                "json",
                "--ignore-not-found",
                "--selector",
                "run=selected",
            ]
        ]
        assert (result.directory / "events.txt").read_text() == "events stdout\nevents stderr\n"
        assert (result.directory / "trainer-0.log").read_text() == "logs stdout\nlogs stderr\n"
        assert (result.directory / "trainer-0.describe.txt").read_text() == "describe stdout\ndescribe stderr\n"
        assert result.missing == ("describe of trainer-0",)
