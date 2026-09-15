import re
import subprocess
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import diagnosis

_NEVER_RESTARTED = 'Error from server (BadRequest): previous terminated container "app" in pod "trainer-0" not found'
_API_SERVER_BLINKED = "Error from server: etcdserver: request timed out"


def _collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pods: list[str] | None,
    previous_failure: str | None = None,
) -> diagnosis.Diagnosis:
    def run_process(command: list[str], capture_output: bool, check: bool) -> subprocess.CompletedProcess:
        if "--previous" in command and previous_failure is not None:
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=previous_failure)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="captured\n", stderr="")

    monkeypatch.setattr(diagnosis, "run_process", run_process)
    monkeypatch.setattr(diagnosis, "_pod_names", lambda *, namespace, selector: pods)
    return diagnosis.collect_diagnosis(namespace="rl", output_dir=tmp_path)


class TestPreviousLogsOfAPod:
    def test_a_container_that_never_restarted_leaves_the_diagnosis_complete(self, monkeypatch, tmp_path):
        """Most pods of a healthy-looking failure never restarted, and that is not evidence going missing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert collected.is_complete
        assert collected.missing == ()

    def test_a_container_that_never_restarted_gets_no_previous_log_file(self, monkeypatch, tmp_path):
        """A file holding nothing but the api server's refusal reads as a crash log that says nothing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert not (collected.directory / "trainer-0.previous.log").exists()

    def test_an_api_error_is_reported_rather_than_read_as_a_pod_that_never_restarted(self, monkeypatch, tmp_path):
        """The crash log is the whole point of the diagnosis, and losing it silently is what hid the crash."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert not collected.is_complete
        assert "previous logs of trainer-0" in collected.missing

    def test_an_api_error_is_written_down_where_the_crash_log_would_be(self, monkeypatch, tmp_path):
        """Whoever reads the directory has to be able to see why the file is not the log they wanted."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert _API_SERVER_BLINKED in (collected.directory / "trainer-0.previous.log").read_text()

    def test_a_collection_that_captured_everything_is_complete(self, monkeypatch, tmp_path):
        """A diagnosis reported as incomplete sends its reader looking for evidence that is right there."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert collected.is_complete
        assert (collected.directory / "trainer-0.previous.log").read_text() == "captured\n"


class TestThePodsACollectionCovers:
    def test_a_run_with_no_pods_left_is_not_a_complete_diagnosis(self, monkeypatch, tmp_path):
        """A directory holding nothing but namespace events answered no question about the failed run."""
        collected = _collect(monkeypatch, tmp_path, pods=[])

        assert not collected.is_complete
        assert "pods of the run in namespace rl" in collected.missing

    def test_a_pod_listing_that_failed_is_reported_as_missing(self, monkeypatch, tmp_path):
        """Not knowing which pods exist is a different gap from knowing there are none."""
        collected = _collect(monkeypatch, tmp_path, pods=None)

        assert "pod listing in namespace rl" in collected.missing
        assert "pods of the run in namespace rl" not in collected.missing

    def test_every_pod_of_the_run_is_captured(self, monkeypatch, tmp_path):
        """A split run fails in one of its deployments, and each pod's log is a candidate answer."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0", "engine-0"])

        assert (collected.directory / "trainer-0.log").exists()
        assert (collected.directory / "engine-0.describe.txt").exists()
        assert collected.is_complete


class TestTheDirectoryACollectionGetsForItself:
    def test_two_collections_of_one_second_do_not_share_a_directory(self, monkeypatch, tmp_path):
        """A relaunch diagnoses twice within a second, and one directory means one set of files."""
        first = _collect(monkeypatch, tmp_path, pods=["trainer-0"])
        second = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert first.directory != second.directory

    def test_the_name_carries_the_microsecond_the_collection_started_at(self, monkeypatch, tmp_path):
        """Second resolution is what let two collections land in one directory in the first place."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert re.fullmatch(r"miles-diagnosis-rl-\d{8}-\d{6}-\d{6}", collected.directory.name)

    def test_a_collection_into_a_directory_that_exists_fails_loudly(self, monkeypatch, tmp_path):
        """Silently reusing it interleaves two runs' evidence, which is worse than collecting nothing."""
        monkeypatch.setattr(diagnosis, "datetime", SimpleNamespace(now=lambda: datetime(2026, 1, 1, 0, 0, 0, 5)))

        _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        with pytest.raises(FileExistsError):
            _collect(monkeypatch, tmp_path, pods=["trainer-0"])

    def test_the_name_says_which_namespace_was_diagnosed(self, monkeypatch, tmp_path):
        """One output directory holds the diagnoses of every run a user launched from this machine."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert collected.directory.name.startswith("miles-diagnosis-rl-")
        assert collected.directory.parent == tmp_path


class TestCollectDiagnosis:
    def test_collection_preserves_outputs_and_reports_a_failed_pod_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Collection keeps command output and reports a failed describe without losing other artifacts."""
        kubectl_commands: list[list[str]] = []

        def fake_kubectl_process(
            command: list[str], *, capture_output: bool, check: bool, input: str | None, timeout: float | None
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
