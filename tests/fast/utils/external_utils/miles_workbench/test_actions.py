from __future__ import annotations

import logging
import subprocess

import pytest

from miles.utils.external_utils.miles_workbench import actions as actions_module


class TestRun:
    def test_a_failed_external_command_exits_with_its_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed external command exits with the exact status reported by the process."""

        def run_process(command: list[str], *, capture_output: bool, check: bool) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=command, returncode=7, stdout="", stderr="")

        monkeypatch.setattr(actions_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(actions_module, "run_process", run_process)

        with pytest.raises(SystemExit) as raised:
            actions_module._run(["kubectl", "get", "pods"])

        assert raised.value.code == 7


class TestEnsureNamespace:
    def test_an_inconclusive_namespaced_probe_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An inconclusive namespaced probe warns and lets installation continue."""
        results = iter(
            (
                subprocess.CompletedProcess(
                    args=["kubectl", "get", "namespace", "--", "rl"],
                    returncode=1,
                    stdout="",
                    stderr="Error from server (Forbidden): namespaces is forbidden",
                ),
                subprocess.CompletedProcess(
                    args=["kubectl", "get", "serviceaccounts", "-n", "rl", "-o", "name"],
                    returncode=1,
                    stdout="",
                    stderr="Unable to connect to the server",
                ),
            )
        )

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            return next(results)

        monkeypatch.setattr(actions_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
        monkeypatch.setattr(actions_module.Kubectl, "run_raw", staticmethod(run_raw))

        with caplog.at_level(logging.WARNING, logger=actions_module.__name__):
            actions_module._ensure_namespace("rl")

        assert "nothing here says whether namespace rl exists" in caplog.text
        assert "Unable to connect to the server" in caplog.text
