from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from miles.utils.external_utils.miles_workbench import __main__ as workbench
from miles.utils.external_utils.miles_workbench import actions


class TestKubernetesObjectNames:
    @pytest.mark.parametrize(
        "empty_option",
        [
            ["-n", "", "--release", "workbench"],
            ["--namespace", "training", "--release", ""],
        ],
    )
    def test_an_empty_kubernetes_object_name_is_rejected_before_cluster_access(
        self, monkeypatch: pytest.MonkeyPatch, empty_option: list[str]
    ) -> None:
        """An explicitly empty namespace or release is a usage error before cluster access."""

        def fail_if_process_runs(*args: object, **kwargs: object) -> NoReturn:
            raise AssertionError("cluster process boundary was reached")

        def fake_which(binary: str) -> str:
            return f"/usr/bin/{binary}"

        monkeypatch.setattr(actions.shutil, "which", fake_which)
        monkeypatch.setattr(actions, "run_process", fail_if_process_runs)

        result = CliRunner().invoke(workbench.app, ["uninstall", *empty_option])

        assert result.exit_code == 2
        assert "Invalid value" in result.output


class TestStop:
    def test_stop_forwards_the_deployment_instance_id_to_the_release_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Stopping one deployment instance uninstalls exactly its instance-specific release."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calls_path = tmp_path / "calls.log"
        helm_path = bin_dir / "helm"
        helm_path.write_text(f'#!/usr/bin/env bash\necho "helm $@" >> {calls_path}\n')
        helm_path.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

        result = CliRunner().invoke(
            app=workbench.app,
            args=[
                "stop",
                "--namespace",
                "rl",
                "demo",
                "--deploy-component",
                "inference",
                "--deploy-instance-id",
                "east",
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls_path.read_text().splitlines() == ["helm uninstall miles-run-demo-inference-east --namespace rl"]
