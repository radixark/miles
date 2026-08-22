import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo


def _infra_file(tmp_path: Path, name: str, values: dict[str, Any]) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(values))
    return str(path)


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False) -> None:
    monkeypatch.setattr(entrypoint, "compute_specs", lambda args: specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(colocate=colocate, deploy_component="all", deploy_instance_id=None, argv=[]),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestWriteHelmValues:
    def test_orders_the_keys_so_a_relaunch_produces_the_same_file(self, tmp_path):
        """The file is compared against what helm already holds, and dict order is not part of the values."""
        path = tmp_path / "values.yaml"

        entrypoint._write_helm_values(path, {"run": {"id": "x", "inferenceEngines": [], "commandJob": False}})

        dumped = path.read_text()
        assert dumped.index("commandJob") < dumped.index("id") < dumped.index("inferenceEngines")

    def test_round_trips_through_yaml(self, tmp_path):
        """helm reads this file back, so anything yaml cannot express would be silently lost."""
        values = {"run": {"id": "260101-000000-000", "inferenceEngines": [{"name": "e", "replicas": 2}]}}
        path = tmp_path / "values.yaml"

        entrypoint._write_helm_values(path, values)

        assert yaml.safe_load(path.read_text()) == values

    def test_writes_a_block_style_document(self, tmp_path):
        """A run's values are read by humans diagnosing a launch, and inline flow style is unreadable."""
        path = tmp_path / "values.yaml"

        entrypoint._write_helm_values(path, {"run": {"orchestrator": {"command": ["python", "train.py"]}}})

        assert "{" not in path.read_text()

    def test_creates_the_run_directory_it_writes_into(self, tmp_path):
        """The launcher writes here before anything else has created the run's directory."""
        path = tmp_path / "miles-runs" / "260101-000000-000" / "values.yaml"

        entrypoint._write_helm_values(path, {"run": {"id": "260101-000000-000"}})

        assert yaml.safe_load(path.read_text()) == {"run": {"id": "260101-000000-000"}}


def _request(namespace: str, run_id: str) -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args="--rollout-num-gpus 8",
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
    )


def _record_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, ci_run: bool) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        argv = [str(part) for part in command]
        commands.append(argv)
        rendered = '{"manifest": ""}' if "--dry-run" in argv else ""
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=rendered, stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    monkeypatch.setattr(Helm, "get_manifest", lambda release, namespace: None)
    infra = _infra_file(
        tmp_path,
        "infra.yaml",
        {
            "infra": {
                "volumes": [
                    {
                        "name": "cluster-storage",
                        "hostPath": {"path": str(tmp_path / "cluster-storage")},
                        "mounts": [{"mountPath": str(tmp_path / "cluster-storage")}],
                    }
                ],
                "paths": {"runsRoot": str(tmp_path / "cluster-storage")},
            }
        },
    )

    _stub_launch_inputs(monkeypatch, specs=[])

    entrypoint.execute_train(
        request=_request(namespace="rl", run_id="260101-000000-000"),
        config=ExecuteTrainConfig(namespace="rl", run_id="260101-000000-000", helm_values=(infra,), ci_run=ci_run),
    )
    return [command for command in commands if command[0] == "helm"]


class TestCiRunCleanup:
    def test_removes_the_leftover_ci_releases_before_installing_this_one(self, monkeypatch, tmp_path):
        """A runner reuses one namespace, and installing first means the cleanup could uninstall the new run."""
        commands = _record_launch(monkeypatch, tmp_path, ci_run=True)
        verbs = [command[1] for command in commands]

        assert "list" in verbs
        assert verbs.index("list") < verbs.index("upgrade")

    def test_asks_only_for_the_ci_releases_of_this_namespace(self, monkeypatch, tmp_path):
        """The same namespace holds human runs, and a broader listing would uninstall someone's experiment."""
        listing = [command for command in _record_launch(monkeypatch, tmp_path, ci_run=True) if command[1] == "list"]

        assert listing[0][:2] == ["helm", "list"]
        assert listing[0][listing[0].index("--namespace") + 1] == "rl"

    def test_a_normal_launch_cleans_up_nothing(self, monkeypatch, tmp_path):
        """Outside CI the neighbouring releases are other people's runs, and none of them is ours to delete."""
        verbs = [command[1] for command in _record_launch(monkeypatch, tmp_path, ci_run=False)]

        assert "list" not in verbs
        assert "uninstall" not in verbs

    def test_labels_the_release_it_installs_so_the_next_run_can_find_it(self, monkeypatch, tmp_path):
        """The cleanup selects on this label, and an unlabelled CI release is one nothing will ever remove."""
        commands = _record_launch(monkeypatch, tmp_path, ci_run=True)
        installed = [command for command in commands if command[1] == "upgrade" and "--dry-run" not in command]

        assert installed[0][installed[0].index("--labels") + 1] == f"{command_wrapper.CI_LABEL}=true"
