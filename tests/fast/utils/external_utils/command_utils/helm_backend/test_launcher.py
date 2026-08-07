import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import launcher


def _infra_file(tmp_path: Path, name: str, values: dict[str, Any]) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(values))
    return str(path)


def _stub_helm_get_values(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(launcher.helm, "run", fake_run)
    return commands


class TestDumpValues:
    def test_orders_the_keys_so_a_relaunch_produces_the_same_file(self):
        """The file is compared against what helm already holds, and dict order is not part of the values."""
        dumped = launcher.dump_values({"run": {"id": "x", "inferenceEngines": [], "adhoc": False}})

        assert dumped.index("adhoc") < dumped.index("id") < dumped.index("inferenceEngines")

    def test_round_trips_through_yaml(self):
        """helm reads this file back, so anything yaml cannot express would be silently lost."""
        values = {"run": {"id": "260101-000000-000", "inferenceEngines": [{"name": "e", "replicas": 2}]}}

        assert yaml.safe_load(launcher.dump_values(values)) == values

    def test_writes_a_block_style_document(self):
        """A run's values are read by humans diagnosing a launch, and inline flow style is unreadable."""
        assert "{" not in launcher.dump_values({"run": {"orchestrator": {"command": ["python", "train.py"]}}})


class TestWriteValues:
    def test_creates_the_run_directory_it_writes_into(self, tmp_path):
        """The launcher writes here before anything else has created the run's directory."""
        path = tmp_path / "miles-runs" / "260101-000000-000" / "values.yaml"

        launcher.write_values(path, {"run": {"id": "260101-000000-000"}})

        assert yaml.safe_load(path.read_text()) == {"run": {"id": "260101-000000-000"}}


class TestMergedValues:
    def test_lets_the_run_win_over_the_cluster_defaults(self, tmp_path):
        """helm applies the run file last, so the check has to predict the same winner."""
        infra = _infra_file(tmp_path, "infra.yaml", {"run": {"id": "from-infra"}})

        merged = launcher.merged_values([infra], {"run": {"id": "from-run"}})

        assert merged["run"]["id"] == "from-run"

    def test_keeps_the_infra_keys_a_run_never_mentions(self, tmp_path):
        """The image and the storage claim live only in the infra values, and dropping them refuses every upgrade."""
        infra = _infra_file(tmp_path, "infra.yaml", {"infra": {"image": "miles:dev"}})

        merged = launcher.merged_values([infra], {"run": {"id": "x"}})

        assert merged == {"infra": {"image": "miles:dev"}, "run": {"id": "x"}}

    def test_merges_deeply_rather_than_replacing_a_section(self, tmp_path):
        """A run setting one key under run must not erase the infra defaults beside it."""
        infra = _infra_file(tmp_path, "infra.yaml", {"run": {"nodeSelector": {"pool": "gpu"}}})

        merged = launcher.merged_values([infra], {"run": {"id": "x"}})

        assert merged["run"] == {"nodeSelector": {"pool": "gpu"}, "id": "x"}

    def test_lets_a_later_infra_file_win_over_an_earlier_one(self, tmp_path):
        """helm applies the values files in order, and an operator overlay is passed last on purpose."""
        first = _infra_file(tmp_path, "a.yaml", {"infra": {"image": "miles:dev", "namespace": "rl"}})
        second = _infra_file(tmp_path, "b.yaml", {"infra": {"image": "miles:other"}})

        merged = launcher.merged_values([first, second], {})

        assert merged["infra"] == {"image": "miles:other", "namespace": "rl"}

    def test_tolerates_an_empty_infra_file(self, tmp_path):
        """A placeholder values file parses as None, which would otherwise abort every launch that uses it."""
        empty = _infra_file(tmp_path, "empty.yaml", {})
        Path(empty).write_text("")

        assert launcher.merged_values([empty], {"run": {"id": "x"}}) == {"run": {"id": "x"}}

    def test_uses_the_run_values_alone_when_there_is_no_infra_file(self):
        """A cluster whose defaults are all baked into the chart still has to launch."""
        assert launcher.merged_values([], {"run": {"id": "x"}}) == {"run": {"id": "x"}}


class TestInstalledValues:
    def test_reads_the_values_helm_recorded_for_the_release(self, monkeypatch):
        """The elastic check compares against what is installed, not against the chart's defaults."""
        _stub_helm_get_values(monkeypatch, '{"run": {"id": "x"}}')

        assert launcher.installed_values("miles-run-x", "rl") == {"run": {"id": "x"}}

    def test_asks_helm_about_the_release_in_its_own_namespace(self, monkeypatch):
        """A release name exists per namespace, so a missing namespace could read a different run."""
        commands = _stub_helm_get_values(monkeypatch, "null")

        launcher.installed_values("miles-run-x", "rl")

        assert commands[0][:4] == ["helm", "get", "values", "miles-run-x"]
        assert commands[0][commands[0].index("--namespace") + 1] == "rl"

    def test_treats_a_release_installed_without_values_as_empty(self, monkeypatch):
        """helm prints null there, and json null would crash every comparison the check makes."""
        _stub_helm_get_values(monkeypatch, "null")

        assert launcher.installed_values("miles-run-x", "rl") == {}

    def test_treats_no_output_at_all_as_empty(self, monkeypatch):
        """An empty stdout is not valid json, and it means the same thing as null."""
        _stub_helm_get_values(monkeypatch, "")

        assert launcher.installed_values("miles-run-x", "rl") == {}


def _request(namespace: str, run_id: str) -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args="--rollout-num-gpus 8",
        num_gpus_per_node=8,
        megatron_model_type="qwen3-4B",
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        config=ExecuteTrainConfig(cluster_backend="kubernetes", namespace=namespace, run_id=run_id),
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
    )


def _record_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, ci_run: bool) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.helm, "run", fake_run)
    monkeypatch.setattr(launcher.helm, "release_exists", lambda release, namespace: False)
    infra = _infra_file(
        tmp_path,
        "infra.yaml",
        {"infra": {"sharedStorage": {"mountPath": str(tmp_path / "cluster-storage")}, "paths": {"runsSubPath": ""}}},
    )

    launcher.launch(
        _request(namespace="rl", run_id="260101-000000-000"),
        specs=[],
        run_id="260101-000000-000",
        namespace="rl",
        infra_values_files=[infra],
        repo_base_dir="/repo",
        train_argv=["--rollout-num-gpus", "8"],
        ci_run=ci_run,
    )
    return commands


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

        assert listing == [launcher.helm.list_ci_releases_command("rl")]

    def test_a_normal_launch_cleans_up_nothing(self, monkeypatch, tmp_path):
        """Outside CI the neighbouring releases are other people's runs, and none of them is ours to delete."""
        verbs = [command[1] for command in _record_launch(monkeypatch, tmp_path, ci_run=False)]

        assert "list" not in verbs
        assert "uninstall" not in verbs

    def test_labels_the_release_it_installs_so_the_next_run_can_find_it(self, monkeypatch, tmp_path):
        """The cleanup selects on this label, and an unlabelled CI release is one nothing will ever remove."""
        commands = _record_launch(monkeypatch, tmp_path, ci_run=True)
        upgrade = [command for command in commands if command[1] == "upgrade"]

        assert upgrade[0][upgrade[0].index("--labels") + 1] == f"{launcher.helm.CI_LABEL}=true"
