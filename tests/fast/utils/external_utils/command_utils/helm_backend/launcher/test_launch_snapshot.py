import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot, sanitize

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "helm_backend"

FROZEN_RUN_ID = "260101-000000-000"
FROZEN_LAUNCH_TOKEN = "260101-000000-000001"
NAMESPACE = "rl"
PYTHON_PLACEHOLDER = "<PYTHON>"


def _router() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=30000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: (
            f"python -m sglang_router.launch_router --host {ctx.self_addrs['primary'].host} --port 30000"
        ),
    )


def _engine() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-engine-0-0",
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
        ],
        env_var=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        scheduling=SchedulingSpec(
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=8,
            num_gpus_per_node=8,
        ),
        launch_command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port}"
        ),
    )


def _trainer() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        scheduling=SchedulingSpec(
            num_cells=2,
            num_workers_per_cell=8,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def _request(
    *,
    train_args: str = "--rollout-num-gpus 8",
    extra_env_vars: dict[str, str] | None = None,
    extra_manifests: list[str] | None = None,
):
    return ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars=extra_env_vars or {},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=extra_manifests or [],
    )


def helm_values_file(sandbox: Path) -> Path:
    values_file = sandbox / "infra.yaml"
    values_file.write_text(
        yaml.safe_dump(
            {
                "infra": {
                    "image": {"repository": "myregistry.example/miles", "tag": "v1"},
                    "sharedStorage": {
                        "type": "hostPath",
                        "hostPath": f"{sandbox}/cluster-storage",
                        "mountPath": f"{sandbox}/cluster-storage",
                    },
                    "paths": {"runsSubPath": "miles_data"},
                }
            }
        )
    )
    return values_file


def run_dir(sandbox: Path) -> Path:
    return sandbox / "cluster-storage" / "miles_data" / "miles-runs" / FROZEN_RUN_ID


def values_file(sandbox: Path) -> Path:
    return run_dir(sandbox) / "values" / f"values-{FROZEN_LAUNCH_TOKEN}.yaml"


def record_launch(monkeypatch, sandbox: Path, **request_overrides) -> list[str]:
    recorded: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        recorded.append(" ".join(str(part) for part in command))
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    monkeypatch.setattr(entrypoint, "repo_base_dir", str(REPO_ROOT))
    monkeypatch.setattr(naming, "_new_launch_token", lambda: FROZEN_LAUNCH_TOKEN)

    _stub_launch_inputs(monkeypatch, specs=[_router(), _engine(), _trainer()])

    entrypoint.execute_train(
        request=_request(**request_overrides),
        config=ExecuteTrainConfig(
            namespace=NAMESPACE, run_id=FROZEN_RUN_ID, helm_values=(str(helm_values_file(sandbox)),)
        ),
    )
    return recorded


def freeze(text: str, sandbox: Path) -> str:
    return sanitize(text.replace(sys.executable, PYTHON_PLACEHOLDER), sandbox=sandbox)


def format_launch(commands: list[str], values_text: str, sandbox: Path) -> str:
    lines: list[str] = []
    for index, command in enumerate(commands):
        lines.append(f"### {index}")
        lines.append(freeze(command, sandbox=sandbox))
        lines.append("")
    lines.append("### pseudo file 1")
    lines.append(freeze(values_text, sandbox=sandbox))
    lines.append("")
    return "\n".join(lines)


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False) -> None:
    monkeypatch.setattr(entrypoint, "compute_specs", lambda args: specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(
            colocate=colocate, deploy_component="all", argv=[], use_wandb=False, wandb_run_id=None
        ),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestTheLauncherRecordsWhatItLaunched:
    def test_the_record_lands_beside_the_values_it_describes(self, monkeypatch, tmp_path):
        """The pods' argv is computed here, so this is the only place it can be recorded from."""
        record_launch(monkeypatch, tmp_path)

        records = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
        assert len(records) == 1
        recorded = json.loads(records[0].read_text())

        assert recorded["run_id"] == FROZEN_RUN_ID
        assert recorded["namespace"] == NAMESPACE
        assert recorded["orchestrator_command"][1] == "/repo/train.py"
        assert recorded["values_file"] == str(values_file(tmp_path))

    def test_the_pods_are_pointed_at_that_record(self, monkeypatch, tmp_path):
        """A record on a shared disk nobody reads is not what reaches the wandb config."""
        record_launch(monkeypatch, tmp_path)
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        records = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
        assert rendered["run"]["launchRecord"] == str(records[0])

    def test_the_pods_are_handed_a_path_rather_than_the_record_itself(self, monkeypatch, tmp_path):
        """A whole environment dump in an env var is passed to every process of every pod."""
        record_launch(monkeypatch, tmp_path)
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        assert rendered["run"]["launchRecord"].startswith("/")
        assert "orchestrator_command" not in rendered["run"]["launchRecord"]

    def test_the_record_carries_no_secret_the_launcher_was_given(self, monkeypatch, tmp_path):
        """The record lands on a shared disk and in the wandb config, so a key in argv would leak twice."""
        record_launch(
            monkeypatch,
            tmp_path,
            train_args="--rollout-num-gpus 8 --wandb-key s3cret",
            extra_env_vars={"HF_TOKEN": "t0ken", "PYTHONUNBUFFERED": "1"},
        )
        on_disk = sorted((run_dir(tmp_path) / "launches").glob("*.json"))[0].read_text()

        assert "s3cret" not in on_disk
        assert "t0ken" not in on_disk
        assert "redacted-sha256:" in on_disk

    def test_the_pods_still_receive_the_credentials_they_run_with(self, monkeypatch, tmp_path):
        """Only the audit copy is redacted; redacting the pods' own environment would break the run."""
        record_launch(monkeypatch, tmp_path, extra_env_vars={"HF_TOKEN": "t0ken"})
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        assert rendered["run"]["env"]["HF_TOKEN"] == "t0ken"

    def test_the_whole_record_is_on_disk_before_helm_installs_anything(self, monkeypatch, tmp_path):
        """helm upgrade is irreversible; recording afterwards can leave running pods with no record."""
        helm_run_snapshot: list[dict] = []

        def note_upgrade(**kwargs) -> None:
            written = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
            assert len(written) == 1
            helm_run_snapshot.append(json.loads(written[0].read_text()))

        monkeypatch.setattr(Helm, "upgrade", staticmethod(note_upgrade))
        record_launch(monkeypatch, tmp_path)

        assert len(helm_run_snapshot) == 1
        assert helm_run_snapshot[0]["values_file"] == str(values_file(tmp_path))


class TestTheLauncherOwnsTheReport:
    def test_a_run_may_not_name_its_own_env_report(self, monkeypatch, tmp_path):
        """--env-report outranks the variable the chart sets, so the pods would report a launch that never ran."""
        with pytest.raises(AssertionError, match="--env-report"):
            record_launch(monkeypatch, tmp_path, train_args="--rollout-num-gpus 8 --env-report {}")

    def test_an_inline_env_report_is_refused_too(self, monkeypatch, tmp_path):
        with pytest.raises(AssertionError, match="--env-report"):
            record_launch(monkeypatch, tmp_path, train_args="--rollout-num-gpus 8 --env-report={}")


class TestKubernetesLaunchSnapshot:
    def test_the_helm_argv_and_the_generated_values_match_the_recording(self, monkeypatch, tmp_path):
        """The values file is the whole training recipe, so a snapshot of only the argv would prove little."""
        commands = record_launch(monkeypatch, tmp_path)

        recorded = format_launch(commands, values_file(tmp_path).read_text(), tmp_path)

        assert_matches_snapshot(SNAPSHOT_DIR / "kubernetes_launch.txt", recorded, "kubernetes launcher recording")

    def test_the_generated_values_carry_no_infra_section(self, monkeypatch, tmp_path):
        """infra is the user's half of the contract; the launcher writing it would silently override a cluster."""
        record_launch(monkeypatch, tmp_path)

        assert set(yaml.safe_load(values_file(tmp_path).read_text())) == {"run"}


class TestTheLauncherPassesExtraManifestsToTheChart:
    def test_the_manifests_the_caller_named_reach_the_values_file_verbatim(self, monkeypatch, tmp_path):
        """The chart installs this text as it stands, so the launcher may not reformat it on the way."""
        manifests = [
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: external-sglang\n",
            "apiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: external-sglang\n",
        ]
        record_launch(monkeypatch, tmp_path, extra_manifests=manifests)

        assert yaml.safe_load(values_file(tmp_path).read_text())["extraManifests"] == manifests

    def test_a_launch_that_named_none_writes_no_such_section(self, monkeypatch, tmp_path):
        """An empty section in the values would still override whatever a user's own values file set."""
        record_launch(monkeypatch, tmp_path)

        assert "extraManifests" not in yaml.safe_load(values_file(tmp_path).read_text())


class TestSnapshotFiles:
    def test_the_recorded_files_are_exactly_the_declared_ones(self):
        """A renamed or deleted case must not leave an orphan golden behind."""
        recorded = {path.name for path in SNAPSHOT_DIR.glob("*.txt")}

        assert recorded == {"kubernetes_launch.txt"}
