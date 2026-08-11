import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _request() -> ExecuteTrainRequest:
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


def record_launch(monkeypatch, sandbox: Path) -> list[str]:
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
        request=_request(),
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
        lambda: SimpleNamespace(colocate=colocate, argv=[], use_wandb=False, wandb_run_id=None),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestKubernetesLaunchSnapshot:
    def test_the_helm_argv_and_the_generated_values_match_the_recording(self, monkeypatch, tmp_path):
        """The values file is the whole training recipe, so a snapshot of only the argv would prove little."""
        commands = record_launch(monkeypatch, tmp_path)
        values_file = (
            tmp_path
            / "cluster-storage"
            / "miles_data"
            / "miles-runs"
            / FROZEN_RUN_ID
            / "values"
            / f"values-{FROZEN_LAUNCH_TOKEN}.yaml"
        )

        recorded = format_launch(commands, values_file.read_text(), tmp_path)

        assert_matches_snapshot(SNAPSHOT_DIR / "kubernetes_launch.txt", recorded, "kubernetes launcher recording")

    def test_the_generated_values_carry_no_infra_section(self, monkeypatch, tmp_path):
        """infra is the user's half of the contract; the launcher writing it would silently override a cluster."""
        record_launch(monkeypatch, tmp_path)
        values_file = (
            tmp_path
            / "cluster-storage"
            / "miles_data"
            / "miles-runs"
            / FROZEN_RUN_ID
            / "values"
            / f"values-{FROZEN_LAUNCH_TOKEN}.yaml"
        )

        assert set(yaml.safe_load(values_file.read_text())) == {"run"}


class TestSnapshotFiles:
    def test_the_recorded_files_are_exactly_the_declared_ones(self):
        """A renamed or deleted case must not leave an orphan golden behind."""
        recorded = {path.name for path in SNAPSHOT_DIR.glob("*.txt")}

        assert recorded == {"kubernetes_launch.txt"}
