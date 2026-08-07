import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import helm, launcher, run_state
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

RUN_ID = "260101-000000-000"
NAMESPACE = "myns"


def _engine(
    *,
    num_cells: int,
    gpus_per_worker_slot: int,
    workers_per_cell: int,
    colocate_with_trainer: bool = True,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-engine-0-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=workers_per_cell,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=gpus_per_worker_slot,
            num_gpus_per_node=8,
            colocate_with_trainer=colocate_with_trainer,
        ),
        launch_command=lambda ctx: "python -m sglang.launch_server",
    )


def _trainer(*, num_cells: int, workers_per_cell: int) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-actor",
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=workers_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def _request() -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args="",
        num_gpus_per_node=8,
        megatron_model_type="qwen3-4B",
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        config=ExecuteTrainConfig(cluster_backend="kubernetes", namespace=NAMESPACE, run_id=RUN_ID),
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
    )


def _infra_file(sandbox: Path) -> str:
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
    return str(values_file)


def _stub_helm(monkeypatch, *, failing_command: str | None = None) -> list[list[str]]:
    issued: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool = False) -> Any:
        issued.append([str(part) for part in command])
        if failing_command is not None and failing_command in " ".join(str(part) for part in command):
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(helm, "run", fake_run)
    monkeypatch.setattr(helm, "release_exists", lambda release, namespace: False)
    return issued


def _launch(monkeypatch, sandbox: Path, *, colocate: bool, specs: list[Any], failing_command: str | None = None):
    _stub_helm(monkeypatch, failing_command=failing_command)
    return launcher.launch(
        _request(),
        specs=specs,
        run_id=RUN_ID,
        namespace=NAMESPACE,
        infra_values_files=[_infra_file(sandbox)],
        repo_base_dir="/repo",
        train_argv=[],
        colocate=colocate,
    )


def _written_values(sandbox: Path) -> dict[str, Any]:
    path = sandbox / "cluster-storage" / "miles_data" / "miles-runs" / RUN_ID / "values.yaml"
    return yaml.safe_load(path.read_text())


class TestColocateReachesTheValues:
    def test_a_colocated_run_writes_the_colocate_section(self, monkeypatch, tmp_path):
        """Everything colocate does hangs off this section, so a launcher that omits it disables the feature."""
        specs = [
            _engine(num_cells=4, gpus_per_worker_slot=8, workers_per_cell=1),
            _trainer(num_cells=2, workers_per_cell=16),
        ]

        _launch(monkeypatch, tmp_path, colocate=True, specs=specs)

        colocate = _written_values(tmp_path)["run"]["colocate"]
        assert colocate == {
            "enabled": True,
            "enginePool": "inference-engine-0-0",
            "trainerPool": "trainer-actor",
        }

    def test_a_plain_run_writes_no_colocate_section(self, monkeypatch, tmp_path):
        """A disaggregated run must not gain a pairing controller with pod write rights."""
        specs = [
            _engine(num_cells=4, gpus_per_worker_slot=8, workers_per_cell=1),
            _trainer(num_cells=2, workers_per_cell=16),
        ]

        _launch(monkeypatch, tmp_path, colocate=False, specs=specs)

        assert "colocate" not in _written_values(tmp_path)["run"]

    def test_an_unsupported_layout_stops_the_launch(self, monkeypatch, tmp_path):
        """An engine cell wider than its trainer cell has no rank mapping, so it must fail before installing."""
        specs = [
            _engine(num_cells=1, gpus_per_worker_slot=8, workers_per_cell=4),
            _trainer(num_cells=1, workers_per_cell=8),
        ]

        with pytest.raises(AssertionError, match="colocate"):
            _launch(monkeypatch, tmp_path, colocate=True, specs=specs)


class TestFailedUpgradeKeepsThePreviousVerdict:
    def test_a_refused_upgrade_leaves_the_old_exit_code_in_place(self, monkeypatch, tmp_path):
        """The reset runs after the upgrade precisely so a failed relaunch cannot erase the last run's outcome."""
        exit_file = tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / RUN_ID / "state" / "orchestrator.exit"
        run_state.write_orchestrator_state(exit_file, run_state.STATUS_EXITED, exit_code=7, generation=3)
        specs = [_trainer(num_cells=1, workers_per_cell=8)]

        with pytest.raises(subprocess.CalledProcessError):
            _launch(monkeypatch, tmp_path, colocate=False, specs=specs, failing_command="helm upgrade")

        state = run_state.read_orchestrator_state(exit_file)
        assert (state.status, state.exit_code, state.generation) == (run_state.STATUS_EXITED, 7, 3)
