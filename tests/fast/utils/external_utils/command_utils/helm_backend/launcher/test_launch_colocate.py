import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pydantic
import pytest
import yaml

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import state as orchestrator_state
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

RUN_ID = "260101-000000-000"
NAMESPACE = "myns"
RELEASE = ReleaseName(run_id=RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()


def _engine(
    *,
    num_cells: int,
    gpus_per_worker_slot: int,
    workers_per_cell: int,
    name: str = "inference-engine-0-0",
    gpu_offset: int = 0,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=workers_per_cell,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=gpus_per_worker_slot,
            num_gpus_per_node=8,
            pg_slot_offset=gpu_offset,
        ),
        launch_command=lambda ctx: "python -m sglang.launch_server",
    )


def _trainer(*, num_cells: int, workers_per_cell: int) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
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
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
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

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        issued.append([str(part) for part in command])
        if "--dry-run" in command:
            return subprocess.CompletedProcess(args=command, returncode=0, stdout='{"manifest": ""}', stderr="")
        if failing_command is not None and failing_command in " ".join(str(part) for part in command):
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    return issued


def _launch(monkeypatch, sandbox: Path, *, colocate: bool, specs: list[Any], failing_command: str | None = None):
    _stub_helm(monkeypatch, failing_command=failing_command)
    _stub_launch_inputs(monkeypatch, specs=specs, colocate=colocate)

    return entrypoint.execute_train(
        request=_request(),
        config=ExecuteTrainConfig(namespace=NAMESPACE, run_id=RUN_ID, helm_values=(_infra_file(sandbox),)),
    )


def _written_values(sandbox: Path) -> dict[str, Any]:
    written = sorted(
        (sandbox / "cluster-storage" / "miles_data" / "miles-runs" / RUN_ID / "values").glob("values-*.yaml")
    )
    assert len(written) == 1, written
    return yaml.safe_load(written[0].read_text())


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False) -> None:
    monkeypatch.setattr(entrypoint, "compute_specs", lambda args: specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(
            colocate=colocate,
            deploy_component="all",
            deploy_instance_id=None,
            argv=[],
            use_wandb=False,
            wandb_run_id=None,
        ),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestColocateReachesTheValues:
    def test_a_colocated_run_writes_the_colocate_section(self, monkeypatch, tmp_path):
        """Everything colocate does hangs off this section, so a launcher that omits it disables the feature."""
        specs = [
            _engine(num_cells=4, gpus_per_worker_slot=8, workers_per_cell=1),
            _trainer(num_cells=2, workers_per_cell=16),
        ]

        _launch(monkeypatch, tmp_path, colocate=True, specs=specs)

        colocate = _written_values(tmp_path)["run"]["colocate"]

        assert colocate["namespace"] == NAMESPACE
        assert colocate["release"] == RELEASE
        assert colocate["trainer_pool_id"] == "trainer-engine-actor"
        assert [pool["pool_id"] for pool in colocate["inference_pools"]] == ["inference-engine-0-0"]

    def test_a_disaggregated_colocated_run_writes_every_pool_with_its_own_offset(self, monkeypatch, tmp_path):
        """Prefill and decode both sit on the trainer's gpus, and only their offsets say which nodes."""
        specs = [
            _engine(num_cells=2, gpus_per_worker_slot=8, workers_per_cell=1, name="inference-engine-0-0"),
            _engine(
                num_cells=2,
                gpus_per_worker_slot=8,
                workers_per_cell=1,
                name="inference-engine-0-1",
                gpu_offset=16,
            ),
            _trainer(num_cells=2, workers_per_cell=16),
        ]

        _launch(monkeypatch, tmp_path, colocate=True, specs=specs)

        pools = _written_values(tmp_path)["run"]["colocate"]["inference_pools"]

        assert [(pool["pool_id"], pool["layout"]["gpu_offset"]) for pool in pools] == [
            ("inference-engine-0-0", 0),
            ("inference-engine-0-1", 16),
        ]

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

        with pytest.raises(pydantic.ValidationError, match="colocate"):
            _launch(monkeypatch, tmp_path, colocate=True, specs=specs)


class TestFailedUpgradeKeepsThePreviousVerdict:
    def test_a_refused_upgrade_leaves_the_old_exit_code_in_place(self, monkeypatch, tmp_path):
        """A launch writes no state of its own, so a relaunch that never installed cannot erase a verdict."""
        state_dir = tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / RUN_ID / "state"
        state_file = state_dir / "orchestrator-260101-000000-000001.state"
        orchestrator_state.OrchestratorState(status=orchestrator_state.OrchestratorStatus.EXITED, exit_code=7).write(
            state_file
        )
        specs = [_trainer(num_cells=1, workers_per_cell=8)]

        with pytest.raises(subprocess.CalledProcessError):
            _launch(monkeypatch, tmp_path, colocate=False, specs=specs, failing_command="helm upgrade")

        state = orchestrator_state.OrchestratorState.read(state_file)
        assert (state.status, state.exit_code) == (orchestrator_state.OrchestratorStatus.EXITED, 7)
