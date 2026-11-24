from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from tests.fast.charts.utils import render_run, requires_helm, single_object_of_kind
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.utils import (
    LauncherArgs,
    launcher_args_orchestrator_command,
)
from tests.fast.utils.external_utils.command_utils.helm_backend.orchestrator.observer_fakes import _ObserverProbe
from tests.fast.utils.workers.fake_specs import FakeServeSpec

from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import command_job
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import (
    CI_JOB_ID_ENV_VAR,
    CI_JOB_ID_LABEL,
    CI_LABEL,
    Helm,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT_INFO, PortInfo, SchedulingSpec


class TestCiJobIdLabelling:
    def test_helm_upgrade_propagates_ci_job_id_without_enabling_namespace_wide_ci_label(self, monkeypatch):
        """Setting MILES_CI_JOB_ID labels only this job's releases and never opts into namespace-wide CI sweeping."""
        monkeypatch.setenv(CI_JOB_ID_ENV_VAR, "gh-12345-1-shard0")

        command = Helm.upgrade_command("r", "miles-ci-k8s", "/chart", ["/infra.yaml"], ci_run=False)

        assert command[command.index("--labels") + 1] == f"{CI_JOB_ID_LABEL}=gh-12345-1-shard0"
        assert CI_LABEL not in command[command.index("--labels") + 1]

    def test_helm_upgrade_combines_ci_run_and_ci_job_id_labels(self, monkeypatch):
        monkeypatch.setenv(CI_JOB_ID_ENV_VAR, "gh-99-2")

        command = Helm.upgrade_command("r", "miles-ci-k8s", "/chart", [], ci_run=True)

        assert command[command.index("--labels") + 1] == f"{CI_LABEL}=true,{CI_JOB_ID_LABEL}=gh-99-2"

    def test_invalid_ci_job_id_fails_loudly(self, monkeypatch):
        monkeypatch.setenv(CI_JOB_ID_ENV_VAR, "not a valid/label")

        with pytest.raises(AssertionError, match=CI_JOB_ID_ENV_VAR):
            command_wrapper.current_ci_job_id()

    def test_command_job_passes_ci_job_id_label_and_combined_deadline_to_helm_template(self, monkeypatch):
        monkeypatch.setenv(CI_JOB_ID_ENV_VAR, "gh-555-1")
        captured: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.append(command)
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="kind: Job\n", stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run)
        job = command_job._CommandJob(
            context=command_job.CommandJobContext(
                namespace="miles-ci-k8s",
                chart_dir=Path("charts/miles-run"),
                gpus_per_node=4,
                timeout_seconds=120.0,
                admission_timeout_seconds=300.0,
            ),
            step="convert-abc",
            completions=1,
            gpus_per_pod=4,
        )

        command_job._render_job(job, command=["bash", "-c", "convert"])

        template_cmd = next(cmd for cmd in captured if cmd[:2] == ["helm", "template"])
        assert "commandJob.activeDeadlineSeconds=420" in template_cmd
        assert f'commandJob.labels={json.dumps({CI_JOB_ID_LABEL: "gh-555-1"})}' in template_cmd

    @requires_helm
    def test_command_job_template_stamps_ci_job_id_on_service_job_and_pod_template(self):
        objects = render_run(
            "--show-only",
            "templates/command-job.yaml",
            "--set",
            "commandJob.enabled=true",
            "--set",
            "commandJob.name=convert",
            "--set",
            "commandJob.objectName=myrun-miles-run-convert",
            "--set-json",
            'commandJob.command=["bash","-c","convert"]',
            "--set-json",
            f'commandJob.labels={json.dumps({CI_JOB_ID_LABEL: "gh-555-1"})}',
        )
        service = single_object_of_kind(objects, "Service")
        job = single_object_of_kind(objects, "Job")

        assert service["metadata"]["labels"][CI_JOB_ID_LABEL] == "gh-555-1"
        assert job["metadata"]["labels"][CI_JOB_ID_LABEL] == "gh-555-1"
        assert job["spec"]["template"]["metadata"]["labels"][CI_JOB_ID_LABEL] == "gh-555-1"
        assert CI_JOB_ID_LABEL not in service["spec"]["selector"]


class TestTimeoutSeparation:
    def test_pending_admission_time_does_not_burn_execution_budget_in_wait_for_run(self, tmp_path, monkeypatch):
        """A run queued in Pending for 30s still gets its full 20s execution budget once Running."""
        path = tmp_path / "state.json"
        probe = _ObserverProbe(
            generations=[path] * 10,
            pods=[
                observer.ObservedPod(phase="Pending"),
                observer.ObservedPod(phase="Pending"),
                observer.ObservedPod(phase="Pending"),
                observer.ObservedPod(phase="Running"),
                observer.ObservedPod(phase="Running"),
                observer.ObservedPod(phase="Running"),
            ],
        )
        original_read_pod = probe.read_pod

        def read_pod_and_finish():
            pod = original_read_pod()
            if probe.pod_reads == 6:
                OrchestratorState(status=OrchestratorStatus.EXITED, exit_code=0).write(path)
            return pod

        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=read_pod_and_finish,
            read_active_state_file=probe.read_generation,
            admission_timeout_seconds=60.0,
            execution_timeout_seconds=30.0,
        )

        assert outcome.exit_code == 0
        assert outcome.started is True

    def test_wait_for_run_fails_clearly_on_admission_timeout(self, tmp_path, monkeypatch):
        path = tmp_path / "state.json"
        probe = _ObserverProbe(
            generations=[path] * 10,
            pods=[observer.ObservedPod(phase="Pending")] * 10,
        )
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=probe.read_pod,
            read_active_state_file=probe.read_generation,
            admission_timeout_seconds=20.0,
            execution_timeout_seconds=600.0,
        )

        assert outcome.exit_code == 1
        assert outcome.started is False
        assert "pending admission" in outcome.reason

    def test_wait_for_run_fails_clearly_on_execution_timeout(self, tmp_path, monkeypatch):
        path = tmp_path / "state.json"
        probe = _ObserverProbe(
            generations=[path] * 10,
            pods=[observer.ObservedPod(phase="Running")] * 10,
        )
        monkeypatch.setattr(observer, "time", SimpleNamespace(sleep=probe.sleep))

        outcome = observer.wait_for_run(
            state_file=path,
            read_pod=probe.read_pod,
            read_active_state_file=probe.read_generation,
            admission_timeout_seconds=600.0,
            execution_timeout_seconds=20.0,
        )

        assert outcome.exit_code == 1
        assert outcome.started is True
        assert "execution timeout" in outcome.reason

    def test_command_job_pending_admission_does_not_burn_execution_timeout(self, monkeypatch):
        pod_phases = ["Pending", "Pending", "Pending", "Running", "Running"]
        job_statuses = ["running", "running", "running", "running", "running", "complete"]

        def fake_run_process(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            args = argv[1:]
            if args[:2] == ["get", "pods"]:
                phase = pod_phases.pop(0) if pod_phases else "Running"
                payload = json.dumps({"items": [{"metadata": {"name": "c-0", "uid": "u-0"}, "status": {"phase": phase}}]})
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=payload, stderr="")
            if args[:2] == ["get", "job"]:
                status = job_statuses.pop(0) if job_statuses else "complete"
                body = (
                    '{"status": {"conditions": [{"type": "Complete", "status": "True"}]}}'
                    if status == "complete"
                    else '{"status": {}}'
                )
                return subprocess.CompletedProcess(args=argv, returncode=0, stdout=body, stderr="")
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run_process)
        monkeypatch.setattr(command_job.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(command_job, "with_observability", lambda **kwargs: contextlib.nullcontext())

        job = command_job._CommandJob(
            context=command_job.CommandJobContext(
                namespace="miles-ci-k8s",
                chart_dir=Path("charts/miles-run"),
                gpus_per_node=4,
                timeout_seconds=10.0,
                admission_timeout_seconds=20.0,
            ),
            step="convert",
            completions=1,
            gpus_per_pod=4,
        )

        assert command_job._wait(job) == "complete"


class TestCiGateRecordDirForwardingAndSplitPreservation:
    def test_forwards_ci_gate_record_dir_to_run_env_and_preserves_non_orchestrator_split_releases(
        self, monkeypatch, tmp_path
    ):
        """MILES_CI_GATE_RECORD_DIR reaches the orchestrator pod, and split trainer deploys are never uninstalled early."""
        record_dir = str(tmp_path / "ci-gate-attempt-0")
        monkeypatch.setenv(RECORD_DIR_ENV, record_dir)
        monkeypatch.setenv(CI_JOB_ID_ENV_VAR, "gh-777-1")

        infra_file = tmp_path / "infra.yaml"
        infra_file.write_text(
            yaml.safe_dump(
                {
                    "infra": {
                        "image": {"repository": "myregistry.example/miles", "tag": "v1"},
                        "volumes": [
                            {
                                "name": "cluster-storage",
                                "hostPath": {"path": f"{tmp_path}/cluster-storage"},
                                "mounts": [{"mountPath": f"{tmp_path}/cluster-storage"}],
                            }
                        ],
                        "paths": {"runsRoot": f"{tmp_path}/cluster-storage/miles_data"},
                    }
                }
            )
        )
        issued: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            issued.append([str(part) for part in command])
            stdout = '{"manifest": ""}' if "--dry-run" in command else ""
            return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run)
        monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
        trainer_spec = FakeServeSpec(
            name="trainer-engine-actor",
            category=POOL_CATEGORY_TRAINER_ENGINE,
            port_infos=[PortInfo(name="master", static_port=9000, mode="master"), DEFAULT_RPC_PORT_INFO],
            env_vars=lambda ctx: {},
            fixed_scheduling=SchedulingSpec(
                num_cells=1,
                num_workers_per_cell=8,
                num_gpus_per_worker=0.4,
                num_gpu_slots_per_worker=1,
                num_gpus_per_node=8,
            ),
            args=BaseLeafConfig(),
            worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        )
        monkeypatch.setattr(entrypoint, "compute_specs", lambda args: [trainer_spec])
        monkeypatch.setattr(
            entrypoint,
            "parse_args",
            lambda: LauncherArgs(deploy_component=DeployComponent.TRAINER.value),
        )
        monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
        monkeypatch.setattr(entrypoint, "_compute_orchestrator_command", launcher_args_orchestrator_command)
        monkeypatch.setattr(entrypoint, "_compute_trainer_controller_addrs", lambda args, **kwargs: {})

        entrypoint.execute_train(
            request=ExecuteTrainRequest(
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
            ),
            config=ExecuteTrainConfig(
                namespace="miles-ci-k8s",
                run_id="260101-000000-001",
                run_uuid="0123456789abcdef",
                deploy_component=DeployComponent.TRAINER,
                helm_values=(str(infra_file),),
            ),
        )

        written_files = sorted(
            (tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / "260101-000000-001" / "values").glob(
                "values-*.yaml"
            )
        )
        written = yaml.safe_load(written_files[0].read_text())
        assert written["run"]["env"][RECORD_DIR_ENV] == record_dir
        upgrade_cmds = [cmd for cmd in issued if cmd[:3] == ["helm", "upgrade", "--install"] and "--dry-run" not in cmd]
        assert len(upgrade_cmds) == 1
        assert f"{CI_JOB_ID_LABEL}=gh-777-1" in upgrade_cmds[0]
        assert not any(cmd[:2] == ["helm", "uninstall"] for cmd in issued)
