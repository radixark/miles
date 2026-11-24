from __future__ import annotations

import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    named_object,
    render_run,
    requires_helm,
    single_object_of_kind,
    with_object_names,
)

_GPU_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"add": ["SYS_PTRACE", "SYS_NICE", "IPC_LOCK"]},
}

_CPU_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
}

_ENGINES = [
    {
        "name": "engine",
        "replicas": 1,
        "size": 1,
        "command": ["python", "-m", "sglang.launch_server"],
        "resources": {"limits": {"nvidia.com/gpu": 4}},
        "needsGpuRuntime": True,
    }
]

_TRAINERS = [
    {
        "name": "trainer",
        "replicas": 1,
        "size": 1,
        "command": ["python", "-m", "miles.utils.workers.serving.serve"],
        "resources": {"limits": {"nvidia.com/gpu": 4}},
        "needsGpuRuntime": True,
    }
]

_STATIC_WORKERS = [
    {
        "name": "router",
        "replicas": 1,
        "command": ["python", "-m", "miles.router"],
        "ports": [{"name": "primary", "port": 8000}],
    }
]

_COLOCATE = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "num_gpus_per_inference_pod": 4,
                "gpu_offset": 0,
            },
        }
    ],
}

_RDMA_FLAG = ("--set-json", 'infra.gpuResources={"rdma/hca_shared_devices_all":1}')


def _lws_pod_spec(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return named_object(objects, "LeaderWorkerSet", name)["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _shm_volume(pod_spec: dict[str, Any]) -> dict[str, Any] | None:
    return next((vol for vol in pod_spec.get("volumes", []) if vol["name"] == "dev-shm"), None)


def _shm_mount(container: dict[str, Any]) -> dict[str, Any] | None:
    return next((mount for mount in container.get("volumeMounts", []) if mount["mountPath"] == "/dev/shm"), None)


@requires_helm
class TestFivePodRoleRendering:
    def test_colocated_trainer_and_zero_gpu_paired_inference_both_receive_rdma_and_gpu_capabilities(self):
        """Zeroing nvidia.com/gpu on a gated inference pod must not drop its RDMA device or IPC_LOCK capability."""
        objects = render_run(
            *_RDMA_FLAG,
            "--set-json",
            f"run.inferenceEngines={json.dumps(with_object_names(_ENGINES))}",
            "--set-json",
            f"run.trainerEngines={json.dumps(with_object_names(_TRAINERS))}",
            "--set-json",
            f"run.colocate={json.dumps(_COLOCATE)}",
        )
        trainer_container = _lws_pod_spec(objects, f"{RUN_RELEASE_NAME}-miles-run-trainer")["containers"][0]
        inference_container = _lws_pod_spec(objects, f"{RUN_RELEASE_NAME}-miles-run-engine")["containers"][0]

        assert trainer_container["securityContext"] == _GPU_SECURITY_CONTEXT
        assert inference_container["securityContext"] == _GPU_SECURITY_CONTEXT
        assert trainer_container["resources"] == {
            "limits": {"nvidia.com/gpu": 4, "rdma/hca_shared_devices_all": 1},
            "requests": {"rdma/hca_shared_devices_all": 1},
        }
        assert inference_container["resources"] == {
            "limits": {"nvidia.com/gpu": 0, "rdma/hca_shared_devices_all": 1},
            "requests": {"rdma/hca_shared_devices_all": 1},
        }

    def test_disaggregated_gpu_worker_uses_default_dev_shm_while_colocated_pair_uses_colocate_dev_shm(self):
        """Non-colocate GPU pods can stay on Memory emptyDir while colocate pairs use a per-release hostPath."""
        common_args = (
            "--set",
            "infra.devShm.hostPath=null",
            "--set-json",
            'infra.devShm.emptyDir={"medium":"Memory","sizeLimit":"32Gi"}',
            "--set-json",
            'infra.devShm.colocateHostPath={"path":"/dev/shm/miles-${RELEASE}","type":"DirectoryOrCreate"}',
            "--set-json",
            f"run.inferenceEngines={json.dumps(with_object_names(_ENGINES))}",
            "--set-json",
            f"run.trainerEngines={json.dumps(with_object_names(_TRAINERS))}",
        )
        disaggregated_pod = _lws_pod_spec(render_run(*common_args), f"{RUN_RELEASE_NAME}-miles-run-engine")
        colocated_objects = render_run(*common_args, "--set-json", f"run.colocate={json.dumps(_COLOCATE)}")
        colocated_trainer_pod = _lws_pod_spec(colocated_objects, f"{RUN_RELEASE_NAME}-miles-run-trainer")
        colocated_engine_pod = _lws_pod_spec(colocated_objects, f"{RUN_RELEASE_NAME}-miles-run-engine")

        assert _shm_volume(disaggregated_pod) == {
            "name": "dev-shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
        }
        expected_colocate_shm = {
            "name": "dev-shm",
            "hostPath": {"path": f"/dev/shm/miles-{RUN_RELEASE_NAME}", "type": "DirectoryOrCreate"},
        }
        assert _shm_volume(colocated_trainer_pod) == expected_colocate_shm
        assert _shm_volume(colocated_engine_pod) == expected_colocate_shm

    def test_gpu_command_job_mounts_dev_shm_and_receives_rdma_and_gpu_capabilities(self):
        """GPU command-jobs run torch/NCCL workloads and need /dev/shm, RDMA, and IPC_LOCK."""
        job = single_object_of_kind(
            render_run(
                "--show-only",
                "templates/command-job.yaml",
                *_RDMA_FLAG,
                "--set",
                "commandJob.enabled=true",
                "--set",
                "commandJob.name=convert",
                "--set",
                "commandJob.objectName=myrun-miles-run-convert",
                "--set-json",
                'commandJob.command=["bash","-c","convert"]',
                "--set",
                "commandJob.gpusPerPod=4",
            ),
            "Job",
        )
        pod_spec = job["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        assert container["securityContext"] == _GPU_SECURITY_CONTEXT
        assert container["resources"] == {
            "limits": {"nvidia.com/gpu": 4, "rdma/hca_shared_devices_all": 1},
            "requests": {"rdma/hca_shared_devices_all": 1},
        }
        assert _shm_mount(container) == {"name": "dev-shm", "mountPath": "/dev/shm"}
        assert _shm_volume(pod_spec) is not None

    def test_cpu_command_job_and_control_plane_pods_drop_all_capabilities_without_rdma_or_shm(self):
        """CPU-only pods never request RDMA or /dev/shm and drop all Linux capabilities."""
        cpu_job = single_object_of_kind(
            render_run(
                "--show-only",
                "templates/command-job.yaml",
                *_RDMA_FLAG,
                "--set",
                "commandJob.enabled=true",
                "--set",
                "commandJob.name=prep",
                "--set",
                "commandJob.objectName=myrun-miles-run-prep",
                "--set-json",
                'commandJob.command=["bash","-c","prep"]',
                "--set",
                "commandJob.gpusPerPod=0",
            ),
            "Job",
        )
        cpu_job_pod = cpu_job["spec"]["template"]["spec"]
        assert cpu_job_pod["containers"][0]["securityContext"] == _CPU_SECURITY_CONTEXT
        assert "resources" not in cpu_job_pod["containers"][0]
        assert _shm_mount(cpu_job_pod["containers"][0]) is None
        assert _shm_volume(cpu_job_pod) is None

        run_objects = render_run(
            *_RDMA_FLAG,
            "--set",
            "run.mooncake.enabled=true",
            "--set-json",
            f"run.staticWorkers={json.dumps(with_object_names(_STATIC_WORKERS))}",
            "--set-json",
            f"run.inferenceEngines={json.dumps(with_object_names(_ENGINES))}",
            "--set-json",
            f"run.trainerEngines={json.dumps(with_object_names(_TRAINERS))}",
            "--set-json",
            f"run.colocate={json.dumps(_COLOCATE)}",
        )
        for kind, name in (
            ("StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-orchestrator"),
            ("StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-router"),
            ("StatefulSet", f"{RUN_RELEASE_NAME}-miles-run-mooncake-master"),
            ("Deployment", f"{RUN_RELEASE_NAME}-miles-run-colocate-pairing"),
        ):
            pod_spec = named_object(run_objects, kind, name)["spec"]["template"]["spec"]
            container = pod_spec["containers"][0]
            assert container["securityContext"] == _CPU_SECURITY_CONTEXT, name
            assert "rdma/hca_shared_devices_all" not in container.get("resources", {}).get("limits", {}), name
            assert _shm_mount(container) is None, name
