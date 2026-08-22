import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    render_run,
    render_run_error,
    requires_helm,
    with_object_names,
)

TRAINER = [
    {
        "name": "trainer-engine-actor",
        "command": ["python", "-m", "miles.utils.workers.process_supervisor"],
        "resources": {"limits": {"nvidia.com/gpu": 4}},
    }
]

ENGINE = [
    {
        "name": "engine",
        "command": ["python", "-m", "miles.utils.workers.process_supervisor"],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    }
]

COLOCATE = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": TRAINER[0]["name"],
    "inference_pools": [
        {
            "pool_id": ENGINE[0]["name"],
            "layout": {
                "num_inference_cells": 1,
                "num_trainer_cells": 1,
                "num_pods_per_inference_cell": 1,
                "num_pods_per_trainer_cell": 1,
                "num_gpus_per_node": 8,
                "num_gpus_per_inference_pod": 8,
                "gpu_offset": 0,
            },
        }
    ],
}


def _pod_spec_of_the_only_pool(*args: str) -> dict[str, Any]:
    rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}", *args)
    pool = objects_of_kind(rendered, "LeaderWorkerSet")[0]
    return pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _pod_spec_of_the_colocated_trainer_pool() -> dict[str, Any]:
    rendered = render_run(
        "--set-json",
        f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}",
        "--set-json",
        f"run.inferenceEngines={json.dumps(with_object_names(ENGINE))}",
        "--set-json",
        f"run.colocate={json.dumps(COLOCATE)}",
    )
    pool = named_object(rendered, "LeaderWorkerSet", f"{RUN_RELEASE_NAME}-miles-run-{TRAINER[0]['name']}")
    return pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def _shm_volume(pod_spec: dict[str, Any]) -> dict[str, Any]:
    [mount] = _shm_mounts(pod_spec)
    [volume] = [volume for volume in pod_spec["volumes"] if volume["name"] == mount["name"]]
    return volume


def _shm_mounts(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        mount
        for container in pod_spec["containers"]
        for mount in container.get("volumeMounts", [])
        if mount["mountPath"] == "/dev/shm"
    ]


@requires_helm
class TestPoolPodsShareTheHostSharedMemory:
    def test_a_pool_container_mounts_dev_shm(self):
        """Kubernetes' default 64Mi of /dev/shm is less than NCCL asks for per peer it cannot reach over p2p."""
        assert len(_shm_mounts(_pod_spec_of_the_only_pool())) == 1

    def test_that_mount_is_the_host_dev_shm_itself(self):
        """The pods share the host IPC namespace, and a CUDA IPC refcounter only works in the shm they all see."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["hostPath"]["path"] == "/dev/shm"

    def test_the_host_directory_has_to_exist_already(self):
        """DirectoryOrCreate would silently make a plain disk directory on a node whose /dev/shm is missing."""
        assert _shm_volume(_pod_spec_of_the_only_pool())["hostPath"]["type"] == "Directory"

    def test_no_private_volume_of_its_own_shadows_it(self):
        """An emptyDir here is what breaks the sharing, so its absence is the property worth pinning."""
        assert "emptyDir" not in _shm_volume(_pod_spec_of_the_only_pool())

    def test_the_pods_that_mount_it_share_the_host_ipc_namespace(self):
        """Sharing the directory buys nothing unless the pods also share the namespace the handles live in."""
        pod_spec = _pod_spec_of_the_colocated_trainer_pool()

        assert len(_shm_mounts(pod_spec)) == 1
        assert pod_spec["hostIPC"] is True

    def test_a_pool_that_shares_no_handles_gets_the_directory_and_not_the_namespace(self):
        """A run nothing colocates needs the size nccl asks for, and no second pod holds handles into it."""
        pod_spec = _pod_spec_of_the_only_pool()

        assert len(_shm_mounts(pod_spec)) == 1
        assert "hostIPC" not in pod_spec

    def test_a_cluster_whose_nodes_keep_shared_memory_elsewhere_can_say_so(self):
        """The path was hardcoded, which left a node that mounts its shm elsewhere with no way to be described."""
        pod_spec = _pod_spec_of_the_only_pool("--set", "infra.devShm.hostPath.path=/run/shm")

        assert _shm_volume(pod_spec)["hostPath"]["path"] == "/run/shm"

    def test_a_cluster_that_would_rather_size_it_itself_can_ask_for_memory(self):
        """The size knob the run lost only ever made sense on a volume of the pod's own, which is this one."""
        pod_spec = _pod_spec_of_the_only_pool(
            "--set", "infra.devShm.hostPath=null", "--set-json", 'infra.devShm.emptyDir={"medium":"Memory"}'
        )

        assert _shm_volume(pod_spec)["emptyDir"] == {"medium": "Memory"}

    def test_a_run_cannot_ask_for_a_private_size_any_more(self):
        """The size knob belonged to the emptyDir; leaving it accepted would let a values file break sharing."""
        error = render_run_error("--set", "run.shmSize=8Gi")

        assert "'shmSize' not allowed" in error

    def test_every_mounted_volume_is_declared_by_the_pod_that_mounts_it(self):
        """A container naming a volume the pod does not declare makes the whole manifest invalid."""
        pod_spec = _pod_spec_of_the_only_pool()
        declared = {volume["name"] for volume in pod_spec["volumes"]}
        mounted = {mount["name"] for container in pod_spec["containers"] for mount in container["volumeMounts"]}

        assert mounted <= declared, f"{mounted - declared} is mounted but never declared"

    def test_a_pod_that_runs_no_collective_is_left_alone(self):
        """The orchestrator only talks to the apiserver, so it has no reason to reach into the host's shm."""
        rendered = render_run("--set-json", f"run.trainerEngines={json.dumps(with_object_names(TRAINER))}")
        [orchestrator] = [
            obj for obj in objects_of_kind(rendered, "StatefulSet") if "orchestrator" in obj["metadata"]["name"]
        ]

        assert not _shm_mounts(orchestrator["spec"]["template"]["spec"])
