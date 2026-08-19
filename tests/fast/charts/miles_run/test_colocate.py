import json
from typing import Any

from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    render_run,
    requires_helm,
    with_object_names,
)

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values import utils as values_utils

from miles.utils.external_utils.colocate_pairing.pods import _GATE_NAME, release_patch
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.workers.env_vars import BASE_GPU_ID_ENV_VAR
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

ENGINES = [
    {
        "name": "engine",
        "replicas": 2,
        "size": 4,
        "command": ["python"],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    },
    {"name": "decode", "replicas": 2, "size": 2, "command": ["python"]},
]
TRAINERS = [{"name": "trainer", "replicas": 3, "size": 2, "command": ["python"]}]

POOLS = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(ENGINES))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINERS))}",
)

PAIRING = "myrun-miles-run-colocate-pairing"
ORCHESTRATOR_ROLE = "myrun-miles-run-orchestrator"


def layout(
    *,
    num_inference_cells: int,
    num_pods_per_inference_cell: int,
    gpu_offset: int,
    num_gpus_per_inference_pod: int = 8,
) -> dict[str, int]:
    return {
        "num_inference_cells": num_inference_cells,
        "num_trainer_cells": 3,
        "num_pods_per_inference_cell": num_pods_per_inference_cell,
        "num_pods_per_trainer_cell": 2,
        "num_gpus_per_node": 8,
        "num_gpus_per_inference_pod": num_gpus_per_inference_pod,
        "gpu_offset": gpu_offset,
    }


PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": layout(num_inference_cells=2, num_pods_per_inference_cell=2, gpu_offset=0),
        },
        {
            "pool_id": "decode",
            "layout": layout(num_inference_cells=2, num_pods_per_inference_cell=2, gpu_offset=32),
        },
    ],
}

ENABLE = (*POOLS, "--set-json", f"run.colocate={json.dumps(PAIRING_CONFIG)}")

GATE = {"name": _GATE_NAME}


def pool_pod(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    workload = named_object(objects, "LeaderWorkerSet", name)
    return workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def colocated_engine_pod(*args: str) -> dict[str, Any]:
    return pool_pod(render_run(*ENABLE, *args), "myrun-miles-run-engine")


def _env_names(container: dict[str, Any]) -> set[str]:
    return {entry["name"] for entry in container.get("env", [])}


def _env_entry(container: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((entry for entry in container.get("env", []) if entry["name"] == name), None)


def pairing_config() -> dict[str, Any]:
    deployment = named_object(render_run(*ENABLE), "Deployment", PAIRING)
    command = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command[3] == "--config", command
    return json.loads(command[4])


@requires_helm
class TestColocatedEnginePool:
    def test_is_held_back_from_the_scheduler_by_the_chart_itself(self):
        """Nothing else can keep a pod unscheduled until another pod's node is known."""
        assert colocated_engine_pod()["schedulingGates"] == [GATE]

    def test_shares_the_host_ipc_namespace(self):
        """A CUDA IPC handle's reference counter lives in shared memory, so both pods need the same one."""
        assert colocated_engine_pod()["hostIPC"] is True

    def test_sees_every_gpu_on_the_node_it_lands_on(self):
        """It requests no gpu of its own, so only the device plugin bypass makes the trainer's gpus visible."""
        assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"} in colocated_engine_pod()["containers"][0]["env"]

    def test_requests_no_gpus_of_its_own(self):
        """The trainer requests the whole node, and two claims on one gpu would never both schedule."""
        assert colocated_engine_pod()["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 0

    def test_gives_a_second_colocated_pool_the_same_treatment(self):
        """A prefill/decode run gates both pool_ids, and one left ungated would race the trainer for its gpus."""
        pod = pool_pod(render_run(*ENABLE), "myrun-miles-run-decode")

        assert pod["schedulingGates"] == [GATE]
        assert pod["hostIPC"] is True
        assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 0
        assert BASE_GPU_ID_ENV_VAR in _env_names(pod["containers"][0])

    def test_is_told_which_card_of_the_node_the_pairing_gave_it(self):
        """The controller writes that annotation in the patch that releases the gate, before the pod runs."""
        entry = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)

        assert entry == {
            "name": BASE_GPU_ID_ENV_VAR,
            "valueFrom": {
                "fieldRef": {"fieldPath": f"metadata.annotations['{DEFAULT_LABEL_KEYS.base_gpu_id_annotation}']"}
            },
        }

    def test_reads_that_card_off_the_pod_rather_than_a_rendered_value(self):
        """helm renders one pod template for the whole pool, and the card differs from pod to pod."""
        entry = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)

        assert "value" not in entry

    def test_carries_no_affinity_at_all_but_keeps_the_node_selector(self):
        """Any affinity would contradict the node the controller picks; the selector it only adds to."""
        pod = colocated_engine_pod(
            "--set-json", 'infra.scheduling={"nodeSelector":{"pool":"gpu"},"affinity":{"nodeAffinity":{}}}'
        )

        assert "affinity" not in pod
        assert pod["nodeSelector"] == {"pool": "gpu"}


@requires_helm
class TestColocatedTrainerPool:
    def test_shares_the_host_ipc_namespace_too(self):
        """The engine's CUDA IPC handles are only usable from a trainer in the same IPC namespace."""
        assert pool_pod(render_run(*ENABLE), "myrun-miles-run-trainer")["hostIPC"] is True

    def test_is_scheduled_normally_and_gets_none_of_the_engine_treatment(self):
        """It is the pod that claims the node, so gating it would leave nothing for the engine to pair with."""
        pod = pool_pod(render_run(*ENABLE), "myrun-miles-run-trainer")

        assert "schedulingGates" not in pod
        assert not _env_names(pod["containers"][0]) & {"NVIDIA_VISIBLE_DEVICES", BASE_GPU_ID_ENV_VAR}


@requires_helm
class TestDisaggregatedRun:
    def test_leaves_the_engine_pool_ungated_and_holding_its_own_gpus(self):
        """The same pool_id values must render an ordinary engine when the run is not colocated."""
        pod = pool_pod(render_run(*POOLS), "myrun-miles-run-engine")

        assert "schedulingGates" not in pod
        assert "hostIPC" not in pod
        assert not _env_names(pod["containers"][0]) & {"NVIDIA_VISIBLE_DEVICES", BASE_GPU_ID_ENV_VAR}
        assert pod["containers"][0]["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_installs_no_pairing_controller(self):
        """A run whose engines have their own nodes must not gain a controller with pod write rights."""
        objects = render_run(*POOLS)

        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "Role")] == [ORCHESTRATOR_ROLE]
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "RoleBinding")] == [ORCHESTRATOR_ROLE]
        assert objects_of_kind(objects, "Deployment") == []
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "ServiceAccount")] == [ORCHESTRATOR_ROLE]


@requires_helm
class TestPairingController:
    def test_holds_only_namespaced_rights_over_pods(self):
        """Releasing a gate is an ordinary pod update, so nothing cluster-scoped is needed."""
        role = named_object(render_run(*ENABLE), "Role", PAIRING)

        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "patch", "update"]}
        ]

    def test_never_asks_for_the_binding_subresource(self):
        """That belongs to the scheduler, and asking for it would make this a scheduler replacement."""
        rules = named_object(render_run(*ENABLE), "Role", PAIRING)["rules"]

        assert not any("binding" in resource for rule in rules for resource in rule["resources"])

    def test_runs_as_a_single_replica(self):
        """If it dies, new engine pods stay Pending, which is safe and visible; two would race each other."""
        assert named_object(render_run(*ENABLE), "Deployment", PAIRING)["spec"]["replicas"] == 1

    def test_is_handed_the_pairing_config_from_the_values_untouched(self):
        """The launcher already knows every pool's shape, so a chart that rebuilt it could only get it wrong."""
        assert pairing_config() == PAIRING_CONFIG

    def test_carries_the_cluster_environment(self):
        """It talks to the api server, so a cluster that needs a proxy needs it here as well."""
        objects = render_run(*ENABLE, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")
        deployment = named_object(objects, "Deployment", PAIRING)

        assert {"name": "HTTP_PROXY", "value": "http://proxy:7890"} in (
            deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        )


@requires_helm
class TestAPoolTheConfigDoesNotName:
    def test_renders_as_an_ordinary_pool(self):
        """The config names object names the launcher derived, so anything else is simply not colocated."""
        config = {**PAIRING_CONFIG, "inference_pools": PAIRING_CONFIG["inference_pools"][:1]}
        pod = pool_pod(
            render_run(*POOLS, "--set-json", f"run.colocate={json.dumps(config)}"), "myrun-miles-run-decode"
        )

        assert "schedulingGates" not in pod
        assert "hostIPC" not in pod
        assert BASE_GPU_ID_ENV_VAR not in _env_names(pod["containers"][0])


def _sub_node_engine_argv() -> list[str]:
    specs = [values_utils.engine(num_cells=2, gpus_per_engine=4), values_utils.trainer(num_cells=1, gpus_per_cell=8)]
    plan = values_utils.LAYOUT.model_copy(update={"colocate": True})
    return build_values(specs, plan).as_values()["run"]["inferenceEngines"][0]["command"]


@requires_helm
class TestTheCardReachesTheEngineByOneNameAndOneKey:
    def test_the_launcher_the_chart_and_the_controller_all_spell_it_the_same_way(self):
        """Three producers of two strings: drift in any one leaves the engine reading an empty base gpu id."""
        argv = _sub_node_engine_argv()
        env = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)
        patch = release_patch(
            node_name="gpu-3", base_gpu_id=4, gates=[_GATE_NAME], has_node_selector=False, annotations={"a": "b"}
        )
        annotation = DEFAULT_LABEL_KEYS.base_gpu_id_annotation

        assert argv[argv.index("--base-gpu-id") + 1] == f"$({BASE_GPU_ID_ENV_VAR})"
        assert env["valueFrom"]["fieldRef"]["fieldPath"] == f"metadata.annotations['{annotation}']"
        assert [operation["path"] for operation in patch if operation["path"].startswith("/metadata")] == [
            f"/metadata/annotations/{annotation.replace('/', '~1')}"
        ]


SUB_NODE_ENGINES = [
    {
        "name": "engine",
        "replicas": 2,
        "size": 1,
        "command": _sub_node_engine_argv(),
        "resources": {"limits": {"nvidia.com/gpu": 4}},
    }
]

SUB_NODE_PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": layout(
                num_inference_cells=2,
                num_pods_per_inference_cell=1,
                gpu_offset=0,
                num_gpus_per_inference_pod=4,
            ),
        }
    ],
}

SUB_NODE_ENABLE = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(SUB_NODE_ENGINES))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINERS))}",
    "--set-json",
    f"run.colocate={json.dumps(SUB_NODE_PAIRING_CONFIG)}",
)


def _pools_by_name(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        pool["metadata"]["name"]: pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
        for pool in objects_of_kind(objects, "LeaderWorkerSet")
    }


@requires_helm
class TestTheKubeletVariableAndTheGateAlwaysTravelTogether:
    def test_the_pod_asking_for_the_card_is_the_pod_that_is_told_it(self):
        """The launcher writes the argument and the chart writes the variable, from two unrelated inputs."""
        container = _pools_by_name(render_run(*SUB_NODE_ENABLE))["myrun-miles-run-engine"]["containers"][0]

        assert f"$({BASE_GPU_ID_ENV_VAR})" in container["command"]
        assert BASE_GPU_ID_ENV_VAR in _env_names(container)

    def test_every_pool_asking_for_the_card_is_a_pool_the_chart_gates(self):
        """An ungated pod runs before the controller annotates it, so the variable expands to nothing."""
        pools = _pools_by_name(render_run(*SUB_NODE_ENABLE))
        asking = {
            name for name, pod in pools.items() if f"$({BASE_GPU_ID_ENV_VAR})" in pod["containers"][0]["command"]
        }
        gated = {name for name, pod in pools.items() if pod.get("schedulingGates")}

        assert asking
        assert asking <= gated
