from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.charts.utils import documents_of, objects_added_by, requires_helm
from tests.fast.e2e.external_rollout_script import load_external_rollout_script

from miles.utils.external_utils.command_utils import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import InfraValues
from miles.utils.workers.types import ClusterBackend

script = load_external_rollout_script()

HOST_PATH_INFRA: dict[str, Any] = {
    "image": {"repository": "radixark/miles", "tag": "dev"},
    "sharedStorage": {"type": "hostPath", "hostPath": "/cluster-storage", "mountPath": "/cluster-storage"},
}


def infra_with(**overrides: Any) -> dict[str, Any]:
    return {**HOST_PATH_INFRA, **overrides}


def manifests_of(infra: dict[str, Any]) -> str:
    return script._engine_manifests(InfraValues.model_validate(infra))


def stateful_set_of(infra: dict[str, Any]) -> dict[str, Any]:
    return documents_of(manifests_of(infra))[1]


def pod_spec_of(infra: dict[str, Any]) -> dict[str, Any]:
    return stateful_set_of(infra)["spec"]["template"]["spec"]


def container_of(infra: dict[str, Any]) -> dict[str, Any]:
    return pod_spec_of(infra)["containers"][0]


def values_file(tmp_path: Path, infra: dict[str, Any], name: str = "infra.yaml") -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"infra": infra}))
    return str(path)


def engines_of_kubernetes(tmp_path: Path, infra: dict[str, Any] = HOST_PATH_INFRA):
    config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, helm_values=(values_file(tmp_path, infra),))
    return script._external_engines(config)


@pytest.fixture(autouse=True)
def host_without_partition_or_wandb_key(monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class TestTheBackendDecidesWhoStartsTheEngines:
    def test_a_ray_run_starts_them_itself_and_installs_nothing(self):
        """Ray installs no release, so the only place left to start an engine is beside the trainer."""
        engines = script._external_engines(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY))

        assert engines.extra_manifests == []
        assert set(engines.prepare_cmd) == {"trainer"}
        assert engines.addrs == [f"127.0.0.1:{port}" for port in script.RAY_ENGINE_PORTS]

    def test_a_kubernetes_run_installs_them_and_asks_no_pod_to_start_them(self, tmp_path):
        """A trainer pod that also launched the engines would take their gpus and pin them to one node."""
        engines = engines_of_kubernetes(tmp_path)

        assert engines.prepare_cmd == {}
        assert len(engines.extra_manifests) == 1

    def test_the_addrs_the_run_is_given_are_the_pods_the_manifest_declares(self, tmp_path):
        """These two are written in different places, and a drift between them is a run dialling nothing."""
        engines = engines_of_kubernetes(tmp_path)
        stateful_set = documents_of(engines.extra_manifests[0])[1]
        service_name = stateful_set["spec"]["serviceName"]
        port = stateful_set["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"]

        assert engines.addrs == [
            f"{stateful_set['metadata']['name']}-{index}.{service_name}:{port}"
            for index in range(stateful_set["spec"]["replicas"])
        ]

    def test_both_backends_run_the_same_engine(self):
        """The point of the second backend is a new place to run the test, not a second thing to test."""
        ray_command = script._external_engines(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)).prepare_cmd
        argv = container_of(HOST_PATH_INFRA)["command"]

        assert " ".join(argv[:5]) in ray_command["trainer"]
        assert argv[-3:] == ["--mem-fraction-static", "0.7", "--trust-remote-code"]

    def test_the_run_sizes_its_router_after_the_engines_that_exist(self, tmp_path):
        """--rollout-num-gpus is cross-checked against what the engines report, and a mismatch aborts the run."""
        engines = engines_of_kubernetes(tmp_path)
        train_args = script._train_args(engines.addrs, object_store_args="").split()

        gpus = int(train_args[train_args.index("--rollout-num-gpus") + 1])
        assert gpus == script.NUM_ENGINES * script.GPUS_PER_ENGINE


class TestTheObjectStoreTheBackendCanActuallyServe:
    def test_a_kubernetes_run_names_the_mooncake_store_it_will_be_given(self):
        """The launcher forces mooncake on kubernetes and then aborts on the init kwargs nobody passed."""
        args = script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES))

        assert "--object-store-backend mooncake" in args
        assert "--mooncake-store-init-kwargs" in args

    def test_a_ray_run_keeps_the_store_it_already_had(self):
        """Ray serves the default store itself, and naming mooncake here would test a second thing at once."""
        assert script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)) == ""

    def test_the_kubernetes_train_args_carry_it(self):
        """These args are the only place the run learns of the store, and a drop aborts before helm installs."""
        args = script._train_args(
            ["engine:30000"],
            object_store_args=script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES)),
        )

        assert "--mooncake-store-init-kwargs" in args


class TestTheEnginesArePublishedOnlyOnceTheyServe:
    def test_the_pods_get_a_dns_name_of_their_own(self):
        """The provider dials one address per engine, which only a headless service gives it."""
        service = documents_of(manifests_of(HOST_PATH_INFRA))[0]

        assert service["spec"]["clusterIP"] == "None"
        assert "publishNotReadyAddresses" not in service["spec"]

    def test_an_engine_counts_as_ready_only_once_it_can_generate(self):
        """Serving http is not serving the model, and the run's own health check is this same path."""
        probe = container_of(HOST_PATH_INFRA)["readinessProbe"]

        assert probe["httpGet"]["path"] == "/health_generate"

    def test_the_engines_start_at_the_same_time(self):
        """The default policy starts them one after the other, doubling a wait the run is already timing."""
        assert stateful_set_of(HOST_PATH_INFRA)["spec"]["podManagementPolicy"] == "Parallel"


class TestTheEnginesAreDescribedFromTheClusterItsOwnInfra:
    def test_they_run_the_image_the_cluster_named(self):
        """A hardcoded image would either miss sglang or pin the test to one registry."""
        container = container_of(HOST_PATH_INFRA)

        assert container["image"] == "radixark/miles:dev"
        assert "imagePullPolicy" not in container

    def test_an_image_the_cluster_qualified_is_taken_whole(self):
        """A private registry needs both halves, and dropping either is a pod that never pulls."""
        infra = infra_with(
            image={**HOST_PATH_INFRA["image"], "pullPolicy": "Always", "pullSecrets": ["regcred"]},
        )

        assert container_of(infra)["imagePullPolicy"] == "Always"
        assert pod_spec_of(infra)["imagePullSecrets"] == [{"name": "regcred"}]

    def test_they_mount_the_storage_the_run_reads_its_checkpoint_from(self):
        """The engines serve the same model the trainer updates, and only the mount makes it one file."""
        assert pod_spec_of(HOST_PATH_INFRA)["volumes"] == [
            {"name": "shared-storage", "hostPath": {"path": "/cluster-storage", "type": "Directory"}}
        ]
        assert container_of(HOST_PATH_INFRA)["volumeMounts"] == [
            {"name": "shared-storage", "mountPath": "/cluster-storage"}
        ]

    def test_a_cluster_that_shares_a_claim_gets_a_claim(self):
        """hostPath is one cluster's answer; a pvc cluster would silently read an empty directory."""
        infra = infra_with(
            sharedStorage={"type": "pvc", "pvcClaimName": "miles-data", "mountPath": "/cluster-storage"},
        )

        assert pod_spec_of(infra)["volumes"] == [
            {"name": "shared-storage", "persistentVolumeClaim": {"claimName": "miles-data"}}
        ]

    def test_a_cluster_that_shares_nothing_mounts_nothing(self):
        """An empty volume section renders a pod kubernetes refuses, which reads as a chart bug."""
        infra = infra_with(sharedStorage={"type": "none", "mountPath": "/cluster-storage"})

        assert "volumes" not in pod_spec_of(infra)
        assert "volumeMounts" not in container_of(infra)

    def test_they_land_where_the_run_own_pods_land(self):
        """A gpu pod that ignores the cluster's scheduling stays Pending until the run times out."""
        infra = infra_with(
            scheduling={"nodeSelector": {"gpu": "h200"}, "tolerations": [{"key": "gpu", "operator": "Exists"}]},
        )
        pod_spec = pod_spec_of(infra)

        assert pod_spec["nodeSelector"] == {"gpu": "h200"}
        assert pod_spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert "affinity" not in pod_spec

    def test_they_inherit_the_environment_every_other_pod_gets(self):
        """Proxy and cache settings are how a pod reaches anything at all on a closed cluster."""
        infra = infra_with(env={"HF_HUB_OFFLINE": "1"})

        assert container_of(infra)["env"] == [{"name": "HF_HUB_OFFLINE", "value": "1"}]

    def test_each_engine_asks_for_the_gpus_it_will_use(self):
        """Without a request the scheduler stacks every engine onto one node and they fight over memory."""
        assert container_of(HOST_PATH_INFRA)["resources"] == {"limits": {"nvidia.com/gpu": script.GPUS_PER_ENGINE}}

    def test_no_object_names_a_namespace_of_its_own(self):
        """These are installed into whichever namespace the run is; a pinned one is another cluster's."""
        documents = documents_of(manifests_of(HOST_PATH_INFRA))

        assert all("namespace" not in document["metadata"] for document in documents)


class TestTheInfraValuesAreReadTheWayHelmReadsThem:
    def test_a_later_values_file_overrides_only_what_it_names(self, tmp_path):
        """helm merges its -f files, so reading only the last one would drop the cluster's own defaults."""
        base = values_file(tmp_path, HOST_PATH_INFRA)
        override = values_file(tmp_path, {"image": {"tag": "mine"}}, name="override.yaml")

        infra = script._infra_values((base, override))

        assert infra.image.repository == "radixark/miles"
        assert infra.image.tag == "mine"
        assert infra.shared_storage.mount_path == "/cluster-storage"

    def test_a_launch_that_names_no_values_file_reads_what_the_chart_would(self):
        """helm starts from the chart's own values.yaml, and the launcher accepts a launch that adds none."""
        infra = script._infra_values(())

        assert infra.image.repository
        assert infra.shared_storage.mount_path

    def test_a_file_that_names_only_a_tag_is_a_legal_override(self, tmp_path):
        """This is a whole run's worth of values for the run itself, and refusing it refuses a valid launch."""
        infra = script._infra_values((values_file(tmp_path, {"image": {"tag": "mine"}}),))

        assert infra.image.tag == "mine"
        assert infra.shared_storage.mount_path

    def test_a_run_that_overrides_sglang_is_refused(self, tmp_path):
        """The run's own pods import that checkout while these engines serve the image's, so the two diverge."""
        override = values_file(tmp_path, {**HOST_PATH_INFRA, "paths": {"repos": {"sglang": "mine/sglang"}}})

        with pytest.raises(AssertionError, match="infra.paths.repos.sglang"):
            script._infra_values((override,))


@requires_helm
class TestTheChartInstallsTheEnginesAsWritten:
    def test_the_objects_the_script_describes_are_the_objects_the_release_installs(self, tmp_path):
        """The manifest only matters once the chart has rendered it, and it is rendered verbatim."""
        manifest = engines_of_kubernetes(tmp_path).extra_manifests[0]

        assert objects_added_by(manifest) == documents_of(manifest)
