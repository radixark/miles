import yaml
from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    container,
    host_path_volume,
    pod_spec,
    render,
    render_error,
    requires_helm,
    single_object_of_kind,
    volumes_args,
)

from miles.utils.workers.types import ClusterBackend


def _volume(objects: list[dict], name: str) -> dict:
    volumes = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"]
    return next(volume for volume in volumes if volume["name"] == name)


@requires_helm
class TestWorkbenchStatefulSet:
    def test_a_single_pod_idles_on_the_training_image(self):
        """The workbench is one long-lived pod on the training image, not a job that exits."""
        objects = render()
        statefulset = single_object_of_kind(objects, "StatefulSet")

        assert statefulset["spec"]["replicas"] == 1
        assert statefulset["spec"]["serviceName"] == RELEASE_NAME
        assert container(objects)["image"] == "radixark/miles:dev"
        assert container(objects)["imagePullPolicy"] == "Always"
        assert container(objects)["command"] == ["sleep", "infinity"]

    def test_the_selector_is_the_stable_label_subset(self):
        """A selector is immutable after creation and must be a subset of the template labels."""
        statefulset = single_object_of_kind(render(), "StatefulSet")
        selector = statefulset["spec"]["selector"]["matchLabels"]
        template_labels = statefulset["spec"]["template"]["metadata"]["labels"]

        assert selector == {
            "app.kubernetes.io/name": "miles-workbench",
            "app.kubernetes.io/instance": RELEASE_NAME,
            "app.kubernetes.io/component": "workbench",
        }
        assert selector.items() <= template_labels.items()

    def test_the_pod_stays_a_cpu_pod(self):
        """The workbench only parses args and follows logs; its requests must reach the pod as configured."""
        objects = render("--set", "resources.requests.cpu=4", "--set", "resources.limits.memory=16Gi")

        assert container(objects)["resources"]["requests"]["cpu"] == 4
        assert container(objects)["resources"]["limits"]["memory"] == "16Gi"

    def test_the_pod_carries_its_service_account_token(self):
        """helm and kubectl inside the pod act as the chart's ServiceAccount, so its token must be mounted."""
        spec = pod_spec(render())

        assert spec["serviceAccountName"] == RELEASE_NAME
        assert spec["automountServiceAccountToken"] is True

    def test_a_host_path_volume_mounts_where_training_pods_see_it(self):
        """A volume is mounted at the configured path verbatim so launch scripts need no path mapping."""
        objects = render(*volumes_args(host_path_volume(path="/gpfs")))
        volume = _volume(objects, "cluster-storage")

        assert volume["hostPath"] == {"path": "/gpfs", "type": "Directory"}
        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container(objects)["volumeMounts"]

    def test_a_pvc_volume_binds_the_named_claim(self):
        """Clusters without host mounts point the same mount at a pre-existing RWX claim."""
        objects = render(
            "--set-json",
            'infra.volumes=[{"name":"cluster-storage","persistentVolumeClaim":{"claimName":"miles-shared"},'
            '"mounts":[{"mountPath":"/cluster-storage"}]}]',
        )

        assert _volume(objects, "cluster-storage")["persistentVolumeClaim"] == {"claimName": "miles-shared"}

    def test_an_empty_volume_list_leaves_the_pod_with_no_storage_of_its_own(self):
        """Storage is optional; declaring none must not leave a dangling mount referencing a missing volume."""
        objects = render("--set-json", "infra.volumes=[]")

        assert [volume["name"] for volume in pod_spec(objects)["volumes"]] == ["infra-values"]
        assert [mount["name"] for mount in container(objects)["volumeMounts"]] == ["infra-values"]

    def test_a_mount_over_the_image_checkout_replaces_it_in_this_pod_too(self):
        """Launch scripts run from this pod, so it must see the same checkout as the training pods do."""
        objects = render(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/root/miles", "subPath": "alice/miles"}]))
        )

        assert {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "alice/miles"} in container(objects)[
            "volumeMounts"
        ]

    def test_a_run_launched_from_the_pod_picks_this_backend_without_being_told(self):
        """The pod exists to install runs into its own cluster, so naming the backend again is noise."""
        assert (
            dict(name="MILES_SCRIPT_CLUSTER_BACKEND", value=ClusterBackend.KUBERNETES.value)
            in container(render())["env"]
        )

    def test_a_cluster_value_still_wins_over_that_default(self):
        """The default is a convenience, and a cluster that sets the variable itself means it."""
        objects = render("--set", "infra.env.MILES_SCRIPT_CLUSTER_BACKEND=ray")

        assert dict(name="MILES_SCRIPT_CLUSTER_BACKEND", value="ray") in container(objects)["env"]

    def test_a_run_lands_in_the_namespace_the_pod_lives_in(self):
        """The pod's Role is namespaced, so its own namespace is the only one it can install into."""
        assert dict(name="MILES_SCRIPT_NAMESPACE", value=NAMESPACE) in container(render())["env"]

    def test_the_installed_helm_values_are_mounted_for_the_runs_launched_here(self):
        """A run rendered from a second copy of these values would schedule its pods against another cluster."""
        objects = render(*volumes_args(host_path_volume(path="/gpfs")))
        data = single_object_of_kind(objects, "ConfigMap")["data"]["infra.yaml"]

        assert yaml.safe_load(data)["infra"]["volumes"][0]["hostPath"]["path"] == "/gpfs"
        assert {"name": "infra-values", "mountPath": "/etc/miles"} in container(objects)["volumeMounts"]
        assert _volume(objects, "infra-values")["configMap"]["name"].endswith("-infra")
        assert dict(name="MILES_SCRIPT_HELM_VALUES", value="/etc/miles/infra.yaml") in container(objects)["env"]

    def test_scheduling_and_environment_values_reach_the_pod(self, tmp_path):
        """Cluster-specific scheduling and environment values are passed through untouched."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(
            yaml.safe_dump(
                dict(
                    infra=dict(
                        scheduling=dict(
                            nodeSelector={"pool": "cpu"},
                            tolerations=[dict(key="gpu", operator="Exists", effect="NoSchedule")],
                            affinity=dict(podAntiAffinity=dict(preferredDuringSchedulingIgnoredDuringExecution=[])),
                        ),
                        env={"HTTP_PROXY": "http://proxy:7890"},
                    )
                )
            )
        )
        objects = render("-f", str(values_file))
        spec = pod_spec(objects)

        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert spec["tolerations"] == [dict(key="gpu", operator="Exists", effect="NoSchedule")]
        assert spec["affinity"] == dict(podAntiAffinity=dict(preferredDuringSchedulingIgnoredDuringExecution=[]))
        assert dict(name="HTTP_PROXY", value="http://proxy:7890") in container(objects)["env"]

    def test_private_registries_get_their_pull_secret(self):
        """The training image often lives in a private registry, so its pull secret must be declared."""
        objects = render("--set", "infra.image.pullSecrets[0]=registry-cred")

        assert pod_spec(objects)["imagePullSecrets"] == [dict(name="registry-cred")]

    def test_defaults_leave_scheduling_and_pull_secrets_unset(self):
        """Empty scheduling values must be omitted rather than rendered as empty API fields."""
        spec = pod_spec(render())

        assert "nodeSelector" not in spec
        assert "tolerations" not in spec
        assert "affinity" not in spec
        assert "imagePullSecrets" not in spec


@requires_helm
class TestNamespaceInterpolation:
    def test_the_workbench_mounts_what_its_own_namespace_names(self):
        """One workbench per namespace, and the same values file has to give each of them its own checkout."""
        objects = render(
            *volumes_args(
                host_path_volume(
                    path="/data/${NAMESPACE}",
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPACE}/miles"},
                    ],
                )
            )
        )

        assert _volume(objects, "cluster-storage")["hostPath"]["path"] == f"/data/{NAMESPACE}"
        assert {
            "name": "cluster-storage",
            "mountPath": "/root/miles",
            "subPath": f"repos/{NAMESPACE}/miles",
        } in container(objects)["volumeMounts"]

    def test_an_unknown_variable_is_refused_by_a_chart_that_renders_no_run(self):
        """This chart writes the infra.yaml every launch from here reads, so a typo has to stop at this render."""
        error = render_error("--set", "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data")

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error
