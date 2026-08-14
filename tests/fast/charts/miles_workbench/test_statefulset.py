import yaml
from tests.fast.charts.utils import RELEASE_NAME, container, pod_spec, render, requires_helm, single_object_of_kind


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

    def test_host_path_storage_mounts_where_training_pods_see_it(self):
        """Shared storage is mounted at the configured path verbatim so launch scripts need no path mapping."""
        objects = render("--set", "infra.sharedStorage.hostPath=/gpfs", "--set", "infra.sharedStorage.mountPath=/cluster-storage")
        volume = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"][0]

        assert volume["hostPath"] == {"path": "/gpfs", "type": "Directory"}
        assert container(objects)["volumeMounts"] == [{"name": volume["name"], "mountPath": "/cluster-storage"}]

    def test_pvc_storage_binds_the_named_claim(self):
        """Clusters without host mounts point the same mount at a pre-existing RWX claim."""
        objects = render("--set", "infra.sharedStorage.type=pvc", "--set", "infra.sharedStorage.pvcClaimName=miles-shared")
        volume = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"][0]

        assert volume["persistentVolumeClaim"] == {"claimName": "miles-shared"}

    def test_storage_type_none_leaves_the_pod_without_volumes(self):
        """Storage is optional; disabling it must not leave a dangling mount referencing a missing volume."""
        objects = render("--set", "infra.sharedStorage.type=none")

        assert "volumes" not in pod_spec(objects)
        assert "volumeMounts" not in container(objects)

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
                    ),
                )
            )
        )
        objects = render("-f", str(values_file))
        spec = pod_spec(objects)

        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert spec["tolerations"] == [dict(key="gpu", operator="Exists", effect="NoSchedule")]
        assert spec["affinity"] == dict(podAntiAffinity=dict(preferredDuringSchedulingIgnoredDuringExecution=[]))
        assert container(objects)["env"] == [dict(name="HTTP_PROXY", value="http://proxy:7890")]

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
