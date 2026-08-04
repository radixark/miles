import pytest
import yaml
from tests.fast.charts.utils import container, pod_spec, render, render_error, requires_helm, single_object_of_kind


@requires_helm
class TestValuesSchema:
    def test_an_unknown_key_inside_a_known_section_is_rejected(self):
        """A typo inside a section the chart owns must fail the install rather than be silently ignored."""
        assert "taag" in render_error("--set", "infra.image.taag=dev")

    def test_an_unknown_top_level_key_is_rejected(self):
        """A mistyped section name must fail: "rbca.create=false" otherwise silently keeps both privileged bindings."""
        assert "rbca" in render_error("--set", "rbca.create=false")

    def test_names_too_long_for_the_api_are_rejected(self):
        """Both names are DNS subdomains, which the API server caps at 253 characters."""
        long_name = "a" * 254

        assert "pvcClaimName" in render_error(
            "--set", "infra.sharedStorage.type=pvc", "--set", f"infra.sharedStorage.pvcClaimName={long_name}"
        )
        assert "serviceAccount/name" in render_error("--set", f"serviceAccount.name={long_name}")

    def test_a_relative_host_path_is_rejected(self):
        """A relative hostPath renders and installs, then leaves the pod stuck on a failed mount."""
        assert "hostPath" in render_error("--set", "infra.sharedStorage.hostPath=relative/dir")

    @pytest.mark.parametrize("path", ["/../cluster-storage", "/cluster-storage/.."])
    def test_a_host_path_containing_a_backstep_is_rejected(self, path):
        """Kubernetes rejects any ".." segment in a volume path, so catch it at install time."""
        assert "hostPath" in render_error("--set", f"infra.sharedStorage.hostPath={path}")

    def test_a_host_path_that_merely_contains_dots_is_accepted(self):
        """Only a whole ".." segment is a backstep; dots inside a name are ordinary characters."""
        objects = render("--set", "infra.sharedStorage.hostPath=/a..b/c")
        volume = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"][0]

        assert volume["hostPath"]["path"] == "/a..b/c"

    def test_an_environment_name_the_api_forbids_is_rejected(self):
        """Kubernetes allows printable ASCII in an env name except "=", which would also break the rendering."""
        assert "propertyName" in render_error("--set-string", "infra.env.A\\=B=value")

    def test_an_unusual_but_legal_environment_name_is_accepted(self):
        """The pattern is the API contract, not the conventional identifier shape, so it must not be stricter."""
        objects = render("--set-string", "infra.env.FOO/BAR=value", "--set-string", "infra.env.FOO\\ BAZ=value")

        assert {entry["name"] for entry in container(objects)["env"]} == {"FOO/BAR", "FOO BAZ"}

    def test_a_malformed_claim_name_is_rejected(self):
        """A claim name is a Kubernetes object name; "a..b" passes a loose pattern but no real API server."""
        assert "pvcClaimName" in render_error(
            "--set", "infra.sharedStorage.type=pvc", "--set", "infra.sharedStorage.pvcClaimName=a..b"
        )

    def test_yaml_boolean_lookalike_names_stay_strings(self):
        """Names like "on" are valid Kubernetes names but YAML booleans, so they must be rendered quoted."""
        objects = render(
            "--set",
            "infra.sharedStorage.type=pvc",
            "--set",
            "infra.sharedStorage.pvcClaimName=on",
            "--set",
            "serviceAccount.name=no",
        )
        volume = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"][0]

        assert volume["persistentVolumeClaim"]["claimName"] == "on"
        assert pod_spec(objects)["serviceAccountName"] == "no"

    def test_yaml_boolean_lookalike_environment_names_stay_strings(self, tmp_path):
        """Names like "on" and "null" are legal environment variable names but YAML booleans and nulls."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump({"infra": {"env": {"on": "1", "null": "3"}}}))
        objects = render("-f", str(values_file))

        assert container(objects)["env"] == [dict(name="null", value="3"), dict(name="on", value="1")]

    def test_a_quote_in_the_image_tag_cannot_inject_pod_spec_keys(self):
        """The image reference is assembled from two free-form values, so it must be quoted as one string."""
        injection = 'v1"\n          securityContext:\n            privileged: true\n          x: "'
        objects = render("--set-string", f"infra.image.tag={injection}")

        assert container(objects)["image"] == f"radixark/miles:{injection}"
        assert "securityContext" not in container(objects)

    def test_a_malformed_service_account_name_is_rejected(self):
        """The account name becomes a Kubernetes object name; catch a bad one before the API server does."""
        assert "serviceAccount/name" in render_error("--set", "serviceAccount.name=Bad_Name")

    def test_pvc_storage_requires_a_claim_name(self):
        """A pvc mount with no claim renders a pod that can never schedule; catch it at install time."""
        assert "pvcClaimName" in render_error("--set", "infra.sharedStorage.type=pvc")

    def test_host_path_storage_requires_a_path(self):
        """A hostPath mount with an empty path is equally unschedulable."""
        assert "hostPath" in render_error("--set", "infra.sharedStorage.hostPath=")

    def test_an_unknown_storage_type_is_rejected(self):
        """Only the storage shapes the templates implement are accepted."""
        assert "infra/sharedStorage/type" in render_error("--set", "infra.sharedStorage.type=nfs")

    def test_non_string_environment_values_are_rejected(self, tmp_path):
        """Kubernetes env values must be strings; a bare number would only fail later, at apply time."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump(dict(infra=dict(env=dict(WORLD_SIZE=8)))))

        assert "/infra/env/WORLD_SIZE" in render_error("-f", str(values_file))
