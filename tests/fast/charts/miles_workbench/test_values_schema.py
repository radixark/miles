import pytest
import yaml
from tests.fast.charts.utils import (
    container,
    host_path_volume,
    pod_spec,
    render,
    render_error,
    requires_helm,
    schema_error_mentions,
    single_object_of_kind,
    volumes_args,
)


def _pvc_volume(claim_name: str) -> dict:
    return {
        "name": "cluster-storage",
        "persistentVolumeClaim": {"claimName": claim_name},
        "mounts": [{"mountPath": "/cluster-storage"}],
    }


def _volume_of_kind(objects: list, key: str) -> dict:
    volumes = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"]
    matched = [volume for volume in volumes if key in volume]
    assert len(matched) == 1, f"expected one {key} volume, got {volumes}"
    return matched[0]


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

        assert "claimName" in render_error(*volumes_args(_pvc_volume(long_name)))
        assert schema_error_mentions(
            render_error("--set", f"serviceAccount.name={long_name}"), path=("serviceAccount", "name")
        )

    def test_a_relative_host_path_is_rejected(self):
        """A relative hostPath renders and installs, then leaves the pod stuck on a failed mount."""
        assert "hostPath" in render_error(*volumes_args(host_path_volume(path="relative/dir")))

    @pytest.mark.parametrize("path", ["/../cluster-storage", "/cluster-storage/.."])
    def test_a_host_path_containing_a_backstep_is_rejected(self, path):
        """Kubernetes rejects any ".." segment in a volume path, so catch it at install time."""
        assert "hostPath" in render_error(*volumes_args(host_path_volume(path=path)))

    def test_a_host_path_that_merely_contains_dots_is_accepted(self):
        """Only a whole ".." segment is a backstep; dots inside a name are ordinary characters."""
        objects = render(*volumes_args(host_path_volume(path="/a..b/c")))

        assert _volume_of_kind(objects, "hostPath")["hostPath"]["path"] == "/a..b/c"

    def test_an_environment_name_the_api_forbids_is_rejected(self):
        """Kubernetes allows printable ASCII in an env name except "=", which would also break the rendering."""
        assert "A=B" in render_error("--set-string", "infra.env.A\\=B=value")

    def test_an_unusual_but_legal_environment_name_is_accepted(self):
        """The pattern is the API contract, not the conventional identifier shape, so it must not be stricter."""
        objects = render("--set-string", "infra.env.FOO/BAR=value", "--set-string", "infra.env.FOO\\ BAZ=value")

        assert {"FOO/BAR", "FOO BAZ"} <= {entry["name"] for entry in container(objects)["env"]}

    def test_a_malformed_claim_name_is_rejected(self):
        """A claim name is a Kubernetes object name; "a..b" passes a loose pattern but no real API server."""
        assert "claimName" in render_error(*volumes_args(_pvc_volume("a..b")))

    def test_yaml_boolean_lookalike_names_stay_strings(self):
        """Names like "on" are valid Kubernetes names but YAML booleans, so they must be rendered quoted."""
        objects = render(*volumes_args(_pvc_volume("on")), "--set", "serviceAccount.name=no")
        assert _volume_of_kind(objects, "persistentVolumeClaim")["persistentVolumeClaim"]["claimName"] == "on"
        assert pod_spec(objects)["serviceAccountName"] == "no"

    def test_yaml_boolean_lookalike_environment_names_stay_strings(self, tmp_path):
        """Names like "on" and "null" are legal environment variable names but YAML booleans and nulls."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump({"infra": {"env": {"on": "1", "null": "3"}}}))
        objects = render("-f", str(values_file))

        env = container(objects)["env"]

        assert dict(name="null", value="3") in env
        assert dict(name="on", value="1") in env

    def test_a_quote_in_the_image_tag_cannot_inject_pod_spec_keys(self):
        """The image reference is assembled from two free-form values, so it must be quoted as one string."""
        injection = 'v1"\n          securityContext:\n            privileged: true\n          x: "'
        objects = render("--set-string", f"infra.image.tag={injection}")

        assert container(objects)["image"] == f"radixark/miles:{injection}"
        assert "securityContext" not in container(objects)

    def test_a_malformed_service_account_name_is_rejected(self):
        """The account name becomes a Kubernetes object name; catch a bad one before the API server does."""
        assert schema_error_mentions(
            render_error("--set", "serviceAccount.name=Bad_Name"), path=("serviceAccount", "name")
        )

    def test_a_pvc_volume_requires_a_claim_name(self):
        """A pvc volume with no claim renders a pod that can never schedule; catch it at install time."""
        assert "claimName" in render_error(
            "--set-json", 'infra.volumes=[{"name":"v","persistentVolumeClaim":{},"mounts":[{"mountPath":"/mnt"}]}]'
        )

    def test_a_host_path_volume_requires_a_path(self):
        """A hostPath volume with no path is equally unschedulable."""
        assert "hostPath" in render_error(
            "--set-json", 'infra.volumes=[{"name":"v","hostPath":{},"mounts":[{"mountPath":"/mnt"}]}]'
        )

    def test_a_volume_that_names_two_sources_is_rejected(self):
        """The free volume list has no storage type enum left, so this is all that stands between the two."""
        error = render_error(
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"persistentVolumeClaim":{"claimName":"c"},'
            '"mounts":[{"mountPath":"/mnt"}]}]',
        )

        assert schema_error_mentions(error, path=("infra", "volumes", "0"))

    def test_a_volume_that_names_no_source_is_rejected(self):
        """A volume with no source is a mount kubernetes cannot satisfy, so its pods only ever stay pending."""
        error = render_error("--set-json", 'infra.volumes=[{"name":"v","mounts":[{"mountPath":"/mnt"}]}]')

        assert schema_error_mentions(error, path=("infra", "volumes", "0"))

    def test_a_source_kind_no_template_implements_is_rejected(self):
        """Only the sources the templates render are accepted; another kind would silently mount nothing."""
        error = render_error(
            "--set-json", 'infra.volumes=[{"name":"v","nfs":{"server":"s"},"mounts":[{"mountPath":"/mnt"}]}]'
        )

        assert "nfs" in error

    def test_an_absolute_mount_subpath_is_rejected(self):
        """A subPath is relative to the volume root by construction, and kubelet refuses an absolute one."""
        error = render_error(
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"mounts":[{"mountPath":"/mnt","subPath":"/abs"}]}]',
        )

        assert "subPath" in error

    def test_a_runs_root_containing_a_backstep_is_rejected(self):
        """A ".." segment would let a run write outside the directory the cluster set aside for miles."""
        assert schema_error_mentions(
            render_error("--set", "infra.paths.runsRoot=/cluster-storage/a/../b"), path=("infra", "paths", "runsRoot")
        )

    def test_a_relative_runs_root_is_rejected(self):
        """The launcher hands this path to the pods as it is, so a relative one resolves per working directory."""
        assert schema_error_mentions(
            render_error("--set", "infra.paths.runsRoot=miles_data"), path=("infra", "paths", "runsRoot")
        )

    def test_a_relative_mount_path_is_rejected(self):
        """A container mountPath must be absolute, and a relative one only fails at apply time."""
        error = render_error(
            "--set-json", 'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"mounts":[{"mountPath":"scratch"}]}]'
        )

        assert "mountPath" in error

    def test_non_string_environment_values_are_rejected(self, tmp_path):
        """Kubernetes env values must be strings; a bare number would only fail later, at apply time."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump(dict(infra=dict(env=dict(WORLD_SIZE=8)))))

        assert schema_error_mentions(render_error("-f", str(values_file)), path=("infra", "env", "WORLD_SIZE"))
