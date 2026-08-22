import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fast.charts.utils import NAMESPACE, SHARED_INFRA_SCHEMA_PATH, host_path_volume, volumes_args

CHARTS_DIR = SHARED_INFRA_SCHEMA_PATH.parent
LIBRARY_CHART = CHARTS_DIR / "miles-common"
RELEASE_NAME = "myrel"

CONSUMER_TEMPLATE = """
apiVersion: v1
kind: Pod
metadata:
  name: consumer
  labels:
    {{- include "miles-common.labels" (dict "context" . "component" .Values.component) | nindent 4 }}
spec:
  {{- with include "miles-common.imagePullSecrets" . }}
  {{- . | nindent 2 }}
  {{- end }}
  {{- with include "miles-common.scheduling" . }}
  {{- . | nindent 2 }}
  {{- end }}
  containers:
    - name: main
      image: {{ include "miles-common.image" . }}
      {{- with include "miles-common.env" . }}
      {{- . | nindent 6 }}
      {{- end }}
      {{- with include "miles-common.volumeMounts" . | trim }}
      volumeMounts:
        {{- . | nindent 8 }}
      {{- end }}
  {{- with include "miles-common.volumes" . | trim }}
  volumes:
    {{- . | nindent 4 }}
  {{- end }}
"""

DEFAULT_VALUES = {
    "component": "worker",
    "infra": {
        "image": {"repository": "registry.local/miles", "tag": "v1"},
        "volumes": [
            {
                "name": "cluster-storage",
                "hostPath": {"path": "/cluster-storage", "type": "Directory"},
                "mounts": [{"mountPath": "/cluster-storage"}],
            }
        ],
        "paths": {"runsRoot": "/cluster-storage/miles_data"},
        "devShm": {"mountPath": "/dev/shm", "hostPath": {"path": "/dev/shm", "type": "Directory"}},
        "scheduling": {"nodeSelector": {}, "tolerations": [], "affinity": {}},
        "env": {},
    },
}


@pytest.fixture(scope="module")
def consumer(tmp_path_factory) -> Path:
    chart = tmp_path_factory.mktemp("consumer") / "consumer"
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text(
        "apiVersion: v2\nname: consumer\nversion: 0.1.0\n"
        f'dependencies:\n  - name: miles-common\n    version: 0.1.0\n    repository: "file://{LIBRARY_CHART}"\n'
    )
    (chart / "values.yaml").write_text(yaml.safe_dump(DEFAULT_VALUES))
    (chart / "templates" / "pod.yaml").write_text(CONSUMER_TEMPLATE)
    subprocess.run(["helm", "dependency", "update", str(chart)], capture_output=True, check=True)
    return chart


def render(consumer: Path, release: str = RELEASE_NAME, *args: str) -> dict[str, Any]:
    result = _template(consumer, release, *args)
    assert result.returncode == 0, result.stderr
    return next(document for document in yaml.safe_load_all(result.stdout) if document)


def render_error(consumer: Path, release: str = RELEASE_NAME, *args: str) -> str:
    result = _template(consumer, release, *args)
    assert result.returncode != 0, result.stdout
    return result.stderr


def _template(consumer: Path, release: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["helm", "template", release, str(consumer), "-n", NAMESPACE, *args], capture_output=True, text=True
    )


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to render the helpers")
class TestNaming:
    def test_the_labels_are_the_standard_set(self, consumer):
        """Selectors and tooling across the repo key off these, so the set is pinned."""
        labels = render(consumer, RELEASE_NAME)["metadata"]["labels"]

        assert labels == {
            "helm.sh/chart": "consumer-0.1.0",
            "app.kubernetes.io/name": "consumer",
            "app.kubernetes.io/instance": RELEASE_NAME,
            "app.kubernetes.io/component": "worker",
            "app.kubernetes.io/version": "",
            "app.kubernetes.io/managed-by": "Helm",
        }


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to render the helpers")
class TestInfra:
    def test_the_image_is_one_quoted_string(self, consumer):
        """Two free-form values are joined here; unquoted they could inject sibling keys."""
        pod = render(consumer, RELEASE_NAME, "--set-string", 'infra.image.tag=v1"\n          privileged: true')

        assert pod["spec"]["containers"][0]["image"].startswith("registry.local/miles:v1")
        assert "privileged" not in pod["spec"]["containers"][0]

    def test_scheduling_and_pull_secrets_are_omitted_when_empty(self, consumer):
        """An empty nodeSelector rendered as `{}` is a different pod spec from no nodeSelector."""
        spec = render(consumer, RELEASE_NAME)["spec"]

        assert "nodeSelector" not in spec
        assert "tolerations" not in spec
        assert "affinity" not in spec
        assert "imagePullSecrets" not in spec

    def test_scheduling_and_pull_secrets_pass_through(self, consumer):
        """The whole point of the shared section is that every chart renders it the same way."""
        spec = render(
            consumer,
            RELEASE_NAME,
            "--set",
            "infra.scheduling.nodeSelector.pool=cpu",
            "--set-json",
            'infra.scheduling.tolerations=[{"key":"gpu","operator":"Exists"}]',
            "--set",
            "infra.image.pullSecrets[0]=cred",
        )["spec"]

        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert spec["imagePullSecrets"] == [{"name": "cred"}]

    def test_environment_names_and_values_stay_strings(self, consumer, tmp_path):
        """Names like "on" are legal environment variables but YAML booleans."""
        values = tmp_path / "env.yaml"
        values.write_text(yaml.safe_dump({"infra": {"env": {"on": "1", "HTTP_PROXY": "http://p:1"}}}))
        container = render(consumer, RELEASE_NAME, "-f", str(values))["spec"]["containers"][0]

        assert container["env"] == [{"name": "HTTP_PROXY", "value": "http://p:1"}, {"name": "on", "value": "1"}]

    def test_a_host_path_volume_renders_a_matched_volume_and_mount(self, consumer):
        """A mount naming a volume that is not there is a pod that never starts."""
        spec = render(consumer, RELEASE_NAME, *volumes_args(host_path_volume(path="/gpfs")))["spec"]

        assert spec["volumes"] == [{"name": "cluster-storage", "hostPath": {"path": "/gpfs", "type": "Directory"}}]
        assert spec["containers"][0]["volumeMounts"] == [{"name": "cluster-storage", "mountPath": "/cluster-storage"}]

    def test_a_host_path_volume_defaults_to_a_directory_that_has_to_exist(self, consumer):
        """DirectoryOrCreate on a mistyped path silently makes an empty directory instead of failing the pod."""
        spec = render(
            consumer,
            RELEASE_NAME,
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/gpfs"},"mounts":[{"mountPath":"/mnt"}]}]',
        )["spec"]

        assert spec["volumes"] == [{"name": "v", "hostPath": {"path": "/gpfs", "type": "Directory"}}]

    def test_a_host_path_volume_can_ask_for_the_directory_to_be_created(self, consumer):
        """A node-local scratch directory is per node and per namespace, so nobody creates it up front."""
        spec = render(
            consumer,
            RELEASE_NAME,
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/data/x","type":"DirectoryOrCreate"},'
            '"mounts":[{"mountPath":"/scratch"}]}]',
        )["spec"]

        assert spec["volumes"] == [{"name": "v", "hostPath": {"path": "/data/x", "type": "DirectoryOrCreate"}}]

    def test_a_pvc_volume_binds_the_named_claim(self, consumer):
        """Clusters without host mounts point the same mount at a pre-existing claim."""
        spec = render(
            consumer,
            RELEASE_NAME,
            "--set-json",
            'infra.volumes=[{"name":"cluster-storage","persistentVolumeClaim":{"claimName":"c1"},'
            '"mounts":[{"mountPath":"/cluster-storage"}]}]',
        )["spec"]

        assert spec["volumes"] == [{"name": "cluster-storage", "persistentVolumeClaim": {"claimName": "c1"}}]

    def test_an_empty_dir_volume_needs_no_source_configuration_of_its_own(self, consumer):
        """An ephemeral scratch volume is the one source whose whole configuration may be the empty object."""
        spec = render(
            consumer,
            RELEASE_NAME,
            "--set-json",
            'infra.volumes=[{"name":"v","emptyDir":{"medium":"Memory","sizeLimit":"8Gi"},'
            '"mounts":[{"mountPath":"/scratch"}]}]',
        )["spec"]

        assert spec["volumes"] == [{"name": "v", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}]

    def test_an_empty_volume_list_renders_neither_volume_nor_mount(self, consumer):
        """Declaring no storage must not leave a mount referencing a volume that was never rendered."""
        spec = render(consumer, RELEASE_NAME, "--set-json", "infra.volumes=[]")["spec"]

        assert "volumes" not in spec
        assert "volumeMounts" not in spec["containers"][0]

    def test_a_read_only_mount_reaches_the_container_as_read_only(self, consumer):
        """A shared model cache is everyone's, and a run that can write it can corrupt every other run."""
        spec = render(
            consumer,
            RELEASE_NAME,
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/models", "readOnly": True}])),
        )["spec"]

        assert spec["containers"][0]["volumeMounts"] == [
            {"name": "cluster-storage", "mountPath": "/models", "readOnly": True}
        ]

    def test_a_mount_that_is_not_read_only_says_nothing_about_it(self, consumer):
        """readOnly: false is the kubernetes default, and spelling it out only makes every diff noisier."""
        spec = render(consumer, RELEASE_NAME, *volumes_args(host_path_volume()))["spec"]

        assert spec["containers"][0]["volumeMounts"] == [{"name": "cluster-storage", "mountPath": "/cluster-storage"}]

    def test_a_mount_that_names_a_subpath_reaches_the_container_with_it(self, consumer):
        """Mounting a checkout over the image's own path is the whole point of subPath, so it must survive."""
        spec = render(
            consumer,
            RELEASE_NAME,
            *volumes_args(
                host_path_volume(mounts=[{"mountPath": "/sgl-workspace/sglang", "subPath": "myuser/sglang"}])
            ),
        )["spec"]

        assert spec["containers"][0]["volumeMounts"][-1] == {
            "name": "cluster-storage",
            "mountPath": "/sgl-workspace/sglang",
            "subPath": "myuser/sglang",
        }

    def test_a_release_that_names_no_cluster_environment_renders_no_env_block(self, consumer):
        """An empty env: list is not the same shape as no env at all, and pods differ on which they accept."""
        spec = render(consumer, RELEASE_NAME, *volumes_args(host_path_volume()))["spec"]

        assert "env" not in spec["containers"][0]

    def test_a_section_the_user_blanked_out_does_not_crash_the_render(self, consumer):
        """A values file with a bare `scheduling:` header deletes the chart default; helm keeps the null."""
        for section in ("scheduling", "image", "volumes", "env", "paths"):
            pod = render(consumer, RELEASE_NAME, "--set", f"infra.{section}=null")

            assert pod["kind"] == "Pod"


class TestContract:
    def test_every_shared_infra_section_is_rendered_by_some_chart_template(self):
        """A section in the schema with no template behind it is never rendered into any pod."""
        shared = set(json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]["infra"]["properties"])
        templates = "".join(path.read_text() for path in CHARTS_DIR.glob("*/templates/*"))
        rendered = {section for section in shared if f".Values.infra.{section}" in templates}

        assert rendered == shared

    def test_the_library_chart_owns_every_section_but_the_shared_memory_one(self):
        """Only miles-run has pods that run a collective, so that one helper stays with the run chart."""
        library = (LIBRARY_CHART / "templates" / "_infra.tpl").read_text()
        shared = set(json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]["infra"]["properties"])

        assert {section for section in shared if f".Values.infra.{section}" in library} == shared - {"devShm"}
