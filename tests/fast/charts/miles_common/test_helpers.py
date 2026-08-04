import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fast.charts.utils import SHARED_INFRA_SCHEMA_PATH

LIBRARY_CHART = SHARED_INFRA_SCHEMA_PATH.parent / "miles-common"

CONSUMER_TEMPLATE = """
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "miles-common.componentName" (dict "context" . "component" .Values.component) }}
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
      {{- with include "miles-common.sharedStorageVolumeMount" . }}
      volumeMounts:
        {{- . | nindent 8 }}
      {{- end }}
  {{- with include "miles-common.sharedStorageVolume" . }}
  volumes:
    {{- . | nindent 4 }}
  {{- end }}
"""

DEFAULT_VALUES = {
    "component": "worker",
    "image": {"repository": "registry.local/miles", "tag": "v1"},
    "sharedStorage": {"type": "hostPath", "hostPath": "/cluster-storage", "mountPath": "/cluster-storage"},
    "scheduling": {"nodeSelector": {}, "tolerations": [], "affinity": {}},
    "env": {},
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


def render(consumer: Path, release: str = "rel", *args: str) -> dict[str, Any]:
    result = subprocess.run(
        ["helm", "template", release, str(consumer), "-n", "rl", *args], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return next(document for document in yaml.safe_load_all(result.stdout) if document)


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to render the helpers")
class TestNaming:
    def test_the_release_name_carries_the_chart_name(self, consumer):
        """Helm's own convention, so an object is traceable to the release that made it."""
        assert render(consumer, "rel")["metadata"]["name"] == "rel-consumer-worker"
        assert render(consumer, "rel-consumer")["metadata"]["name"] == "rel-consumer-worker"

    def test_a_component_name_stays_inside_the_label_budget(self, consumer):
        """It becomes a StatefulSet name, whose derived pod and revision labels cap at 63."""
        name = render(consumer, "a" * 53)["metadata"]["name"]

        assert len(name) + len("-0123456789") <= 63

    def test_distinct_components_never_collapse_onto_one_name(self, consumer):
        """Truncating a name that is already at the limit silently renders two workloads as one."""
        long_release = "a" * 53
        first = render(consumer, long_release, "--set", "component=leader")["metadata"]["name"]
        second = render(consumer, long_release, "--set", "component=logger")["metadata"]["name"]

        assert first.endswith("-leader")
        assert second.endswith("-logger")
        assert first != second

    def test_the_labels_are_the_standard_set(self, consumer):
        """Selectors and tooling across the repo key off these, so the set is pinned."""
        labels = render(consumer, "rel")["metadata"]["labels"]

        assert labels == {
            "helm.sh/chart": "consumer-0.1.0",
            "app.kubernetes.io/name": "consumer",
            "app.kubernetes.io/instance": "rel",
            "app.kubernetes.io/component": "worker",
            "app.kubernetes.io/version": "",
            "app.kubernetes.io/managed-by": "Helm",
        }


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to render the helpers")
class TestInfra:
    def test_the_image_is_one_quoted_string(self, consumer):
        """Two free-form values are joined here; unquoted they could inject sibling keys."""
        pod = render(consumer, "rel", "--set-string", 'image.tag=v1"\n          privileged: true')

        assert pod["spec"]["containers"][0]["image"].startswith("registry.local/miles:v1")
        assert "privileged" not in pod["spec"]["containers"][0]

    def test_scheduling_and_pull_secrets_are_omitted_when_empty(self, consumer):
        """An empty nodeSelector rendered as `{}` is a different pod spec from no nodeSelector."""
        spec = render(consumer, "rel")["spec"]

        assert "nodeSelector" not in spec
        assert "tolerations" not in spec
        assert "affinity" not in spec
        assert "imagePullSecrets" not in spec

    def test_scheduling_and_pull_secrets_pass_through(self, consumer):
        """The whole point of the shared section is that every chart renders it the same way."""
        spec = render(
            consumer,
            "rel",
            "--set",
            "scheduling.nodeSelector.pool=cpu",
            "--set-json",
            'scheduling.tolerations=[{"key":"gpu","operator":"Exists"}]',
            "--set",
            "image.pullSecrets[0]=cred",
        )["spec"]

        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert spec["imagePullSecrets"] == [{"name": "cred"}]

    def test_environment_names_and_values_stay_strings(self, consumer, tmp_path):
        """Names like "on" are legal environment variables but YAML booleans."""
        values = tmp_path / "env.yaml"
        values.write_text(yaml.safe_dump({"env": {"on": "1", "HTTP_PROXY": "http://p:1"}}))
        container = render(consumer, "rel", "-f", str(values))["spec"]["containers"][0]

        assert container["env"] == [{"name": "HTTP_PROXY", "value": "http://p:1"}, {"name": "on", "value": "1"}]

    def test_host_path_storage_renders_a_matched_volume_and_mount(self, consumer):
        """A mount naming a volume that is not there is a pod that never starts."""
        spec = render(consumer, "rel", "--set", "sharedStorage.hostPath=/gpfs")["spec"]

        assert spec["volumes"] == [{"name": "shared-storage", "hostPath": {"path": "/gpfs", "type": "Directory"}}]
        assert spec["containers"][0]["volumeMounts"] == [{"name": "shared-storage", "mountPath": "/cluster-storage"}]

    def test_pvc_storage_binds_the_named_claim(self, consumer):
        """Clusters without host mounts point the same mount at a pre-existing claim."""
        spec = render(consumer, "rel", "--set", "sharedStorage.type=pvc", "--set", "sharedStorage.pvcClaimName=c1")[
            "spec"
        ]

        assert spec["volumes"] == [{"name": "shared-storage", "persistentVolumeClaim": {"claimName": "c1"}}]

    def test_storage_type_none_renders_neither_volume_nor_mount(self, consumer):
        """Disabling storage must not leave a mount referencing a volume that no longer exists."""
        spec = render(consumer, "rel", "--set", "sharedStorage.type=none")["spec"]

        assert "volumes" not in spec
        assert "volumeMounts" not in spec["containers"][0]

    def test_a_section_the_user_blanked_out_does_not_crash_the_render(self, consumer):
        """A values file with a bare `scheduling:` header deletes the chart default; helm keeps the null."""
        for section in ("scheduling", "image", "sharedStorage", "env"):
            pod = render(consumer, "rel", "--set", f"{section}=null")

            assert pod["kind"] == "Pod"


class TestContract:
    def test_the_helpers_cover_exactly_the_shared_sections(self):
        """A section in the schema with no helper behind it is never rendered into any pod."""
        shared = set(json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"])
        rendered = {
            section
            for section in shared
            if f".Values.{section}" in (LIBRARY_CHART / "templates" / "_infra.tpl").read_text()
        }

        assert shared == {"image", "sharedStorage", "scheduling", "env"}
        assert rendered == shared
