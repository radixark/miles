from pathlib import Path
from typing import Any

import yaml
from tests.fast.charts.utils import (
    SHARED_INFRA_SCHEMA_PATH,
    chart_directories,
    container,
    only_container_of,
    pod_spec,
    pod_spec_of,
    render,
    render_run,
    requires_helm,
    resolved_schema,
)

CLUSTER_VALUES = dict(
    infra=dict(
        image=dict(repository="registry.local/miles", tag="v1"),
        volumes=[
            dict(
                name="cluster-storage",
                persistentVolumeClaim=dict(claimName="shared"),
                mounts=[
                    dict(mountPath="/cluster-storage"),
                    dict(mountPath="/root/miles", subPath="myuser/miles"),
                ],
            ),
            dict(
                name="scratch",
                hostPath=dict(path="/local", type="DirectoryOrCreate"),
                mounts=[dict(mountPath="/scratch")],
            ),
        ],
        paths=dict(runsRoot="/cluster-storage/teamdata"),
        scheduling=dict(nodeSelector={"pool": "cpu"}),
        env={"HF_ENDPOINT": "https://mirror"},
    )
)

MILES_CODE_MOUNT = {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "myuser/miles"}


def shared_infra_schema() -> dict[str, Any]:
    return resolved_schema(SHARED_INFRA_SCHEMA_PATH)["properties"]["infra"]


def cluster_values_file(tmp_path: Path) -> Path:
    values_file = tmp_path / "cluster.yaml"
    values_file.write_text(yaml.safe_dump(CLUSTER_VALUES))
    return values_file


def chart_infra_defaults(chart_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((chart_dir / "values.yaml").read_text())["infra"]


class TestSharedInfraContract:
    def test_the_contract_is_the_single_infra_subtree(self):
        """One top-level key is what lets a cluster values file be handed to any chart unchanged."""
        assert set(resolved_schema(SHARED_INFRA_SCHEMA_PATH)["properties"]) == {"infra"}

    def test_the_infra_subtree_is_exactly_the_cluster_shaped_sections(self):
        """A section with no helper behind it would be accepted by the schema and never reach a pod."""
        assert set(shared_infra_schema()["properties"]) == {
            "image",
            "volumes",
            "paths",
            "devShm",
            "scheduling",
            "env",
        }

    def test_every_chart_inlines_the_shared_infra_schema_verbatim(self):
        """Helm cannot $ref across files, so every chart carries its own copy of the same contract."""
        shared = shared_infra_schema()

        for chart_dir in chart_directories():
            properties = resolved_schema(chart_dir / "values.schema.json")["properties"]
            assert properties["infra"] == shared, chart_dir

    def test_every_chart_defaults_every_shared_infra_key(self):
        """A shared values file is only partly honoured by a chart that leaves one of the keys undefaulted."""
        sections = set(shared_infra_schema()["properties"])

        for chart_dir in chart_directories():
            assert sections <= set(chart_infra_defaults(chart_dir)), chart_dir

    def test_every_chart_ships_the_files_the_contract_is_pinned_through(self):
        """A chart shipping no values.schema.json would silently escape the two assertions above."""
        for chart_dir in chart_directories():
            assert (chart_dir / "values.schema.json").is_file(), chart_dir
            assert (chart_dir / "values.yaml").is_file(), chart_dir

    def test_no_chart_leaves_a_shared_key_undefined(self):
        """A chart that defaults a shared section to null accepts the file and then renders nothing from it."""
        sections = set(shared_infra_schema()["properties"])

        for chart_dir in chart_directories():
            defaults = chart_infra_defaults(chart_dir)
            for section in sections:
                assert defaults[section] is not None, (chart_dir, section)

    @requires_helm
    def test_one_infra_nested_cluster_file_renders_the_workbench_chart(self, tmp_path):
        """The infra section alone must drive the chart, or a cluster file would need per-chart edits."""
        objects = render("-f", str(cluster_values_file(tmp_path)))

        assert container(objects)["image"] == "registry.local/miles:v1"
        assert pod_spec(objects)["nodeSelector"] == {"pool": "cpu"}
        assert MILES_CODE_MOUNT in container(objects)["volumeMounts"]

    @requires_helm
    def test_the_very_same_cluster_file_renders_the_run_chart(self, tmp_path):
        """Two charts reading the same file differently is exactly what the shared schema exists to prevent."""
        objects = render_run("-f", str(cluster_values_file(tmp_path)))
        orchestrator = only_container_of(objects, "StatefulSet", "myrun-miles-run-orchestrator")

        assert orchestrator["image"] == "registry.local/miles:v1"
        assert pod_spec_of(objects, "StatefulSet", "myrun-miles-run-orchestrator")["nodeSelector"] == {"pool": "cpu"}
        assert MILES_CODE_MOUNT in orchestrator["volumeMounts"]
