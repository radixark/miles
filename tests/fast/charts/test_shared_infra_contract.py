import json

import yaml

from tests.fast.charts.utils import (
    SHARED_INFRA_SCHEMA_PATH,
    chart_directories,
    container,
    pod_spec,
    render,
    requires_helm,
)


class TestSharedInfraContract:
    def test_the_contract_is_exactly_the_four_cluster_shaped_sections(self):
        """A fifth section would have no helper behind it and would never reach a pod."""
        shared = json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]

        assert set(shared) == {"image", "sharedStorage", "scheduling", "env"}

    def test_every_chart_inlines_the_shared_infra_schema_verbatim(self):
        """One cluster values file must fit every Miles chart, so no chart may let the shared keys drift."""
        shared = json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]

        for chart_dir in chart_directories():
            properties = json.loads((chart_dir / "values.schema.json").read_text())["properties"]
            assert {key: properties.get(key) for key in shared} == shared, chart_dir

    def test_every_chart_defaults_every_shared_infra_key(self):
        """A shared values file is only partly honoured by a chart that leaves one of the keys undefaulted."""
        shared = json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]

        for chart_dir in chart_directories():
            defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
            assert set(shared) <= set(defaults), chart_dir

    def test_every_chart_ships_the_files_the_contract_is_pinned_through(self):
        """A chart shipping no values.schema.json would silently escape the two assertions above."""
        for chart_dir in chart_directories():
            assert (chart_dir / "values.schema.json").is_file(), chart_dir
            assert (chart_dir / "values.yaml").is_file(), chart_dir

    def test_no_chart_leaves_a_shared_key_undefined(self):
        """A chart that defaults a shared section to null accepts the file and then renders nothing from it."""
        shared = json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]

        for chart_dir in chart_directories():
            defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
            for key in shared:
                assert defaults[key] is not None, (chart_dir, key)

    @requires_helm
    def test_a_shared_cluster_values_file_renders_the_chart(self, tmp_path):
        """The shared sections alone must drive this chart, so the same file can be passed to every Miles chart."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(
            yaml.safe_dump(
                dict(
                    image=dict(repository="registry.local/miles", tag="v1"),
                    sharedStorage=dict(type="pvc", pvcClaimName="shared", mountPath="/cluster-storage"),
                    scheduling=dict(nodeSelector={"pool": "cpu"}),
                    env={"HF_ENDPOINT": "https://mirror"},
                )
            )
        )
        objects = render("-f", str(values_file))

        assert container(objects)["image"] == "registry.local/miles:v1"
        assert pod_spec(objects)["nodeSelector"] == {"pool": "cpu"}
