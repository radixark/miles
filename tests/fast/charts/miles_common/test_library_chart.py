import shutil
import subprocess

import yaml

from tests.fast.charts.utils import (
    CHART_DIR,
    NAMESPACE,
    REPO_ROOT,
    chart_directories,
    container,
    library_chart_directories,
    pod_spec,
    render,
    requires_helm,
)


class TestLibraryChart:
    def test_every_application_chart_depends_on_the_library_chart(self):
        """The shared helpers only stay shared if every chart actually pulls them in."""
        libraries = {chart_dir.name for chart_dir in library_chart_directories()}

        assert libraries == {"miles-common"}
        for chart_dir in chart_directories():
            dependencies = yaml.safe_load((chart_dir / "Chart.yaml").read_text()).get("dependencies", [])
            assert libraries <= {dependency["name"] for dependency in dependencies}, chart_dir

    def test_the_dependency_is_pinned_by_a_committed_lock(self):
        """A lock file is what makes `helm dependency build` reproducible offline."""
        for chart_dir in chart_directories():
            lock = yaml.safe_load((chart_dir / "Chart.lock").read_text())
            assert [dependency["name"] for dependency in lock["dependencies"]] == ["miles-common"], chart_dir

    @requires_helm
    def test_the_shared_helpers_render_the_helm_values(self):
        """The library chart owns image, scheduling, env and storage rendering for every chart."""
        objects = render(
            "--set",
            "infra.image.repository=registry.local/miles",
            "--set",
            "infra.image.tag=v1",
            "--set",
            "infra.image.pullSecrets[0]=cred",
            "--set",
            "infra.scheduling.nodeSelector.pool=cpu",
            "--set",
            "infra.env.HTTP_PROXY=http://proxy:7890",
        )
        spec = pod_spec(objects)

        assert container(objects)["image"] == "registry.local/miles:v1"
        assert spec["imagePullSecrets"] == [dict(name="cred")]
        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert dict(name="HTTP_PROXY", value="http://proxy:7890") in container(objects)["env"]

    @requires_helm
    def test_the_chart_still_renders_as_a_subchart(self, tmp_path):
        """Helm injects `global` and a per-dependency section into subchart values; a strict root must allow them."""
        umbrella = tmp_path / "umbrella"
        (umbrella / "charts").mkdir(parents=True)
        (umbrella / "Chart.yaml").write_text(
            yaml.safe_dump(
                dict(
                    apiVersion="v2",
                    name="umbrella",
                    version="0.1.0",
                    dependencies=[dict(name="miles-workbench", version="0.1.0", repository="")],
                )
            )
        )
        shutil.copytree(CHART_DIR, umbrella / "charts" / "miles-workbench")
        shutil.copytree(REPO_ROOT / "charts" / "miles-common", umbrella / "charts" / "miles-common")
        (umbrella / "values.yaml").write_text(yaml.safe_dump({"global": {"imageRegistry": "registry.local"}}))

        result = subprocess.run(
            ["helm", "template", "myrel", str(umbrella), "-n", NAMESPACE], capture_output=True, text=True
        )

        assert result.returncode == 0, result.stderr
        assert "kind: StatefulSet" in result.stdout
