import json
import subprocess

from tests.fast.charts.utils import SHARED_INFRA_SCHEMA_PATH, library_chart_directories, requires_helm


class TestLibraryChart:
    def test_it_is_the_only_library_chart_and_lints(self):
        """It ships no templates of its own, so linting is the only thing that can go wrong in isolation."""
        chart_dirs = library_chart_directories()

        assert [chart_dir.name for chart_dir in chart_dirs] == ["miles-common"]

    @requires_helm
    def test_helm_accepts_it(self):
        """A library chart still has to be a valid chart."""
        for chart_dir in library_chart_directories():
            result = subprocess.run(["helm", "lint", str(chart_dir)], capture_output=True, text=True)

            assert result.returncode == 0, result.stdout + result.stderr

    def test_the_shared_contract_covers_the_sections_the_helpers_render(self):
        """The helpers read exactly these four sections; a fifth one would have no schema behind it."""
        shared = json.loads(SHARED_INFRA_SCHEMA_PATH.read_text())["properties"]["infra"]["properties"]

        assert set(shared) == {"image", "sharedStorage", "scheduling", "env"}
