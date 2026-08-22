import importlib.util
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.fast.source_scan import REPO_ROOT

LINT_CHARTS_PATH = REPO_ROOT / "tools" / "kubernetes" / "lint_charts.py"

requires_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to lint charts")


def load_lint_charts():
    spec = importlib.util.spec_from_file_location("lint_charts", LINT_CHARTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lint_charts(tmp_path, monkeypatch):
    module = load_lint_charts()
    monkeypatch.setattr(module, "CHARTS_DIR", tmp_path)
    monkeypatch.delenv("CI", raising=False)
    return module


def write_chart(root: Path, name: str, *, library: bool = False, template: str = "") -> Path:
    chart = root / name
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text(
        textwrap.dedent(
            f"""
            apiVersion: v2
            name: {name}
            version: 0.1.0
            {"type: library" if library else ""}
            """
        ).strip()
        + "\n"
    )
    (chart / "values.yaml").write_text("{}\n")
    if template:
        (chart / "templates" / "object.yaml").write_text(template)
    return chart


VALID_TEMPLATE = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {{ .Release.Name }}\n"
BROKEN_TEMPLATE = "apiVersion: v1\nkind: ConfigMap\n  name: broken: [\n"


class TestChartDiscovery:
    def test_library_charts_are_linted_too(self, lint_charts, tmp_path):
        """The hook promises to lint every chart, and a library chart can still fail to parse."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        write_chart(tmp_path, "lib", library=True)

        assert [chart.name for chart in lint_charts.all_charts()] == ["app", "lib"]

    def test_it_looks_in_the_repository_charts_directory(self):
        """Every other test points the constant at a tmp_path, so nothing else would notice a typo here."""
        module = load_lint_charts()

        assert (module.REPO_ROOT / "pyproject.toml").exists()
        assert module.CHARTS_DIR == REPO_ROOT / "charts"

    def test_a_repo_with_no_charts_is_not_a_failure(self, lint_charts):
        """The hook lands before the first chart, and an empty repo must not block every commit."""
        assert lint_charts.all_charts() == []
        assert lint_charts.main([]) == 0


@requires_helm
class TestLinting:
    def test_a_valid_chart_passes(self, lint_charts, tmp_path):
        """The baseline: a chart helm accepts must not be reported as broken."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)

        assert lint_charts.main([]) == 0

    def test_a_broken_chart_fails(self, lint_charts, tmp_path):
        """A template helm cannot parse is exactly what this hook exists to catch."""
        write_chart(tmp_path, "app", template=BROKEN_TEMPLATE)

        assert lint_charts.main([]) == 1

    def test_every_chart_is_linted_even_after_one_fails(self, lint_charts, tmp_path, capsys):
        """Stopping at the first failure costs the contributor one round trip per broken chart."""
        write_chart(tmp_path, "a-broken", template=BROKEN_TEMPLATE)
        write_chart(tmp_path, "z-also-broken", template=BROKEN_TEMPLATE)

        assert lint_charts.main([]) == 1
        errors = capsys.readouterr().err

        assert "a-broken" in errors
        assert "z-also-broken" in errors

    def test_a_chart_whose_dependencies_cannot_be_vendored_fails(self, lint_charts, tmp_path):
        """helm lint only warns about a missing dependency, so the build step is the only guard."""
        app = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (app / "Chart.yaml").write_text(
            "apiVersion: v2\nname: app\nversion: 0.1.0\n"
            'dependencies:\n  - name: absent\n    version: 0.1.0\n    repository: "file://../absent"\n'
        )
        (app / "Chart.lock").write_text(
            'dependencies:\n- name: absent\n  repository: "file://../absent"\n  version: 0.1.0\n'
            'digest: sha256:0\ngenerated: "2026-01-01T00:00:00Z"\n'
        )

        assert lint_charts.main([]) == 1

    def test_every_variant_of_a_chart_is_linted(self, lint_charts, tmp_path):
        """A chart is only as good as its worst supported value combination."""
        chart = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (chart / "values.schema.json").write_text(
            '{"type": "object", "properties": {"size": {"enum": ["small", "large"]}}}'
        )
        lint_charts.VARIANTS["app"] = [["--set", "size=huge"]]

        assert lint_charts.main([]) == 1

    def test_dependencies_are_vendored_before_linting(self, lint_charts, tmp_path):
        """The vendored copy is gitignored, so a fresh clone has a lock file and nothing else."""
        write_chart(tmp_path, "lib", library=True)
        (tmp_path / "lib" / "templates" / "_helpers.tpl").write_text('{{- define "lib.name" -}}lib{{- end }}\n')
        app = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (app / "Chart.yaml").write_text(
            "apiVersion: v2\nname: app\nversion: 0.1.0\n"
            'dependencies:\n  - name: lib\n    version: 0.1.0\n    repository: "file://../lib"\n'
        )
        subprocess.run(["helm", "dependency", "update", str(app)], capture_output=True, check=True)
        shutil.rmtree(app / "charts")

        assert (app / "Chart.lock").exists()
        assert lint_charts.main([]) == 0
        assert list((app / "charts").glob("lib-*.tgz"))


class TestMissingHelm:
    def test_it_skips_when_helm_is_absent_locally(self, lint_charts, tmp_path, monkeypatch):
        """Contributors without helm should not be blocked by a hook that runs on every commit."""
        write_chart(tmp_path, "app", template=BROKEN_TEMPLATE)
        monkeypatch.setattr(lint_charts.shutil, "which", lambda _: None)

        assert lint_charts.main([]) == 0

    def test_it_fails_when_ci_provides_no_helm(self, lint_charts, tmp_path, monkeypatch):
        """In CI a missing helm means the charts went unchecked, which must not look like success."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        monkeypatch.setattr(lint_charts.shutil, "which", lambda _: None)
        monkeypatch.setenv("CI", "true")

        assert lint_charts.main([]) == 1


def colocate_variant() -> list[str]:
    module = load_lint_charts()
    [variant] = [args for args in module.VARIANTS["miles-run"] if any(a.startswith("run.colocate=") for a in args)]
    return variant


def set_json_value(variant: list[str], prefix: str) -> object:
    [value] = [argument[len(prefix) :] for argument in variant if argument.startswith(prefix)]
    return json.loads(value)


class TestTheColocateVariant:
    def test_names_pools_the_same_variant_renders(self):
        """A pool id no pool carries makes the variant a disaggregated run, and the gated branch goes unlinted."""
        colocate = set_json_value(colocate_variant(), "run.colocate=")
        rendered = {
            entry["name"]
            for key in ("run.inferenceEngines=", "run.trainerEngines=")
            for entry in set_json_value(colocate_variant(), key)
        }

        named = {pool["pool_id"] for pool in colocate["inference_pools"]} | {colocate["trainer_pool_id"]}
        assert named <= rendered, sorted(named - rendered)

    def test_covers_a_pool_narrower_than_a_node(self):
        """Only a sub-node pool renders the base gpu id env, so a whole-node variant never lints that branch."""
        colocate = set_json_value(colocate_variant(), "run.colocate=")

        assert any(
            pool["layout"]["num_gpus_per_inference_pod"] < pool["layout"]["num_gpus_per_node"]
            for pool in colocate["inference_pools"]
        )
