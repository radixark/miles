"""release-wheel.yml publishes exactly the tagged version, once, and marks an rc as a prerelease."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release-wheel.yml").read_text())
SMOKE = yaml.safe_load((ROOT / ".github/workflows/wheel-smoke.yml").read_text())
ACTION = yaml.safe_load((ROOT / ".github/actions/resolve-release-version/action.yml").read_text())


def _step(workflow, job, name):
    return next(step["run"] for step in workflow["jobs"][job]["steps"] if step.get("name") == name)


def _fake_gh(tmp_path, body):
    """A gh on PATH that logs every call to gh.log and runs *body* (python) with `args`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport os, sys\nargs = sys.argv[1:]\n"
        f"open(os.environ['GH_LOG'], 'a').write(' '.join(args) + '\\n')\n{body}\n"
    )
    gh.chmod(0o755)
    return bin_dir


def _run(script, tmp_path, env, cwd=None):
    output = tmp_path / "output"
    output.touch()
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        cwd=cwd or tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "radixark/miles",
            "GH_TOKEN": "x",
            "GH_LOG": str(tmp_path / "gh.log"),
            **env,
        },
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    log = (tmp_path / "gh.log").read_text().splitlines() if (tmp_path / "gh.log").exists() else []
    return result, outputs, log


# --- resolve-release-version ------------------------------------------------------------------

RESOLVE = ACTION["runs"]["steps"][0]["run"]


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"INPUT_VERSION": "0.3.0", "INPUT_REF": ""}, ("0.3.0", "v0.3.0")),
        ({"INPUT_VERSION": "0.3.0.post1", "INPUT_REF": "release/v0.3.0"}, ("0.3.0.post1", "release/v0.3.0")),
        ({"INPUT_VERSION": "", "INPUT_REF": "", "GITHUB_REF_NAME": "v0.3.0rc1"}, ("0.3.0rc1", "v0.3.0rc1")),
    ],
    ids=["dispatch-defaults-to-the-tag", "dispatch-with-ref", "tag-push"],
)
def test_the_version_and_ref_come_from_the_dispatch_or_the_pushed_tag(tmp_path, env, expected):
    result, outputs, _ = _run(RESOLVE, tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert (outputs["version"], outputs["ref"]) == expected


def test_a_version_that_is_not_x_y_z_is_refused(tmp_path):
    result, _, _ = _run(RESOLVE, tmp_path, {"INPUT_VERSION": "main", "INPUT_REF": ""})
    assert result.returncode == 1 and "Invalid version" in result.stdout


# --- the version stamp in wheel-smoke.yml ---------------------------------------------------------

STAMP = _step(SMOKE, "wheel-smoke", "Stamp the release version")


@pytest.mark.parametrize("version", ["0.1.3", "0.1.3rc1", "0.1.3.post2"])
def test_the_exact_version_is_stamped_over_setup_pys_base_version(tmp_path, version):
    (tmp_path / "setup.py").write_text('setup(\n    name="miles",\n    version="0.1.3",\n)\n')
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": version})
    assert result.returncode == 0, result.stderr
    assert f'version="{version}"' in (tmp_path / "setup.py").read_text()


def test_a_tag_whose_base_differs_from_setup_py_is_refused(tmp_path):
    (tmp_path / "setup.py").write_text('setup(version="0.1.3")\n')
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": "0.2.0"})
    assert result.returncode == 1 and 'does not carry version="0.2.0"' in result.stdout


# --- publish ------------------------------------------------------------------------------------

EXACT = _step(WORKFLOW, "publish", "The dist carries the exact version")
CREATE = _step(WORKFLOW, "publish", "Create the release if the tag has none")
UPLOAD = _step(WORKFLOW, "publish", "Upload the sdist and wheel")
HARNESS = _step(WORKFLOW, "resolve", "The ref carries the wheel harness")


def test_publish_wants_the_wheel_named_after_the_exact_version(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/miles-0.1.3-py3-none-any.whl").touch()
    assert _run(EXACT, tmp_path, {"VERSION": "0.1.3"})[0].returncode == 0
    result, _, _ = _run(EXACT, tmp_path, {"VERSION": "0.1.3rc1"})
    assert result.returncode == 1 and "no miles-0.1.3rc1 wheel" in result.stdout


@pytest.mark.parametrize(
    "version,flags",
    [("0.3.0", ""), ("0.3.0.post1", ""), ("0.3.0rc1", " --prerelease --latest=false")],
    ids=["final", "post", "rc"],
)
def test_a_missing_release_is_created_and_only_an_rc_is_a_prerelease(tmp_path, version, flags):
    _fake_gh(tmp_path, "sys.exit(1 if args[:2] == ['release', 'view'] else 0)")
    result, _, log = _run(CREATE, tmp_path, {"VERSION": version, "TAG": f"v{version}"})
    assert result.returncode == 0, result.stderr
    assert log[-1] == (
        f"release create v{version} --repo radixark/miles --verify-tag --title v{version} --generate-notes{flags}"
    )


def test_an_existing_release_is_left_alone(tmp_path):
    _fake_gh(tmp_path, "")
    result, _, log = _run(CREATE, tmp_path, {"VERSION": "0.3.0", "TAG": "v0.3.0"})
    assert result.returncode == 0 and log == ["release view v0.3.0 --repo radixark/miles"]


def test_a_published_file_is_never_replaced(tmp_path):
    (tmp_path / "dist").mkdir()
    for name in ("miles-0.3.0-py3-none-any.whl", "miles-0.3.0.tar.gz"):
        (tmp_path / "dist" / name).touch()
    _fake_gh(tmp_path, "if args[:2] == ['release', 'view']: print('miles-0.3.0-py3-none-any.whl')")
    result, _, log = _run(UPLOAD, tmp_path, {"VERSION": "0.3.0", "TAG": "v0.3.0"})
    assert result.returncode == 0, result.stderr
    assert "keeping published miles-0.3.0-py3-none-any.whl" in result.stdout
    assert [call for call in log if call.startswith("release upload")] == [
        "release upload v0.3.0 dist/miles-0.3.0.tar.gz --repo radixark/miles"
    ]


@pytest.mark.parametrize("present", [True, False])
def test_a_ref_without_the_smoke_harness_is_refused_before_anything_runs(tmp_path, present):
    _fake_gh(tmp_path, f"sys.exit({0 if present else 1})")
    result, _, log = _run(HARNESS, tmp_path, {"REF": "v0.1.3"})
    assert result.returncode == (0 if present else 1)
    assert log == ["api repos/radixark/miles/contents/tests/ci/wheel_smoke.py?ref=v0.1.3 --silent"]
    if not present:
        assert "predates the pure wheel" in result.stdout
