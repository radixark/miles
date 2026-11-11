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


# --- the release stamp in wheel-smoke.yml -------------------------------------------------------------

STAMP = _step(SMOKE, "wheel-smoke", "Stamp the release version and its sglang pin")
SGLANG_WHEEL = "0.5.22.post62+miles.g6136e6d2bd"


def _release_tree(tmp_path):
    (tmp_path / "setup.py").write_text('setup(\n    name="miles",\n    version="0.1.3",\n)\n')
    (tmp_path / "requirements.txt").write_text("ray")  # no trailing newline
    (tmp_path / "release-lock.json").write_text(f'{{"sglang_wheel": "{SGLANG_WHEEL}"}}')


@pytest.mark.parametrize("version", ["0.1.3", "0.1.3rc1", "0.1.3.post2"])
def test_the_exact_version_and_the_locked_sglang_wheel_are_stamped(tmp_path, version):
    _release_tree(tmp_path)
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": version})
    assert result.returncode == 0, result.stderr
    assert f'version="{version}"' in (tmp_path / "setup.py").read_text()
    assert (tmp_path / "requirements.txt").read_text().split() == ["ray", f"sglang=={SGLANG_WHEEL}"]


@pytest.mark.parametrize(
    "version,error",
    [("0.2.0", 'does not carry version="0.2.0"'), ("main", "Invalid version"), ("0.1.3/x", "Invalid version")],
)
def test_a_version_setup_py_does_not_carry_or_that_is_no_release_version_is_refused(tmp_path, version, error):
    _release_tree(tmp_path)
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": version})
    assert result.returncode == 1 and error in result.stdout
    assert 'version="0.1.3"' in (tmp_path / "setup.py").read_text()
    assert (tmp_path / "requirements.txt").read_text() == "ray"


def test_a_branch_whose_lock_names_no_sglang_wheel_gets_no_release_wheel(tmp_path):
    _release_tree(tmp_path)
    (tmp_path / "release-lock.json").write_text("{}")
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": "0.1.3"})
    assert result.returncode == 1 and "sglang_wheel" in result.stderr


# --- publish ------------------------------------------------------------------------------------

EXACT = _step(WORKFLOW, "publish", "The dist carries the exact version")
CREATE = _step(WORKFLOW, "publish", "Create the release if the tag has none")
UPLOAD = _step(WORKFLOW, "publish", "Upload the sdist and wheel")


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
