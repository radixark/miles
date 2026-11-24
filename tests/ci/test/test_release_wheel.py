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
SGLANG_WHEEL = "0.5.22.post62+miles.g6136e6d2bd"


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


# --- the dispatch in release-tag.yml -------------------------------------------------------------------

DISPATCH = _step(
    yaml.safe_load((ROOT / ".github/workflows/release-tag.yml").read_text()),
    "create-tag",
    "Trigger the release-docker and release-wheel builds",
)


@pytest.mark.parametrize("carried", [True, False], ids=["tag-carries-the-workflow", "older-release-line"])
def test_the_wheel_build_is_dispatched_on_the_tag_only_when_the_tag_carries_it(tmp_path, carried):
    if carried:
        (tmp_path / ".github/workflows").mkdir(parents=True)
        (tmp_path / ".github/workflows/release-wheel.yml").touch()
    _fake_gh(tmp_path, "")
    result, _, log = _run(DISPATCH, tmp_path, {"VERSION": "0.3.0"})
    assert result.returncode == 0, result.stderr
    assert log[0] == "workflow run release-docker.yml -f version=0.3.0 -f ref=v0.3.0"
    assert log[1:] == (["workflow run release-wheel.yml --ref v0.3.0"] if carried else [])
    assert ("publishes no pip wheel" in result.stdout) == (not carried)


def test_a_failed_wheel_dispatch_fails_the_tag_workflow(tmp_path):
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/release-wheel.yml").touch()
    _fake_gh(tmp_path, "sys.exit('release-wheel.yml' in args)")
    assert _run(DISPATCH, tmp_path, {"VERSION": "0.3.0"})[0].returncode == 1


# --- the version and the sglang wheel a release stamps ---------------------------------------------------

PINS = next(step["run"] for step in WORKFLOW["jobs"]["pins"]["steps"] if step.get("id") == "pins")


def _tagged_tree(tmp_path):
    """The tag's lock, and a lookup script that answers for its commit without the network."""
    (tmp_path / "release-lock.json").write_text('{"sglang_commit": "abc123"}')
    script = tmp_path / ".github/scripts/release/sglang_wheel_version.py"
    script.parent.mkdir(parents=True)
    script.write_text(f"import sys\nassert sys.argv[1:] == ['abc123']\nprint('{SGLANG_WHEEL}')\n")


def test_the_version_is_the_tag_and_the_sglang_wheel_is_the_locked_commit(tmp_path):
    _tagged_tree(tmp_path)
    result, outputs, _ = _run(PINS, tmp_path, {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.3.0rc1"})
    assert result.returncode == 0, result.stderr
    assert outputs == {"version": "0.3.0rc1", "sglang_wheel": SGLANG_WHEEL}


@pytest.mark.parametrize(
    "ref_type,ref_name", [("branch", "main"), ("branch", "v0.3.0"), ("tag", "sglang-miles-2026w46")]
)
def test_a_dispatch_off_a_release_tag_is_refused(tmp_path, ref_type, ref_name):
    _tagged_tree(tmp_path)
    result, outputs, _ = _run(PINS, tmp_path, {"GITHUB_REF_TYPE": ref_type, "GITHUB_REF_NAME": ref_name})
    assert result.returncode == 1 and "dispatch this on a release tag" in result.stdout
    assert outputs == {}


# --- the release stamp in wheel-smoke.yml -------------------------------------------------------------

STAMP = _step(SMOKE, "wheel-smoke", "Stamp the release version and its sglang pin")


def _release_tree(tmp_path):
    (tmp_path / "setup.py").write_text('setup(\n    name="miles",\n    version="0.1.3",\n)\n')
    (tmp_path / "requirements.txt").write_text("ray")  # no trailing newline


@pytest.mark.parametrize("version", ["0.1.3", "0.1.3rc1", "0.1.3.post2"])
def test_the_exact_version_and_the_sglang_wheel_are_stamped(tmp_path, version):
    _release_tree(tmp_path)
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": version, "SGLANG_WHEEL": SGLANG_WHEEL})
    assert result.returncode == 0, result.stderr
    assert f'version="{version}"' in (tmp_path / "setup.py").read_text()
    assert (tmp_path / "requirements.txt").read_text().split() == ["ray", f"sglang=={SGLANG_WHEEL}"]


@pytest.mark.parametrize(
    "version,sglang_wheel,error",
    [
        ("0.2.0", SGLANG_WHEEL, 'does not carry version="0.2.0"'),
        ("main", SGLANG_WHEEL, "Invalid version"),
        ("0.1.3/x", SGLANG_WHEEL, "Invalid version"),
        ("0.1.3", "", "needs the sglang wheel version"),
    ],
    ids=["other-base-version", "not-a-version", "not-a-release-version", "no-sglang-wheel"],
)
def test_a_release_stamp_without_a_valid_version_and_sglang_wheel_is_refused(tmp_path, version, sglang_wheel, error):
    _release_tree(tmp_path)
    result, _, _ = _run(STAMP, tmp_path, {"VERSION": version, "SGLANG_WHEEL": sglang_wheel})
    assert result.returncode == 1 and error in result.stdout
    assert (tmp_path / "requirements.txt").read_text() == "ray"


RESOLVE = _step(SMOKE, "wheel-smoke", "Resolve the release wheel with its sglang pin")


def test_a_release_without_the_pinned_sglang_wheel_to_resolve_against_is_refused(tmp_path):
    _fake_gh(tmp_path, "")  # no matching asset
    result, _, log = _run(RESOLVE, tmp_path, {"SGLANG_WHEEL": SGLANG_WHEEL, "RUNNER_TEMP": str(tmp_path)})
    assert result.returncode == 1 and f"no cp312 x86_64 wheel of sglang {SGLANG_WHEEL}" in result.stdout
    assert log[0].startswith("api repos/radixark/miles-wheels/releases")


# --- publish ------------------------------------------------------------------------------------

EXACT = _step(WORKFLOW, "publish", "The dist carries the exact version")
CREATE = _step(WORKFLOW, "publish", "Create a draft release if the tag has none")
UPLOAD = _step(WORKFLOW, "publish", "Upload the sdist and wheel")


def test_publish_wants_the_wheel_named_after_the_exact_version(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/miles-0.1.3-py3-none-any.whl").touch()
    assert _run(EXACT, tmp_path, {"VERSION": "0.1.3"})[0].returncode == 0
    result, _, _ = _run(EXACT, tmp_path, {"VERSION": "0.1.3rc1"})
    assert result.returncode == 1 and "no miles-0.1.3rc1 wheel" in result.stdout


@pytest.mark.parametrize(
    "version,flags",
    [("0.3.0", ""), ("0.3.0.post1", ""), ("0.3.0rc1", " --prerelease")],
    ids=["final", "post", "rc"],
)
def test_a_missing_release_is_created_as_a_draft_and_only_an_rc_is_a_prerelease(tmp_path, version, flags):
    _fake_gh(tmp_path, "sys.exit(1 if args[:2] == ['release', 'view'] else 0)")
    result, _, log = _run(CREATE, tmp_path, {"VERSION": version, "TAG": f"v{version}"})
    assert result.returncode == 0, result.stderr
    call = f"release create v{version} --repo radixark/miles --verify-tag --draft --title v{version} --notes{flags}"
    assert log[-1].split() == call.split()  # split() drops the empty --notes value


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
