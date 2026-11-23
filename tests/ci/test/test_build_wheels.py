import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
# the week's history release check reads, as `date -u +%Gw%V` names it
_YEAR, _WEEK, _ = datetime.datetime.now(datetime.timezone.utc).isocalendar()
SGLANG_RELEASE = f"sglang-miles-{_YEAR}w{_WEEK:02d}"
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/build-wheels.yml").read_text())
# The check job reads its release tags from the Dockerfile.
WHEELS_TAG_X86 = re.search(r"^ARG WHEELS_TAG_X86=(\S+)$", (ROOT / "docker/Dockerfile").read_text(), re.M).group(1)
SOURCES = {
    "sgl-router": ("repos/radixark/sgl-router-for-miles/commits/main", "a" * 40),
    "int4_qat": (
        "repos/radixark/miles/commits?sha=main&path=miles/kernels/quant/int4_fake&per_page=1",
        "b" * 40,
    ),
    "te": ("repos/radixark/TransformerEngine/commits/miles-main", "c" * 40),
    "sglang": ("repos/sgl-project/sglang/commits/sglang-miles", "e" * 40),
}


def run_check(
    tmp_path, wheels, selected, *, force="true", failure=None, built=None, missing_release=False, sglang_built=()
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "endpoint = args[{'-H': 3, '--paginate': 2}.get(args[1], 1)]\n"
        "sources = json.loads(os.environ['SOURCES'])\n"
        "if endpoint == os.environ['FAILURE']:\n"
        "    sys.exit('selected lookup unavailable')\n"
        "if '/releases/tags/' in endpoint and os.environ['MISSING_RELEASE']:\n"
        "    sys.exit('gh: Not Found (HTTP 404)')\n"
        "if endpoint in sources:\n"
        "    print(sources[endpoint])\n"
        "elif endpoint.startswith('repos/radixark/miles-wheels/releases?'):\n"
        "    print(os.environ['MOCK_SGLANG_BUILT'])\n"
        "elif '/releases/tags/' in endpoint:\n"
        "    print('https://example.test/manifest' if os.environ['MOCK_RELEASE_COMMIT'] else '')\n"
        "elif endpoint == 'https://example.test/manifest':\n"
        "    print(json.dumps({'commit': os.environ['MOCK_RELEASE_COMMIT']}))\n"
        "else:\n"
        "    sys.exit('unexpected lookup: ' + endpoint)\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    script = next(step["run"] for step in WORKFLOW["jobs"]["check"]["steps"] if step.get("id") == "check")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            **WORKFLOW["env"],
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GITHUB_REPOSITORY": "radixark/miles",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "WHEELS_INPUT": wheels,
            "FORCE": force,
            "TE_REF": "miles-main",
            "SGLANG_REF": "sglang-miles",
            "SOURCES": json.dumps(dict(SOURCES[name] for name in selected)),
            "FAILURE": failure or "",
            "MOCK_RELEASE_COMMIT": built or "",
            "MISSING_RELEASE": "1" if missing_release else "",
            # the published wheels of the resolved commit, as gh's --jq filter returns them
            "MOCK_SGLANG_BUILT": "\n".join(sglang_built),
        },
        capture_output=True,
        text=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
    return result, outputs


@pytest.mark.parametrize("wheels", SOURCES)
def test_scoped_dispatch_does_not_resolve_other_sources(tmp_path, wheels):
    result, outputs = run_check(tmp_path, wheels, [wheels])
    assert result.returncode == 0, result.stderr
    assert bool(json.loads(outputs["router_matrix"])) == (wheels == "sgl-router")
    assert outputs["int4_x86"] == str(wheels == "int4_qat").lower()
    assert outputs["te_x86"] == str(wheels == "te").lower()
    assert bool(json.loads(outputs["sglang_matrix"])) == (wheels == "sglang")
    for name, output in [
        ("sgl-router", "router_sha"),
        ("int4_qat", "int4_sha"),
        ("te", "te_sha"),
        ("sglang", "sglang_sha"),
    ]:
        assert outputs[output] == (SOURCES[name][1] if name == wheels else "")


@pytest.mark.parametrize("built", [None, "d" * 40, "a" * 40])
def test_router_builds_only_when_manifest_is_missing_or_stale(tmp_path, built):
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", built=built)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(outputs["router_matrix"])) == (0 if built == "a" * 40 else 2)
    assert len(json.loads(outputs["publish_matrix"])) == (0 if built == "a" * 40 else 2)


def _sglang_wheel(python, arch):
    return f"sglang-0.5.22.post57+miles.geeeeeeeeee-cp{python}-cp{python}-manylinux_2_35_{arch}.whl"


_EVERY_SGLANG_WHEEL = [_sglang_wheel(py, arch) for py in ("311", "312") for arch in ("x86_64", "aarch64")]


@pytest.mark.parametrize(
    "built,force,expected",
    [
        ([], "false", {(arch, py) for arch in ("x86_64", "aarch64") for py in ("3.11", "3.12")}),
        (
            [_sglang_wheel(py, "x86_64") for py in ("311", "312")] + [_sglang_wheel("311", "aarch64")],
            "false",
            {("aarch64", "3.12")},
        ),
        # a published wheel is never rebuilt, even when forced: pinned hashes must keep matching
        (_EVERY_SGLANG_WHEEL, "true", set()),
    ],
    ids=["nothing-built", "retry-missing-pythons", "all-built-even-forced"],
)
def test_sglang_builds_only_the_wheels_no_history_release_holds(tmp_path, built, force, expected):
    result, outputs = run_check(tmp_path, "sglang", ["sglang"], force=force, sglang_built=built)
    assert result.returncode == 0, result.stderr
    build_matrix = json.loads(outputs["sglang_build_matrix"])
    assert {(entry["arch"], entry["python"]) for entry in build_matrix} == expected
    runners = {"x86_64": "ubuntu-22.04", "aarch64": "ubuntu-22.04-arm"}
    assert all(entry["runner"] == runners[entry["arch"]] for entry in build_matrix)
    # one publish per arch with anything to build
    assert {entry["arch"] for entry in json.loads(outputs["sglang_matrix"])} == {arch for arch, _ in expected}
    # one history release per ISO week keeps each under GitHub's 1000-asset cap
    assert outputs["sglang_release"] == SGLANG_RELEASE
    # sglang publishes through its own job, never into the rolling releases
    assert json.loads(outputs["publish_matrix"]) == []


@pytest.mark.parametrize("wheels", ["all", "sglang"])
def test_an_unresolvable_sglang_ref_stops_only_an_sglang_dispatch(tmp_path, wheels):
    """With wheels=all the image's wheel sets must still build when sglang cannot be resolved."""
    result, outputs = run_check(tmp_path, wheels, list(SOURCES), force="false", failure=SOURCES["sglang"][0])
    if wheels == "sglang":
        assert result.returncode != 0
        assert not outputs
        return
    assert result.returncode == 0, result.stderr
    assert "skipping sglang" in result.stdout
    assert json.loads(outputs["sglang_build_matrix"]) == []
    assert len(json.loads(outputs["router_matrix"])) == 2


def test_an_unlistable_history_stops_the_run(tmp_path):
    """Without the published list every wheel would read as unbuilt and be built again."""
    failure = "repos/radixark/miles-wheels/releases?per_page=100"
    result, outputs = run_check(tmp_path, "sglang", ["sglang"], force="false", failure=failure)
    assert result.returncode != 0
    assert "cannot list the sglang-miles releases" in result.stdout
    assert not outputs


def test_a_missing_rolling_release_still_stops_the_run(tmp_path):
    """Only the history releases are created by their first publish; the image's releases must exist."""
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", missing_release=True)
    assert result.returncode != 0
    assert "cannot read release" in result.stdout  # the ::error:: annotation
    assert not outputs


def test_each_sglang_arch_publishes_whatever_built():
    job = WORKFLOW["jobs"]["publish-sglang"]
    assert '"failure"' in job["if"] and '"success"' in job["if"]
    download = next(step["with"] for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4")
    upload = WORKFLOW["jobs"]["build-sglang"]["steps"]
    upload = next(step["with"] for step in upload if step.get("uses") == "actions/upload-artifact@v4")
    # every Python version's artifact of this arch, and none of the other arch's
    assert upload["name"] == "sglang-${{ matrix.arch }}-py${{ matrix.python }}"
    assert download == {**download, "pattern": "sglang-${{ matrix.arch }}-py*", "merge-multiple": True}


@pytest.mark.parametrize(
    "failure",
    [SOURCES["sgl-router"][0], f"repos/radixark/miles-wheels/releases/tags/{WHEELS_TAG_X86}"],
)
def test_selected_lookup_failure_stops_the_run(tmp_path, failure):
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", failure=failure)
    assert result.returncode != 0
    assert "selected lookup unavailable" in result.stderr
    assert not outputs


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize(
    "arch,build_arch,sets",
    [
        ("x86_64", "x86", ["sgl-router", "int4_qat", "te"]),
        ("x86_64", "x86", ["te"]),
        ("aarch64", "aarch64", ["sgl-router"]),
    ],
)
@pytest.mark.parametrize("upload_fails", [False, True])
def test_publish_keeps_superseded_wheels_only_when_asked(tmp_path, keep, arch, build_arch, sets, upload_fails):
    steps = WORKFLOW["jobs"]["publish"]["steps"]
    download = next(step["with"] for step in steps if step.get("uses") == "actions/download-artifact@v4")
    for name in sets:
        directory = tmp_path / "wheels"
        if not download["merge-multiple"]:
            directory /= f"wheels-{arch}-{name}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.whl").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    pip = bin_dir / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)
    uploader = tmp_path / "miles-wheels" / "build_wheels.py"
    uploader.parent.mkdir()
    uploader.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['UPLOAD_LOG'], 'a') as f:\n"
        "    f.write(json.dumps({'files': sorted(p.name for p in Path(os.environ['WHEEL_DIR']).glob('*.whl')), "
        "'args': sys.argv[1:]}) + '\\n')\n"
        "sys.exit(1 if os.environ['UPLOAD_FAILS'] == 'true' else 0)\n"
    )
    log = tmp_path / "uploads.jsonl"
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", steps[-1]["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "UPLOAD_LOG": str(log),
            "ARCH": arch,
            "BUILD_ARCH": build_arch,
            "CUDA": "130",
            "TORCH": "213",
            "KEEP_SUPERSEDED": "--keep-superseded" if keep else "",
            "UPLOAD_FAILS": str(upload_fails).lower(),
        },
        capture_output=True,
        text=True,
    )
    uploads = [json.loads(line) for line in log.read_text().splitlines()]
    if upload_fails:
        assert result.returncode != 0
        assert len(uploads) == 1
        return
    assert result.returncode == 0, result.stderr
    assert len(uploads) == len(sets)
    for name in sets:
        upload = next(upload for upload in uploads if upload["files"] == [f"{name}.whl"])
        expected = ["upload", "--cuda", "130", "--arch", build_arch, "--torch", "213"]
        if keep:
            expected.append("--keep-superseded")
        assert upload["args"] == expected


@pytest.mark.parametrize("arch,build_arch", [("x86_64", "x86"), ("aarch64", "aarch64")])
@pytest.mark.parametrize("built_wheels", [True, False])
def test_sglang_publish_uploads_into_the_history_release(tmp_path, arch, build_arch, built_wheels):
    """An arch with wheels uploads them into its history release; one whose builds all failed skips."""
    steps = WORKFLOW["jobs"]["publish-sglang"]["steps"]
    build = WORKFLOW["jobs"]["build-sglang"]["steps"]
    upload = next(step["with"] for step in build if step.get("uses") == "actions/upload-artifact@v4")
    # the rolling-release publish job downloads wheels-<arch>-*; sglang must not match it
    assert not upload["name"].startswith("wheels-")
    (tmp_path / "wheels").mkdir()
    if built_wheels:
        (tmp_path / "wheels" / "sglang-0.5.22+miles.g0-cp312-cp312-manylinux_2_35_x86_64.whl").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    pip = bin_dir / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n")
    pip.chmod(0o755)
    uploader = tmp_path / "miles-wheels" / "build_wheels.py"
    uploader.parent.mkdir()
    log = tmp_path / "upload.json"
    uploader.write_text(
        "import json, os, sys\n"
        "json.dump({'wheel_dir': os.environ['WHEEL_DIR'], 'args': sys.argv[1:]}, open(os.environ['UPLOAD_LOG'], 'w'))\n"
    )
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", steps[-1]["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            **WORKFLOW["env"],
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "UPLOAD_LOG": str(log),
            "BUILD_ARCH": build_arch,
            "CUDA": "130",
            "SGLANG_RELEASE": SGLANG_RELEASE,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if not built_wheels:
        assert not log.exists()
        return
    upload = json.loads(log.read_text())
    assert upload["wheel_dir"] == f"{tmp_path}/wheels"
    assert upload["args"] == ["upload", "--cuda", "130", "--arch", build_arch, "--history-release", SGLANG_RELEASE]


@pytest.mark.parametrize("job", ["publish", "publish-sglang"])
def test_wheels_publish_scopes_the_ci_app_token(job):
    token = next(
        step["with"]
        for step in WORKFLOW["jobs"][job]["steps"]
        if step.get("uses", "").startswith("actions/create-github-app-token@")
    )
    assert token["client-id"] == "${{ vars.CI_APP_CLIENT_ID }}"
    assert token["private-key"] == "${{ secrets.CI_APP_PRIVATE_KEY }}"
    assert token["owner"] == "radixark"
    assert token["repositories"] == "miles-wheels"
    assert {key: value for key, value in token.items() if key.startswith("permission-")} == {
        "permission-contents": "write"
    }


def test_failure_notifier_covers_all_jobs_without_build_or_publish_credentials():
    notify = WORKFLOW["jobs"]["notify-build-failure"]
    assert set(notify["needs"]) == set(WORKFLOW["jobs"]) - {"notify-build-failure"}
    assert "always()" in notify["if"]
    assert "contains(needs.*.result, 'failure')" in notify["if"]
    assert "github.repository == 'radixark/miles'" in notify["if"]
    assert "github.event_name" not in notify["if"]
    assert notify["runs-on"] == "ubuntu-latest"
    assert notify["permissions"] == {"actions": "read", "contents": "read"}
    checkout = notify["steps"][0]["with"]
    assert checkout["persist-credentials"] is False
    assert set(checkout["sparse-checkout"].splitlines()) == {
        ".github/workflows/scripts/lark_notify.py",
        ".github/workflows/scripts/ci_failure_analysis.py",
    }
    post = notify["steps"][-1]
    assert "lark_notify.py wheels-build-failure" in post["run"]
    assert post["env"]["LARK_WEBHOOK"] == "${{ secrets.LARK_WEBHOOK }}"
    assert post["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert post["env"]["RUN_ID"] == "${{ github.run_id }}"
    assert "CI_APP" not in json.dumps(notify)
