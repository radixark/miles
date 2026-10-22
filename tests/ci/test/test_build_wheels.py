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


def run_check(tmp_path, wheels, selected, *, force="true", failure=None, built=None, missing_release=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "endpoint = args[3] if args[1] == '-H' else args[1]\n"
        "sources = json.loads(os.environ['SOURCES'])\n"
        "if endpoint == os.environ['FAILURE']:\n"
        "    sys.exit('selected lookup unavailable')\n"
        "if '/releases/tags/' in endpoint and os.environ['MISSING_RELEASE']:\n"
        "    sys.exit('gh: Not Found (HTTP 404)')\n"
        "if endpoint in sources:\n"
        "    print(sources[endpoint])\n"
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


@pytest.mark.parametrize("built", [None, "d" * 40, "e" * 40])
def test_sglang_builds_both_arches_on_github_runners_only_when_stale(tmp_path, built):
    result, outputs = run_check(tmp_path, "sglang", ["sglang"], force="false", built=built)
    assert result.returncode == 0, result.stderr
    matrix = json.loads(outputs["sglang_matrix"])
    if built == "e" * 40:
        assert matrix == []
        return
    assert {(entry["arch"], entry["runner"]) for entry in matrix} == {
        ("x86_64", "ubuntu-22.04"),
        ("aarch64", "ubuntu-22.04-arm"),
    }
    # sglang publishes through its own job, never into the rolling releases.
    assert json.loads(outputs["publish_matrix"]) == []


def test_a_history_release_that_does_not_exist_yet_reads_as_never_built(tmp_path):
    result, outputs = run_check(tmp_path, "sglang", ["sglang"], force="false", missing_release=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(outputs["sglang_matrix"])) == 2


def test_a_missing_rolling_release_still_stops_the_run(tmp_path):
    """Only the history releases are created by their first publish; the image's releases must exist."""
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", missing_release=True)
    assert result.returncode != 0
    assert "cannot read release" in result.stdout  # the ::error:: annotation
    assert not outputs


def test_each_sglang_arch_publishes_whatever_the_other_did():
    job = WORKFLOW["jobs"]["publish-sglang"]
    assert '"failure"' in job["if"] and '"success"' in job["if"]
    download = next(step for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4")
    upload = job["steps"][-1]
    assert download["continue-on-error"] is True
    assert upload["if"] == f"steps.{download['id']}.outcome == 'success'"


@pytest.mark.parametrize(
    "failure",
    [
        SOURCES["sgl-router"][0],
        f"repos/radixark/miles-wheels/releases/tags/{WHEELS_TAG_X86}",
        "repos/radixark/miles-wheels/releases/tags/sglang-miles-x86_64",
    ],
)
def test_selected_lookup_failure_stops_the_run(tmp_path, failure):
    wheels = "sglang" if "sglang-miles" in failure else "sgl-router"
    result, outputs = run_check(tmp_path, wheels, [wheels], force="false", failure=failure)
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
def test_sglang_publish_uploads_into_the_history_release(tmp_path, arch, build_arch):
    steps = WORKFLOW["jobs"]["publish-sglang"]["steps"]
    download = next(step["with"] for step in steps if step.get("uses") == "actions/download-artifact@v4")
    assert download["name"] == "sglang-${{ matrix.arch }}"
    build = WORKFLOW["jobs"]["build-sglang"]["steps"]
    upload = next(step["with"] for step in build if step.get("uses") == "actions/upload-artifact@v4")
    # The rolling-release publish job downloads wheels-<arch>-*; sglang must not match it.
    assert not upload["name"].startswith("wheels-")

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
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    upload = json.loads(log.read_text())
    assert upload["wheel_dir"] == f"{tmp_path}/wheels"
    assert upload["args"] == ["upload", "--cuda", "130", "--arch", build_arch, "--history-release", "sglang-miles"]


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
