import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/build-wheels.yml").read_text())
SOURCES = {
    "sgl-router": ("repos/radixark/sgl-router-for-miles/commits/main", "a" * 40),
    "int4_qat": (
        "repos/radixark/miles/commits?sha=main&path=miles/backends/megatron_utils/kernels/int4_qat&per_page=1",
        "b" * 40,
    ),
    "te": ("repos/radixark/TransformerEngine/commits/miles-main", "c" * 40),
}


def run_check(tmp_path, wheels, selected, *, event="workflow_dispatch", force="true", failure=None, built=None):
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
            "EVENT": event,
            "WHEELS_INPUT": wheels,
            "FORCE": force,
            "TE_REF": "miles-main",
            "SOURCES": json.dumps(dict(SOURCES[name] for name in selected)),
            "FAILURE": failure or "",
            "MOCK_RELEASE_COMMIT": built or "",
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
    for name, output in [("sgl-router", "router_sha"), ("int4_qat", "int4_sha"), ("te", "te_sha")]:
        assert outputs[output] == (SOURCES[name][1] if name == wheels else "")


def test_schedule_does_not_resolve_disabled_te_source(tmp_path):
    result, outputs = run_check(tmp_path, "all", ["sgl-router", "int4_qat"], event="schedule")
    assert result.returncode == 0, result.stderr
    assert outputs["te_x86"] == "false"
    assert outputs["te_sha"] == ""
    assert outputs["int4_x86"] == "true"
    assert len(json.loads(outputs["router_matrix"])) == 2


@pytest.mark.parametrize("built", [None, "d" * 40, "a" * 40])
def test_router_builds_only_when_manifest_is_missing_or_stale(tmp_path, built):
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", built=built)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(outputs["router_matrix"])) == (0 if built == "a" * 40 else 2)
    assert len(json.loads(outputs["publish_matrix"])) == (0 if built == "a" * 40 else 2)


@pytest.mark.parametrize(
    "failure",
    [SOURCES["sgl-router"][0], "repos/radixark/miles-wheels/releases/tags/cu130-torch213-x86_64"],
)
def test_selected_lookup_failure_stops_the_run(tmp_path, failure):
    result, outputs = run_check(tmp_path, "sgl-router", ["sgl-router"], force="false", failure=failure)
    assert result.returncode != 0
    assert "selected lookup unavailable" in result.stderr
    assert not outputs


@pytest.mark.parametrize("te_on_schedule,keep", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize(
    "arch,build_arch,sets",
    [
        ("x86_64", "x86", ["sgl-router", "int4_qat", "te"]),
        ("x86_64", "x86", ["te"]),
        ("aarch64", "aarch64", ["sgl-router"]),
    ],
)
@pytest.mark.parametrize("upload_fails", [False, True])
def test_publish_retains_only_te_during_transition(
    tmp_path, te_on_schedule, keep, arch, build_arch, sets, upload_fails
):
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
            "TE_ON_SCHEDULE": str(te_on_schedule).lower(),
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
        if keep or (name == "te" and not te_on_schedule):
            expected.append("--keep-superseded")
        assert upload["args"] == expected


def test_wheels_publish_scopes_the_ci_app_token():
    token = next(
        step["with"]
        for step in WORKFLOW["jobs"]["publish"]["steps"]
        if step.get("uses", "").startswith("actions/create-github-app-token@")
    )
    assert token["client-id"] == "${{ vars.CI_APP_CLIENT_ID }}"
    assert token["private-key"] == "${{ secrets.CI_APP_PRIVATE_KEY }}"
    assert token["owner"] == "radixark"
    assert token["repositories"] == "miles-wheels"
    assert {key: value for key, value in token.items() if key.startswith("permission-")} == {
        "permission-contents": "write"
    }
