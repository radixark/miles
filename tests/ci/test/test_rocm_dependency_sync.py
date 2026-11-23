"""CPU checks for the ROCm ref resolver and the container dependency setup."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu")

ROOT = Path(__file__).resolve().parents[3]
ACTION = ROOT / ".github/actions/resolve-dependency-refs/action.yml"
WORKFLOW = ROOT / ".github/workflows/_run-ci-rocm.yml"
CONTAINER_STEPS = ROOT / ".github/workflows/scripts/spur/container_steps.sh"


def test_rocm_refs_resolve_for_suite_and_preselected_file():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow[True]["workflow_call"]["inputs"]["skip_dependency_install"]["default"] is False
    jobs = workflow["jobs"]
    assert jobs["plan"]["if"] == "${{ !inputs.plan_already_resolved }}"
    deps = jobs["deps"]
    assert deps["needs"] == "plan"
    assert deps["runs-on"] == "ubuntu-latest"
    assert deps["permissions"] == {"contents": "read"}
    assert "!cancelled()" in deps["if"] and "inputs.plan_already_resolved ||" in deps["if"]
    source, _, actions, resolver = deps["steps"]
    assert source["with"]["ref"] == "${{ inputs.plan_already_resolved && inputs.ref || needs.plan.outputs.ref }}"
    assert actions["with"]["ref"] == "${{ github.workflow_sha }}"
    assert all(step["with"]["persist-credentials"] is False for step in (source, actions))
    assert resolver["uses"] == "./.ci-actions/.github/actions/resolve-dependency-refs"
    assert jobs["run"]["needs"] == ["plan", "deps"]
    launch = next(step for step in jobs["run"]["steps"] if step["name"] == "Run tests on a cluster node")
    assert launch["env"]["SGLANG_PR"] == "${{ needs.deps.outputs.ci_sglang_pr }}"
    assert launch["env"]["MEGATRON_PR"] == "${{ needs.deps.outputs.ci_megatron_pr }}"
    assert launch["env"]["BRIDGE_REF"] == "${{ steps.resolve-bridge.outputs.ci_bridge_pr }}"
    for filename, job in [("run-ci-file.yml", "run-rocm-file"), ("pr-test-rocm.yml", "changed-rocm-file")]:
        caller = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"][job]
        assert caller["with"]["plan_already_resolved"] is True
        assert "skip_dependency_install" not in caller["with"]
        assert "resolve-ci-deps" not in str(caller.get("needs"))


def _condition(text, values):
    # Evaluate the actual workflow boolean expression with fixed job results.
    for key, value in sorted(values.items(), key=lambda item: -len(item[0])):
        text = text.replace(key, repr(value))
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", "not ", text)
    return eval(" ".join(text.split()), {"__builtins__": {}}, {})


@pytest.mark.parametrize(
    ("preselected", "plan_result", "has_tests", "skip", "deps_result", "cancelled", "resolve", "execute"),
    [
        (False, "success", "true", False, "success", False, True, True),
        (False, "success", "false", False, "skipped", False, False, False),
        (False, "failure", "", False, "skipped", False, False, False),
        (True, "skipped", "", False, "success", False, True, True),
        (True, "skipped", "", False, "failure", False, True, False),
        (True, "skipped", "", True, "skipped", False, False, True),
        (True, "skipped", "", False, "success", True, False, False),
    ],
)
def test_dependency_and_gpu_gates(preselected, plan_result, has_tests, skip, deps_result, cancelled, resolve, execute):
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    values = {
        "inputs.plan_already_resolved": preselected,
        "inputs.skip_dependency_install": skip,
        "needs.plan.result": plan_result,
        "needs.plan.outputs.has_tests": has_tests,
        "needs.deps.result": deps_result,
        "cancelled()": cancelled,
    }
    assert _condition(jobs["deps"]["if"], values) is resolve
    assert _condition(jobs["run"]["if"], values) is execute


@pytest.mark.parametrize(
    ("dispatch", "body", "lock", "expected"),
    [
        (("", ""), "", None, ("sglang-miles", "miles-main")),
        (("", ""), "", ("a" * 40, "b" * 40), ("a" * 40, "b" * 40)),
        (
            ("", "test-branch"),
            "ci-sglang-pr: #41\nci-megatron-pr: #42",
            ("a" * 40, "b" * 40),
            ("refs/pull/41/head", "test-branch"),
        ),
        (("#43", ""), "ci-sglang-pr: ignored", ("a" * 40, "b" * 40), ("refs/pull/43/head", "b" * 40)),
    ],
)
def test_shared_resolver_reads_the_selected_snapshot(tmp_path, dispatch, body, lock, expected):
    source = tmp_path / "tested-source"
    source.mkdir()
    if lock:
        (source / "release-lock.json").write_text(
            json.dumps(dict(zip(("sglang_commit", "megatron_commit"), lock, strict=True)))
        )
    output = tmp_path / "github-output"
    action = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", action],
        cwd=source,
        env={
            "PATH": os.defpath,
            "INPUT_SGLANG_PR": dispatch[0],
            "INPUT_MEGATRON_PR": dispatch[1],
            "PR_BODY": body,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "ci_sglang_pr": expected[0],
        "ci_megatron_pr": expected[1],
    }


_STUB = """#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
with open(os.environ["TRACE"], "a") as log:
    log.write(json.dumps([name, Path.cwd().name, sys.argv[1:], os.environ.get("RUSTUP_AUTO_INSTALL")]) + "\\n")
toolchain = Path(os.environ["TOOLCHAIN"])
if name == "cargo":
    sys.exit(0 if toolchain.is_file() else 1)
if name == "rustup":
    if os.environ["FAIL_INSTALL"] == "true":
        sys.exit(42)
    toolchain.touch()
if name == "apt-get":
    sys.exit(97)
"""


@pytest.mark.parametrize("mode", ["preinstalled", "new-pin", "failed-install", "skip", "missing-ref"])
def test_container_sync_checks_rust_before_installing_or_running_tests(tmp_path, mode):
    sglang, megatron, miles = (tmp_path / name for name in ("sglang", "megatron", "miles"))
    for directory in (sglang, megatron, miles):
        directory.mkdir()
    (sglang / "rust").mkdir()
    (sglang / "rust/rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.93"\n')
    trace = tmp_path / "trace"
    toolchain = tmp_path / "installed"
    if mode == "preinstalled":
        toolchain.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("git", "pip", "protoc", "cargo", "rustup", "apt-get"):
        path = bin_dir / name
        path.write_text(_STUB)
        path.chmod(0o755)

    # Run the production install block with temporary checkout paths and stub
    # executables. The GPU setup and test command are outside this block.
    block = CONTAINER_STEPS.read_text().split('echo "::group::Install dependencies"\n', 1)[1]
    block = block.split('echo "::endgroup::"', 1)[0]
    block = block.replace("/sgl-workspace/sglang", shlex.quote(str(sglang)))
    block = block.replace("/root/Megatron-LM", shlex.quote(str(megatron)))
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", block],
        cwd=miles,
        env={
            "PATH": str(bin_dir) + os.pathsep + os.defpath,
            "TRACE": str(trace),
            "TOOLCHAIN": str(toolchain),
            "FAIL_INSTALL": str(mode == "failed-install").lower(),
            "CI_SKIP_DEPENDENCY_INSTALL": str(mode == "skip").lower(),
            "SGLANG_PR": "refs/pull/41/head",
            "MEGATRON_PR": "" if mode == "missing-ref" else "miles-main",
            "GITHUB_WORKSPACE": str(miles),
        },
        capture_output=True,
        text=True,
    )
    if mode == "missing-ref":
        assert result.returncode != 0
        assert "resolved Megatron-LM ref is required" in result.stderr
        assert not trace.exists()
        return

    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    if mode == "failed-install":
        assert result.returncode == 42
        assert not any(call[0] == "pip" for call in calls)
        assert calls[-1] == ["rustup", "rust", ["toolchain", "install", "--no-self-update"], None]
        return

    assert result.returncode == 0, result.stderr
    assert "Install dependencies elapsed=" in result.stdout
    if mode == "skip":
        assert all(call[0] == "pip" and call[1] == "miles" for call in calls)
        return

    rust_calls = [call for call in calls if call[0] in {"cargo", "rustup"}]
    check = ["cargo", "rust", ["--version", "--verbose"], "0"]
    if mode == "preinstalled":
        assert rust_calls == [check]
    else:
        assert rust_calls == [check, ["rustup", "rust", ["toolchain", "install", "--no-self-update"], None], check]
    first_pip = next(i for i, call in enumerate(calls) if call[0] == "pip")
    assert first_pip > max(i for i, call in enumerate(calls) if call[0] == "cargo")
    assert [call[1:3] for call in calls if call[0] == "git" and call[2][0] == "fetch"] == [
        ["sglang", ["fetch", "--", "origin", "refs/pull/41/head"]],
        ["megatron", ["fetch", "--", "origin", "miles-main"]],
    ]
    editable = [(call[1], call[2][2]) for call in calls if call[0] == "pip" and call[2][1] == "-e"]
    assert editable == [("sglang", "python"), ("sglang", "radix_transfer_plugins"), ("megatron", "."), ("miles", ".")]
    assert all("--no-deps" in call[2] for call in calls if call[0] == "pip" and call[2][1] == "-e")
