"""GPU allocation plans must select a nonempty shard without importing tests."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
SUITES = {"cuda": "stage-c-4-gpu-h200", "rocm": "stage-c-4-gpu-mi350"}


def _plan(tmp_path, hw, selection_args, *, registration_options="", empty=False):
    if not empty:
        test_file = tmp_path / "tests/e2e/test_selected.py"
        test_file.parent.mkdir(parents=True)
        hardware = ', hardware=["hopper", "blackwell"]' if hw == "cuda" else ""
        test_file.write_text(
            f"from tests.ci.ci_register import register_{hw}_ci\n"
            f'register_{hw}_ci(1, "{SUITES[hw]}", labels=["megatron"]{hardware}{registration_options})\n'
            'raise AssertionError("Planning must never import or execute this test")\n'
        )
    output = tmp_path / "github-output"
    output.write_text("existing=value\n")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tests/ci/run_suite.py"),
            "--hw",
            hw,
            "--suite",
            SUITES[hw],
            "--list-only",
            "--github-output",
            str(output),
            *selection_args,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize(
    ("options", "selection_args", "has_tests"),
    [
        ("", [], False),
        ("", ["--labels", "run-ci-fsdp"], False),
        ("", ["--labels", "run-ci-megatron"], True),
        ("", ["--match-all-labels"], True),
        (', disabled="flaky"', ["--match-all-labels"], False),
        (", nightly=True", ["--match-all-labels"], False),
        (", nightly=True", ["--cadence", "nightly"], True),
        (", nightly=True", ["--cadence", "weekly"], True),
        (", nightly=True", ["--cadence", "release"], True),
        (', disabled="flaky"', ["--cadence", "weekly"], False),
    ],
)
def test_plan_applies_selection_without_runtime_dependencies(tmp_path, hw, options, selection_args, has_tests):
    result, output = _plan(tmp_path, hw, selection_args, registration_options=options)

    assert result.returncode == 0, result.stderr
    assert output == {"existing": "value", "has_tests": str(has_tests).lower()}
    assert ("No tests to run." in result.stdout) is not has_tests


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize("cadence", ["regular", "nightly", "weekly", "release"])
def test_empty_suite_never_requests_a_gpu(tmp_path, hw, cadence):
    result, output = _plan(tmp_path, hw, ["--cadence", cadence, "--match-all-labels"], empty=True)

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == "false"


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize("cadence", ["regular", "nightly", "weekly", "release"])
@pytest.mark.parametrize("partition", [0, 1, 8])
def test_plan_checks_each_shard_after_partitioning(tmp_path, hw, cadence, partition):
    result, output = _plan(
        tmp_path,
        hw,
        [
            "--cadence",
            cadence,
            "--match-all-labels",
            "--auto-partition-id",
            str(partition),
            "--auto-partition-size",
            "9",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == str(partition == 0).lower()


def test_arch_dispatch_does_not_allocate_the_tests_original_runner(tmp_path):
    result, output = _plan(tmp_path, "cuda", ["--labels", "run-ci-megatron", "run-on-blackwell"])

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == "false"


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
def test_invalid_registration_fails_without_an_empty_success_plan(tmp_path, hw):
    result, output = _plan(tmp_path, hw, ["--match-all-labels"], registration_options=", nightly=1")

    assert result.returncode != 0
    assert "nightly must be a boolean" in result.stderr
    assert "has_tests" not in output


def test_selection_output_cannot_be_requested_while_executing_tests(tmp_path):
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "tests.ci.run_suite",
            "--hw",
            "cuda",
            "--suite",
            SUITES["cuda"],
            "--github-output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--github-output requires --list-only." in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("workflow", ["_run-ci.yml", "_run-ci-rocm.yml"])
def test_gpu_workflow_requires_a_successful_hosted_plan(workflow):
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    plan, run = jobs["plan"], jobs["run"]

    assert plan["runs-on"] == "ubuntu-latest"
    assert "container" not in plan
    assert plan["permissions"] == {"contents": "read"}
    assert plan["outputs"]["has_tests"] == "${{ steps.select.outputs.has_tests }}"
    assert plan["outputs"]["ref"] == "${{ steps.select.outputs.ref }}"
    checkout = next(step for step in plan["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ inputs.ref }}"
    assert checkout["with"]["persist-credentials"] is False
    select = next(step for step in plan["steps"] if step.get("id") == "select")
    assert '${{ inputs.execute_command }} --list-only --github-output "$GITHUB_OUTPUT"' in select["run"]
    assert 'echo "ref=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"' in select["run"]
    assert run["needs"] == "plan"
    assert "needs.plan.result == 'success'" in run["if"]
    assert "needs.plan.outputs.has_tests == 'true'" in run["if"]
    assert not any("--list-only" in step.get("run", "") for step in run["steps"])
    checkout = next(step for step in run["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert "needs.plan.outputs.ref" in checkout["with"]["ref"]
    if workflow == "_run-ci.yml":
        assert plan["if"] == "${{ !inputs.plan_already_resolved }}"
        assert "inputs.plan_already_resolved ||" in run["if"]
        assert "!cancelled()" in run["if"]
        assert "inputs.plan_already_resolved && inputs.ref" in checkout["with"]["ref"]


@pytest.mark.parametrize(
    ("expected", "actual"), [(1, 1), (2, 2), (4, 4), (8, 8), (1, 8), (2, 4), (4, 8), (4, 0), (8, 4)]
)
def test_b200_partition_check_rejects_wrong_gpu_exposure(monkeypatch, expected, actual):
    run = yaml.safe_load((ROOT / ".github/workflows/_run-ci.yml").read_text())["jobs"]["run"]
    step = next(step for step in run["steps"] if step.get("name") == "Verify B200 CUDA partition")
    code = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    monkeypatch.setenv("EXPECTED_GPUS", str(expected))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: actual)))
    if expected == actual:
        exec(code)
    else:
        with pytest.raises(AssertionError, match=f"requires {expected} CUDA-visible GPUs, found {actual}"):
            exec(code)


def test_b200_suite_and_file_runs_allow_long_test_timeout():
    workflows = ROOT / ".github/workflows"
    b200 = yaml.safe_load((workflows / "_run-ci-b200.yml").read_text())
    batches = b200["jobs"]["test"]
    assert batches["uses"] == "./.github/workflows/_run-ci-b200-batch.yml"
    assert batches["strategy"]["max-parallel"] == 1
    assert batches["strategy"]["fail-fast"] is False
    assert batches["with"]["matrix"] == "${{ matrix.jobs }}"
    batch = yaml.safe_load((workflows / "_run-ci-b200-batch.yml").read_text())["jobs"]["test"]
    assert batch["strategy"]["matrix"] == "${{ fromJSON(inputs.matrix) }}"
    assert "max-parallel" not in batch["strategy"]
    assert batch["strategy"]["fail-fast"] is False
    assert batch["with"]["timeout_minutes"] == "${{ matrix.timeout_minutes }}"
    assert "--timeout-per-file" not in batch["with"]["execute_command"]
    rerun = yaml.safe_load((workflows / "run-ci-file.yml").read_text())["jobs"]["run-cuda-file"]
    assert "'stage-c-4-gpu-b200' && 540 || 360" in rerun["with"]["timeout_minutes"]
    reusable = yaml.safe_load((workflows / "_run-ci.yml").read_text())
    assert reusable["jobs"]["run"]["timeout-minutes"] == "${{ inputs.timeout_minutes }}"
