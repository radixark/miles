import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ci.b200_plan import plan_jobs
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.file_run import plan_file_run

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def registration(name, count, *, nightly=False, disabled=None, est_time=60):
    return CIRegistry(
        backend=HWBackend.CUDA,
        filename=f"tests/fast-gpu/test_{name}.py",
        suite="stage-c-4-gpu-h200" if count <= 4 else "stage-c-8-gpu-h200",
        est_time=est_time,
        hardware=["hopper", "blackwell"],
        labels=["megatron"],
        num_gpus=count,
        nightly=nightly,
        disabled=disabled,
    )


def test_blackwell_plan_mixes_minimum_counts_without_stage_barriers():
    tests = [registration(str(count), count) for count in (1, 2, 4, 8)]
    jobs = plan_jobs(
        tests, ["stage-c-4-gpu-b200", "stage-c-8-gpu-b200"], "regular", ["run-ci-megatron", "run-on-blackwell"]
    )
    assert {job["num_gpus"] for job in jobs} == {1, 2, 4, 8}
    assert {job["file"] for job in jobs} == {test.filename for test in tests}
    for job in jobs:
        assert json.loads(job["runs_on"]) == ["b200", f"{job['num_gpus']}gpu"]


@pytest.mark.parametrize(("cadence", "expected"), [("regular", {"regular"}), ("weekly", {"regular", "nightly"})])
def test_dynamic_plan_preserves_selection_and_disabled_coverage(cadence, expected):
    tests = [
        registration("regular", 1),
        registration("nightly", 2, nightly=True),
        registration("disabled", 4, disabled="broken"),
    ]
    jobs = plan_jobs(tests, ["stage-c-4-gpu-b200"], cadence, ["run-ci-megatron", "run-on-blackwell"])
    assert {job["file"] for job in jobs} == {f"tests/fast-gpu/test_{name}.py" for name in expected}
    assert not plan_jobs(tests, ["stage-c-4-gpu-b200"], "regular", ["run-ci-megatron"])


def test_long_test_gets_timeout_plus_setup_budget():
    jobs = plan_jobs(
        [registration("long", 4, est_time=21600)],
        ["stage-c-4-gpu-b200"],
        "regular",
        ["run-ci-megatron", "run-on-blackwell"],
    )
    assert jobs[0]["timeout_minutes"] * 60 >= 21600 * 1.25 + 1200


def test_file_rerun_uses_minimum_blackwell_allocation():
    test = registration("quantizer", 2)
    test.suite = "stage-c-4-gpu-b200"
    test.hardware = ["blackwell"]
    plan = plan_file_run([test], test.filename, "dev")
    assert plan["num_gpus"] == "2"
    assert json.loads(plan["runs_on"]) == ["b200", "2gpu"]


def test_snapshot_plan_never_executes_source_python(tmp_path):
    snapshot = tmp_path / "snapshot"
    tests = snapshot / "tests/fast-gpu"
    tests.mkdir(parents=True)
    labels = snapshot / "tests/ci/labels.py"
    labels.parent.mkdir()
    labels.write_text('KNOWN_LABELS: dict = {"megatron": "test"}\nraise AssertionError("imported labels")\n')
    (snapshot / "tests/__init__.py").write_text('raise AssertionError("imported snapshot")\n')
    (tests / "test_one.py").write_text(
        'register_cuda_ci(10, "stage-c-4-gpu-b200", labels=["megatron"], hardware=["blackwell"], num_gpus=1)\n'
        'raise AssertionError("executed test while planning")\n'
    )
    output = tmp_path / "output"
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-S", "-m", "tests.ci.b200_plan", "--source-root", str(snapshot), "--cadence", "regular", "--labels", "run-ci-megatron", "--suites", "stage-c-4-gpu-b200"],
        cwd=root,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["has_tests"] == "true"
    assert json.loads(values["matrix"])["include"][0]["num_gpus"] == 1


def test_planned_filename_remains_one_literal_shell_argument():
    test = registration("quoted'$(touch unwanted)", 1)
    jobs = plan_jobs([test], ["stage-c-4-gpu-b200"], "regular", ["run-ci-megatron", "run-on-blackwell"])
    assert shlex.split(jobs[0]["shell_file"]) == [test.filename]
