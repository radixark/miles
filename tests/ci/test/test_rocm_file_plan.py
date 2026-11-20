"""CPU tests for the opt-in ROCm changed-file planner."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu")

ROOT = Path(__file__).resolve().parents[3]
TWO_GPU = "stage-c-2-gpu-mi350"
FOUR_GPU = "stage-c-4-gpu-mi350"
EIGHT_GPU = "stage-c-8-gpu-mi350"
PR_SUITES = (TWO_GPU, FOUR_GPU, EIGHT_GPU)


def _registration(root, path, suite=TWO_GPU, *, labels=("precision",), est_time=1, disabled=None):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as stream:
        stream.write(
            f"register_rocm_ci({est_time}, {suite!r}, labels={list(labels)!r}, disabled={disabled!r})\n"
            'raise AssertionError("Planning should only read the AST")\n'
        )


def _plan(root, diff, labels, *, cadence="regular", skip_suite=True, pr_body=""):
    changed = root / "changed-files.z"
    changed.write_bytes(diff)
    output = root / "github-output"
    summary = root / "summary.md"
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tests.ci.rocm_file_plan"],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CHANGED_FILES_PATH": str(changed),
            "RAW_LABELS": " ".join(labels),
            "CADENCE": cadence,
            "SKIPPED_STAGES": json.dumps(list(PR_SUITES) if skip_suite else []),
            "PR_BODY": pr_body,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    jobs = json.loads(dict(line.split("=", 1) for line in output.read_text().splitlines())["rocm_file_jobs"])
    return jobs, result.stdout, summary.read_text() if summary.is_file() else ""


@pytest.mark.parametrize(
    ("labels", "cadence", "expected"),
    [
        ([], "regular", 0),
        (["run-ci-precision"], "regular", 0),
        (["run-ci-amd"], "regular", 1),
        (["run-ci-all"], "regular", 1),
        (["nightly"], "nightly", 1),
    ],
)
def test_only_amd_opt_in_labels_plan_files(tmp_path, labels, cadence, expected):
    filename = "tests/e2e/test_plan's file.py"
    _registration(tmp_path, filename)
    jobs, _, _ = _plan(tmp_path, f"M\0{filename}\0".encode(), labels, cadence=cadence)
    assert len(jobs) == expected
    if jobs:
        command = shlex.split(jobs[0]["execute_command"])
        assert command[command.index("--test-file") + 1] == filename
        assert jobs[0]["num_gpus"] == "2"


@pytest.mark.parametrize(
    ("pr_body", "image"),
    [
        ("ci-image-tag: pr-1\r\n", "rocm/sgl-dev:miles-rocm10-mi35x"),
        (
            "ci-image-tag: pr-1\r\nci-rocm-image-tag: miles-rocm10-mi35x-20261113\r\n",
            "rocm/sgl-dev:miles-rocm10-mi35x-20261113",
        ),
    ],
)
def test_file_jobs_use_the_pr_rocm_image(tmp_path, pr_body, image):
    filename = "tests/e2e/test_pinned.py"
    _registration(tmp_path, filename)
    jobs, _, _ = _plan(tmp_path, f"M\0{filename}\0".encode(), ["run-ci-amd"], pr_body=pr_body)
    assert [job["container_image"] for job in jobs] == [image]


def test_renames_use_destination_and_registered_width(tmp_path):
    two, four, eight, old, deleted, disabled = (
        f"tests/e2e/test_{s}.py" for s in ("two", "four", "eight", "old", "deleted", "disabled")
    )
    for path, suite in [(two, TWO_GPU), (four, FOUR_GPU), (eight, EIGHT_GPU), (old, FOUR_GPU), (deleted, FOUR_GPU)]:
        _registration(tmp_path, path, suite)
    _registration(tmp_path, disabled, TWO_GPU, disabled="see #1")
    diff = f"A\0{two}\0R100\0{old}\0{four}\0M\0{eight}\0D\0{deleted}\0M\0{disabled}\0".encode()
    jobs, _, _ = _plan(tmp_path, diff, ["run-ci-amd"])
    assert {job["test_file"]: job["num_gpus"] for job in jobs} == {two: "2", four: "4", eight: "8"}


@pytest.mark.parametrize("suite", PR_SUITES)
def test_regular_suite_owns_files_it_already_selects(tmp_path, suite):
    both, nightly = ("tests/e2e/test_both.py", "tests/e2e/test_nightly.py")
    _registration(tmp_path, both, suite, labels=("amd",))
    _registration(tmp_path, nightly, suite)
    diff = f"M\0{both}\0M\0{nightly}\0".encode()
    jobs, _, _ = _plan(tmp_path, diff, ["run-ci-amd"], skip_suite=False)
    assert [job["test_file"] for job in jobs] == [nightly]


@pytest.mark.parametrize("count", [4, 5])
def test_large_changes_report_all_candidates_without_partial_matrix(tmp_path, count):
    files = [f"tests/e2e/test_file_{i}.py" for i in range(count)]
    for path in files:
        _registration(tmp_path, path)
    jobs, stdout, summary = _plan(tmp_path, "".join(f"M\0{path}\0" for path in files).encode(), ["run-ci-amd"])
    if count == 4:
        assert [job["test_file"] for job in jobs] == files
        assert not summary
    else:
        assert jobs == []
        assert "::warning::" in stdout
        assert "exceed the limit of 4" in summary
        assert "/rerun-test <file> --hw rocm" in summary
        assert all(path in summary for path in files)


@pytest.mark.parametrize("label", ["long", "ft-long"])
@pytest.mark.parametrize("explicit", [False, True])
def test_long_registrations_need_their_own_labels(tmp_path, label, explicit):
    filename = "tests/e2e/test_long.py"
    _registration(tmp_path, filename, labels=("precision", label))
    labels = ["run-ci-all", *([f"run-ci-{label}"] if explicit else [])]
    jobs, stdout, summary = _plan(tmp_path, f"M\0{filename}\0".encode(), labels)
    if explicit:
        assert len(jobs) == 1
    else:
        assert jobs == []
        assert "::warning::" in stdout
        assert f"requires run-ci-{label}" in summary


def test_over_budget_file_does_not_fail_policy_or_other_files(tmp_path):
    too_long, normal = ("tests/e2e/test_oversized.py", "tests/e2e/test_normal.py")
    _registration(tmp_path, too_long, labels=("long",), est_time=24000)
    _registration(tmp_path, normal)
    diff = f"M\0{too_long}\0M\0{normal}\0".encode()
    jobs, stdout, summary = _plan(tmp_path, diff, ["run-ci-amd", "run-ci-long"])
    assert [job["test_file"] for job in jobs] == [normal]
    assert "::warning::" in stdout
    assert too_long in summary
    assert "30000s" in summary and "9900s" in summary
    assert summary.count(too_long) == 1
    assert "Comment reruns share this file budget" in summary
    assert "/rerun-test" not in summary


def test_invalid_diff_yields_no_file_jobs(tmp_path):
    jobs, _, _ = _plan(tmp_path, b"invalid", ["run-ci-amd"])
    assert jobs == []


def test_workflow_uses_separate_planner_and_shared_file_queue():
    workflow = (ROOT / ".github/workflows/pr-test-rocm.yml").read_text()
    job = workflow.split("\n  changed-rocm-file:\n", 1)[1]
    manual = (ROOT / ".github/workflows/run-ci-file.yml").read_text().split("\n  run-rocm-file:\n", 1)[1]
    assert "python -S -m tests.ci.rocm_file_plan" in workflow
    assert "if: github.event_name == 'pull_request' && !inputs.cadence" in workflow
    assert "SKIPPED_STAGES: ${{ steps.resolve.outputs.skipped_stages }}" in workflow
    assert "rocm_file_jobs: ${{ steps.rocm-files.outputs.rocm_file_jobs }}" in workflow
    assert "needs: [resolve-ci-policy, resolve-ci-image]" in job
    assert "needs.resolve-ci-image.result == 'success'" in job
    assert "container_image: ${{ needs.resolve-ci-image.outputs.container_image }}" in job
    assert (
        "(steps.rocm-files.outputs.rocm_file_jobs || '[]') != '[]'"
        in workflow.split("  resolve-ci-policy:", 1)[1].split("    steps:", 1)[0]
    )
    assert "ref: ${{ github.sha }}" in job
    assert "execute_command: ${{ matrix.execute_command }}" in job
    assert "num_gpus: ${{ fromJSON(matrix.num_gpus) }}" in job
    assert "withhold_runner_secrets:" in job
    for source in (job, manual):
        assert "group: rocm-file-run" in source
        assert "cancel-in-progress: false" in source
        assert "queue: max" in source
