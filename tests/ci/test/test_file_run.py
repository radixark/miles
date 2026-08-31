"""Unit tests for `tests/ci/file_run.py`, the /rerun-test resolve step."""

import json
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.file_run import CPU_SUITES, CUDA_SUITE_RUNS_ON, FileRunError, main, plan_file_run, resolve_file_run
from tests.ci.run_suite import CI_SUITES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _make(
    filename: str,
    *,
    backend: HWBackend = HWBackend.CUDA,
    suite: str = "stage-c-8-gpu-h100",
    est_time: float = 60.0,
    nightly: bool = False,
    disabled: str | None = None,
) -> CIRegistry:
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        labels=["megatron"],
        nightly=nightly,
        disabled=disabled,
        implicit=False,
    )


def test_every_cuda_suite_has_a_runner_mapping():
    assert set(CUDA_SUITE_RUNS_ON) == set(CI_SUITES[HWBackend.CUDA])


def test_every_cpu_suite_is_allowed():
    assert set(CPU_SUITES) == set(CI_SUITES[HWBackend.CPU])


def test_cuda_file_resolves_to_its_suite_runner_and_image():
    tests = [
        _make("tests/e2e/x/test_a.py", suite="stage-c-4-gpu-h200", nightly=True),
        _make("tests/e2e/x/test_b.py"),
    ]
    plan = plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")
    assert plan == {
        "hw": "cuda",
        "suite": "stage-c-4-gpu-h200",
        "runs_on": json.dumps(["h200", "4gpu"]),
        "container_image": "radixark/miles:dev",
        "timeout_seconds": "1800",
    }


def test_cpu_file_resolves_without_runner_labels():
    tests = [_make("tests/fast/test_a.py", backend=HWBackend.CPU, suite="stage-a-cpu")]
    plan = plan_file_run(tests, "tests/fast/test_a.py", "pr-42")
    assert plan == {
        "hw": "cpu",
        "suite": "stage-a-cpu",
        "runs_on": "",
        "container_image": "radixark/miles:pr-42",
        "timeout_seconds": "1800",
    }


def test_long_file_extends_the_default_timeout():
    tests = [_make("tests/e2e/x/test_a.py", est_time=2000)]
    assert plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")["timeout_seconds"] == "2500"


def test_unknown_cpu_suite_is_a_hard_error():
    tests = [_make("tests/fast/test_a.py", backend=HWBackend.CPU, suite="stage-a-cpu; echo unexpected")]
    with pytest.raises(FileRunError, match="is not allowed"):
        plan_file_run(tests, "tests/fast/test_a.py", "dev")


def test_rocm_registration_is_ignored_next_to_a_cuda_one():
    tests = [
        _make("tests/e2e/x/test_a.py"),
        _make("tests/e2e/x/test_a.py", backend=HWBackend.ROCM, suite="stage-c-8-gpu-mi350"),
    ]
    assert plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")["suite"] == "stage-c-8-gpu-h100"


def test_unregistered_file_is_a_hard_error():
    with pytest.raises(FileRunError, match="/rerun-test runs only registered test files"):
        plan_file_run([_make("tests/e2e/x/test_a.py")], "tests/e2e/x/test_missing.py", "dev")


def test_rocm_only_file_is_a_hard_error():
    tests = [_make("tests/e2e/x/test_a.py", backend=HWBackend.ROCM, suite="stage-c-8-gpu-mi350")]
    with pytest.raises(FileRunError, match="/rerun-test supports CPU and CUDA"):
        plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")


def test_multiple_cpu_cuda_registrations_are_a_hard_error():
    tests = [
        _make("tests/e2e/x/test_a.py"),
        _make("tests/e2e/x/test_a.py", suite="stage-c-4-gpu-h200"),
    ]
    with pytest.raises(FileRunError, match="multiple CPU/CUDA registrations"):
        plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")


def test_disabled_file_is_a_hard_error():
    tests = [_make("tests/e2e/x/test_a.py", disabled="flaky, see #1")]
    with pytest.raises(FileRunError, match="disabled: flaky"):
        plan_file_run(tests, "tests/e2e/x/test_a.py", "dev")


@pytest.mark.parametrize("tag", ["", "-bad", "a" * 129, "radixark/miles:dev"])
def test_invalid_image_tag_is_a_hard_error(tag):
    with pytest.raises(FileRunError, match="invalid CI image tag"):
        plan_file_run([], "tests/e2e/x/test_a.py", tag)


def test_resolve_reads_the_real_registry():
    # This test file registers itself as a stage-a-cpu CPU test above, so the
    # real registry must resolve it to the CPU plan.
    plan = resolve_file_run("tests/ci/test/test_file_run.py", "dev")
    assert plan["hw"] == "cpu"
    assert plan["suite"] == "stage-a-cpu"


def test_resolve_accepts_a_label_declared_by_the_source_snapshot(tmp_path):
    """Resolve registrations against the checked-out source label schema."""
    source_root = tmp_path / "source"
    test_root = source_root / "tests" / "e2e" / "example"
    label_root = source_root / "tests" / "ci"
    test_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    (label_root / "labels.py").write_text(
        'KNOWN_LABELS: dict[str, str] = {"source-new": "Source-only test domain"}\n'
    )
    (test_root / "test_source_new.py").write_text(
        "from tests.ci.ci_register import register_cuda_ci\n"
        'register_cuda_ci(est_time=60, suite="stage-c-4-gpu-h200", labels=["source-new"])\n'
    )

    plan = resolve_file_run("tests/e2e/example/test_source_new.py", "dev", source_root)

    assert plan["hw"] == "cuda"
    assert plan["suite"] == "stage-c-4-gpu-h200"


def test_resolve_rejects_a_symlinked_test_file(tmp_path):
    source_root = tmp_path / "source"
    test_root = source_root / "tests" / "fast"
    test_root.mkdir(parents=True)
    payload = source_root / "payload.py"
    payload.write_text("def test_payload(): pass\n")
    (test_root / "test_payload.py").symlink_to(payload)

    with pytest.raises(FileRunError, match="must not be a symlink"):
        resolve_file_run("tests/fast/test_payload.py", "dev", source_root)


def test_resolve_rejects_a_symlinked_tests_root(tmp_path):
    source_root = tmp_path / "source"
    payload_root = source_root / "payload" / "fast"
    payload_root.mkdir(parents=True)
    (payload_root / "test_payload.py").write_text("def test_payload(): pass\n")
    (source_root / "tests").symlink_to(source_root / "payload")

    with pytest.raises(FileRunError, match="must not be a symlink"):
        resolve_file_run("tests/fast/test_payload.py", "dev", source_root)


def test_main_writes_github_outputs(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("TEST_FILE", "tests/ci/test/test_file_run.py")
    monkeypatch.setenv("CI_IMAGE_TAG", "")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert main() == 0
    lines = output_path.read_text().splitlines()
    assert "hw=cpu" in lines
    assert "suite=stage-a-cpu" in lines
    assert "runs_on=" in lines
    assert "container_image=radixark/miles:dev" in lines
    assert "timeout_seconds=1800" in lines


def test_main_fails_closed_on_an_unregistered_file(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "github-output"
    monkeypatch.setenv("TEST_FILE", "tests/e2e/test_does_not_exist.py")
    monkeypatch.setenv("CI_IMAGE_TAG", "")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert main() == 1
    assert not output_path.exists()
    assert "::error::" in capsys.readouterr().err


def test_target_workflow_keeps_orchestration_trusted_and_checks_out_exact_head():
    root = Path(__file__).parents[3]
    workflow = (root / ".github/workflows/run-ci-file.yml").read_text()
    gpu_workflow = (root / ".github/workflows/_run-ci.yml").read_text()
    cpu_workflow = (root / ".github/workflows/_run-cpu-ci.yml").read_text()

    assert "permissions:\n  contents: read" in workflow
    assert "name: Rerun Test" in workflow
    assert 'run-name: "/rerun-test ' in workflow
    assert workflow.count("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683") == 4
    assert workflow.count("ref: ${{ inputs.head_sha }}") == 3
    assert workflow.count("plan_already_resolved: true") == 2
    assert "path: pr-source" in workflow
    assert "CI_SOURCE_ROOT: ${{ github.workspace }}/pr-source" in workflow
    assert "run: python3 -S -m tests.ci.file_run" in workflow
    assert "DISPATCHED_SHA" not in workflow
    assert "checkout_ref" not in workflow
    assert "secrets: inherit" not in workflow
    assert "CI_COMMAND_APP_PRIVATE_KEY" not in workflow
    assert "NEON_DATABASE_URL" not in workflow
    # Fork-ness comes from the live PR, and fork heads run without repository
    # secrets, matching the pr-test fork policy.
    assert "head_is_fork: ${{ steps.head-repo.outputs.head_is_fork }}" in workflow
    assert (
        "WANDB_API_KEY: ${{ needs.resolve-file-run.outputs.head_is_fork != 'true' && secrets.WANDB_API_KEY || '' }}"
    ) in workflow
    assert (
        "HF_TOKEN: ${{ needs.resolve-file-run.outputs.head_is_fork != 'true' && secrets.HF_TOKEN || '' }}"
    ) in workflow
    assert "WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}" not in workflow
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" not in workflow
    assert "group: run-ci-file-${{ inputs.pull_number }}-${{ inputs.test_file }}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "queue: max" in workflow
    assert "comment_id: ${{ steps.announce.outputs.comment_id }}" in workflow
    assert "id: announce" in workflow
    assert "resolve-file-run:\n    needs: announce-file-run" in workflow
    assert "FILE_RUN_COMMENT_ID: ${{ needs.announce-file-run.outputs.comment_id }}" in workflow
    assert workflow.count("CI_COMMAND_FILE_RUN_STATUS: announce") == 1
    assert workflow.count("CI_COMMAND_FILE_RUN_STATUS: report") == 1
    assert workflow.count("issues: write") == 2
    assert workflow.count("pull-requests: write") == 2
    report_job = workflow.split("  report-file-run:", 1)[1]
    assert "actions: read" in report_job
    assert "needs: [announce-file-run, resolve-file-run, run-cuda-file, run-cpu-file]" in report_job
    assert "if: always()" in report_job
    assert "tests.ci.run_suite" not in workflow
    assert "pytest '${{ inputs.test_file }}' -v -x" in workflow
    assert "python3 '${{ inputs.test_file }}'" in workflow
    assert workflow.count("timeout --signal=TERM --kill-after=30s") == 2
    assert workflow.count("'${{ needs.resolve-file-run.outputs.timeout_seconds }}s'") == 2
    for reusable in (gpu_workflow, cpu_workflow):
        assert "plan_already_resolved:" in reusable
        assert "if: ${{ !inputs.plan_already_resolved }}" in reusable
        assert "control_ref" not in reusable
    assert (
        "GITHUB_COMMIT_NAME: ${{ inputs.ref || github.sha }}_"
        "${{ github.event.pull_request.number || github.event.inputs.pull_number || 'non-pr' }}"
    ) in gpu_workflow
