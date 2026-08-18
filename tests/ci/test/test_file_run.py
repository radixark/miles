"""Unit tests for `tests/ci/file_run.py`, the /run-ci resolve step."""

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
    nightly: bool = False,
    disabled: str | None = None,
) -> CIRegistry:
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=60.0,
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
    }


def test_cpu_file_resolves_without_runner_labels():
    tests = [_make("tests/fast/test_a.py", backend=HWBackend.CPU, suite="stage-a-cpu")]
    plan = plan_file_run(tests, "tests/fast/test_a.py", "pr-42")
    assert plan == {
        "hw": "cpu",
        "suite": "stage-a-cpu",
        "runs_on": "",
        "container_image": "radixark/miles:pr-42",
    }


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
    with pytest.raises(FileRunError, match="has no CI registration"):
        plan_file_run([_make("tests/e2e/x/test_a.py")], "tests/e2e/x/test_missing.py", "dev")


def test_rocm_only_file_is_a_hard_error():
    tests = [_make("tests/e2e/x/test_a.py", backend=HWBackend.ROCM, suite="stage-c-8-gpu-mi350")]
    with pytest.raises(FileRunError, match="registered only for ROCm"):
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
    assert workflow.count("actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683") == 2
    assert "ref: ${{ inputs.head_sha }}" in workflow
    assert "path: pr-source" in workflow
    assert "CI_SOURCE_ROOT: ${{ github.workspace }}/pr-source" in workflow
    assert "run: python3 -S -m tests.ci.file_run" in workflow
    assert "DISPATCHED_SHA" not in workflow
    assert workflow.count("checkout_ref: ${{ inputs.head_sha }}") == 2
    assert "secrets: inherit" not in workflow
    assert "CI_COMMAND_APP_PRIVATE_KEY" not in workflow
    assert "NEON_DATABASE_URL" not in workflow
    assert workflow.count("--files '${{ inputs.test_file }}'") == 2
    assert "ref: ${{ inputs.checkout_ref || github.sha }}" in gpu_workflow
    assert "ref: ${{ inputs.checkout_ref || github.sha }}" in cpu_workflow
    assert (
        "GITHUB_COMMIT_NAME: ${{ inputs.checkout_ref || github.sha }}_"
        "${{ github.event.pull_request.number || github.event.inputs.pull_number || 'non-pr' }}"
    ) in gpu_workflow


def test_trusted_resolver_and_dependencies_have_workflow_owners():
    codeowners = (Path(__file__).parents[3] / ".github/CODEOWNERS").read_text()
    owners = "@yushengsu-thu @guapisolo @yueming-yuan"
    for path in (
        "/tests/__init__.py",
        "/tests/ci/__init__.py",
        "/tests/ci/file_run.py",
        "/tests/ci/ci_register.py",
        "/tests/ci/labels.py",
        "/tests/ci/test/test_file_run.py",
    ):
        assert f"{path} {owners}" in codeowners
