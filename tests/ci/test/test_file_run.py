"""Unit tests for `tests/ci/file_run.py`, the /run-ci resolve step."""

import json

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.file_run import CUDA_SUITE_RUNS_ON, FileRunError, main, plan_file_run, resolve_file_run
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
