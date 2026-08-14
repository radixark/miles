"""Tests for PR changed-path to GPU-stage selection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.ci.ci_policy import REGULAR_CADENCE, resolve_policy
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.stage_selection import PR_GPU_STAGES, ChangedFile, read_changed_files, select_skipped_gpu_stages

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _registration(
    filename: str,
    suite: str,
    *,
    labels: list[str] | None = None,
    disabled: str | None = None,
) -> CIRegistry:
    backend = HWBackend.ROCM if suite == "stage-c-4-gpu-mi350" else HWBackend.CUDA
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=1,
        suite=suite,
        labels=["precision"] if labels is None else labels,
        disabled=disabled,
    )


def _all_runnable_registrations() -> list[CIRegistry]:
    return [_registration(f"tests/e2e/{stage}.py", stage) for stage in PR_GPU_STAGES]


def _select(
    changed_files: tuple[ChangedFile, ...] | None,
    registrations: list[CIRegistry],
    *,
    raw_labels: tuple[str, ...] = ("run-ci-precision",),
    event_name: str = "pull_request",
) -> tuple[str, ...]:
    return select_skipped_gpu_stages(
        event_name=event_name,
        changed_files=changed_files,
        registrations=registrations,
        run_policy=resolve_policy(REGULAR_CADENCE, set(raw_labels)),
        raw_labels=raw_labels,
    )


def test_read_changed_files_preserves_rename_ends_and_special_names(tmp_path):
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(b"M\0docs/line\nname.md\0R100\0docs/old.md\0docs/new.md\0")

    assert read_changed_files(str(diff_path)) == (
        ChangedFile(status="M", paths=("docs/line\nname.md",)),
        ChangedFile(status="R100", paths=("docs/old.md", "docs/new.md")),
    )


@pytest.mark.parametrize("payload", [b"", b"M\0docs/index.md", b"Q\0docs/index.md\0", b"M\0\xff\0"])
def test_read_changed_files_fails_open_on_unusable_input(tmp_path, payload):
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(payload)

    assert read_changed_files(str(diff_path)) is None


def test_docs_tooling_and_cpu_only_tests_skip_every_gpu_stage():
    cpu_test = "tests/fast/doc/test_sync_example_docs.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        CIRegistry(HWBackend.CPU, cpu_test, 1, "stage-a-cpu"),
    ]
    changed_files = (
        ChangedFile("M", ("docs/ci/00-stage.md",)),
        ChangedFile("M", ("examples/README.md",)),
        ChangedFile("M", (".pre-commit-config.yaml",)),
        ChangedFile("M", ("scripts/tools/sync_example_docs.py",)),
        ChangedFile("M", (cpu_test,)),
    )

    assert set(_select(changed_files, registrations, raw_labels=())) == PR_GPU_STAGES


def test_changed_multi_backend_test_and_explicit_domain_run_their_stages():
    shared_test = "tests/e2e/test_shared.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration(shared_test, "stage-c-8-gpu-h100"),
        _registration(shared_test, "stage-c-4-gpu-mi350"),
    ]

    skipped = set(_select((ChangedFile("M", (shared_test,)),), registrations))

    assert skipped == PR_GPU_STAGES - {
        "stage-b-2-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-4-gpu-mi350",
    }


def test_domain_label_adds_only_stages_with_matching_tests():
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration("tests/e2e/test_megatron.py", "stage-c-8-gpu-h100", labels=["megatron"]),
    ]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    skipped = set(_select(changed_files, registrations, raw_labels=("run-ci-megatron",)))

    assert skipped == PR_GPU_STAGES - {"stage-c-8-gpu-h100"}


def test_broad_scope_adds_every_runnable_stage():
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration("tests/e2e/test_megatron.py", "stage-c-8-gpu-h100", labels=["megatron"]),
    ]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    skipped = set(_select(changed_files, registrations, raw_labels=("run-ci-all",)))

    assert skipped == PR_GPU_STAGES - {"stage-b-2-gpu-h200", "stage-c-8-gpu-h100"}


def test_bypass_fastfail_does_not_make_a_docs_change_affect_gpu_stages():
    registrations = [_registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200")]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    assert set(_select(changed_files, registrations, raw_labels=("bypass-fastfail",))) == PR_GPU_STAGES


def test_unknown_source_path_affects_every_runnable_gpu_stage():
    assert (
        _select(
            (ChangedFile("M", ("miles/trainer.py",)),),
            _all_runnable_registrations(),
        )
        == ()
    )


def test_rename_from_unknown_source_path_affects_every_runnable_stage():
    changed_files = (ChangedFile("R100", ("miles/old.py", "docs/old-api.md")),)

    assert _select(changed_files, _all_runnable_registrations()) == ()


@pytest.mark.parametrize(
    ("event_name", "changed_files"),
    [("schedule", (ChangedFile("M", ("docs/index.md",)),)), ("workflow_dispatch", ()), ("pull_request", None)],
)
def test_non_pr_or_missing_diff_never_prunes(event_name, changed_files):
    assert _select(changed_files, _all_runnable_registrations(), event_name=event_name) == ()


def test_changed_labeled_test_without_its_label_keeps_current_selection_semantics():
    changed_test = "tests/e2e/test_megatron.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration(changed_test, "stage-c-8-gpu-h100", labels=["megatron"]),
    ]

    assert set(_select((ChangedFile("M", (changed_test,)),), registrations)) == (
        PR_GPU_STAGES - {"stage-b-2-gpu-h200"}
    )


def test_cli_publishes_all_gpu_stages_for_docs_diff(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(b"M\0docs/ci/00-stage.md\0")
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": "[]",
            "CHANGED_FILES_PATH": str(diff_path),
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert set(json.loads(outputs["skipped_stages"])) == PR_GPU_STAGES
