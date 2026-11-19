"""Resolve one explicitly requested test file to its CI execution plan.

Consumed by the `resolve-file-run` job of `.github/workflows/run-ci-file.yml`,
which runs on a bare hosted runner before any dependency install; this module
may import only the stdlib and the dependency-free registry modules.
"""

import ast
import json
import os
import re
import shlex
import sys
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, file_timeout
from tests.ci.hardware import CUDA_STAGES
from tests.ci.rocm import ROCM_FILE_RUN_LIMIT_SECONDS, ROCM_STAGES
from tests.ci.rocm import container_image as rocm_container_image

CPU_SUITES = frozenset({"stage-a-cpu", "stage-b-cpu"})
DISCOVERY_ROOTS = ("tests/fast", "tests/fast-gpu", "tests/e2e", "tests/ci")

CUDA_SUITE_RUNS_ON = {name: list(stage.runs_on) for name, stage in CUDA_STAGES.items()}

# Same shape the pr-test.yml resolve-ci-image step enforces for a Docker tag.
_IMAGE_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


class FileRunError(Exception):
    pass


class RocmFileTimeoutError(FileRunError):
    def __init__(self, filename: str, timeout: int):
        self.timeout = timeout
        super().__init__(
            f"{filename} needs {timeout}s, beyond the {ROCM_FILE_RUN_LIMIT_SECONDS}s ROCm file budget "
            "under Spur's 3-hour job limit"
        )


def _discover_regular_ci_files() -> list[str]:
    """Discover test files without following symlinks in scanned roots."""
    files = []
    for root in DISCOVERY_ROOTS:
        component = Path()
        for part in Path(root).parts:
            component /= part
            if component.is_symlink():
                raise FileRunError(f"CI discovery path must not be a symlink: {component.as_posix()}")
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            for name in directory_names:
                path = Path(directory, name)
                if path.is_symlink():
                    raise FileRunError(f"CI test directory must not be a symlink: {path.as_posix()}")
            for name in file_names:
                if not name.startswith("test_") or not name.endswith(".py"):
                    continue
                path = Path(directory, name)
                if path.is_symlink():
                    raise FileRunError(f"CI test file must not be a symlink: {path.as_posix()}")
                files.append(path.as_posix())
    return sorted(files)


def plan_file_run(
    all_tests, test_file: str, image_tag: str, *, hw: str = "auto", image_repo: str = "radixark/miles"
) -> dict[str, str]:
    """Map one registered test file to the hw/suite/runner/image that runs it.

    An explicit file request is the selection, so domain labels and the
    nightly cadence gate do not apply; only `disabled` still blocks the run.
    AUTO preserves CPU/CUDA selection. An explicit ROCm request uses the file's
    one enabled ROCm registration. Ambiguous selections are errors.
    """
    if hw not in {"auto", "rocm"}:
        raise FileRunError(f"unsupported file-run hardware: {hw!r}")
    if hw == "rocm":
        try:
            image = rocm_container_image(image_tag)
        except ValueError as error:
            raise FileRunError(str(error)) from error
    else:
        if _IMAGE_TAG_PATTERN.fullmatch(image_tag) is None:
            raise FileRunError(f"invalid CI image tag: {image_tag!r}")
        image = f"{image_repo}:{image_tag}"
    registrations = [t for t in all_tests if t.filename == test_file]
    if not registrations:
        raise FileRunError(f"{test_file} has no CI registration; /rerun-test runs only registered test files")
    requested = {HWBackend.CPU, HWBackend.CUDA} if hw == "auto" else {HWBackend.ROCM}
    supported = [t for t in registrations if t.backend in requested]
    if not supported:
        if hw == "auto":
            raise FileRunError(f"{test_file} is registered only for ROCm; use /rerun-test {test_file} --hw rocm")
        raise FileRunError(f"{test_file} has no {hw.upper()} CI registration")
    if hw == "rocm":
        enabled = [t for t in supported if t.disabled is None]
        if not enabled:
            raise FileRunError(f"{test_file} is disabled: {supported[0].disabled}")
        supported = enabled
    if len(supported) > 1:
        suites = ", ".join(sorted(f"{t.backend.name}:{t.suite}" for t in supported))
        scope = "CPU/CUDA" if hw == "auto" else hw.upper()
        raise FileRunError(f"{test_file} has multiple {scope} registrations ({suites}); expected exactly one")
    registration = supported[0]
    if registration.disabled is not None:
        raise FileRunError(f"{test_file} is disabled: {registration.disabled}")

    timeout = file_timeout(registration.est_time, registration.labels, 1800)
    if registration.backend is HWBackend.CUDA:
        num_gpus = registration.required_gpus
        runs_on = CUDA_SUITE_RUNS_ON.get(registration.suite)
        if runs_on is None:
            raise FileRunError(f"CUDA suite {registration.suite} has no runner mapping in CUDA_SUITE_RUNS_ON")
        if CUDA_STAGES[registration.suite].arch == "blackwell":
            runs_on = ["b200", f"{num_gpus}gpu"]
        hw = "cuda"
        runs_on_json = json.dumps(runs_on)
    elif registration.backend is HWBackend.ROCM:
        stage = ROCM_STAGES.get(registration.suite)
        if stage is None:
            raise FileRunError(f"ROCm suite {registration.suite} has no runner mapping in ROCM_STAGES")
        if timeout > ROCM_FILE_RUN_LIMIT_SECONDS:
            raise RocmFileTimeoutError(test_file, timeout)
        hw = "rocm"
        num_gpus = stage.num_gpus
        runs_on_json = json.dumps(stage.runs_on)
    else:
        if registration.suite not in CPU_SUITES:
            raise FileRunError(f"CPU suite {registration.suite} is not allowed for an explicit file run")
        hw = "cpu"
        num_gpus = 0
        runs_on_json = ""
    plan = {
        "hw": hw,
        "num_gpus": str(num_gpus),
        "suite": registration.suite,
        "runs_on": runs_on_json,
        "container_image": image,
        "timeout_seconds": str(timeout),
    }
    if hw == "rocm":
        # Both PR and comment callers execute this command in the PR snapshot.
        plan["execute_command"] = shlex.join(
            [
                "python3",
                "tests/ci/run_suite.py",
                "--hw",
                "rocm",
                "--suite",
                plan["suite"],
                "--test-file",
                test_file,
                "--cadence",
                "release",
                "--timeout-per-file",
                plan["timeout_seconds"],
            ]
        )
    return plan


def _read_snapshot_labels() -> dict[str, str]:
    path = Path("tests/ci/labels.py")
    if path.is_symlink():
        raise FileRunError(f"CI label registry must not be a symlink: {path}")
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions = [
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "KNOWN_LABELS"
        ]
        if len(definitions) != 1:
            raise FileRunError(f"{path} must define KNOWN_LABELS once as a literal dictionary")
        labels = ast.literal_eval(definitions[0])
    except (OSError, SyntaxError, ValueError, TypeError) as error:
        raise FileRunError(f"cannot read CI label registry {path}: {error}") from error
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
    ):
        raise FileRunError(f"{path}: KNOWN_LABELS must be a literal string-to-string dictionary")
    return labels


def collect_snapshot_tests(source_root: str | Path):
    """Parse a source snapshot as data, without importing its Python modules."""
    try:
        root = Path(source_root).resolve(strict=True)
    except OSError as error:
        raise FileRunError(f"cannot resolve CI source root {source_root}: {error}") from error
    if not root.is_dir():
        raise FileRunError(f"CI source root is not a directory: {root}")

    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        files = _discover_regular_ci_files()
        return collect_tests(files, sanity_check=True, known_labels=_read_snapshot_labels())
    finally:
        os.chdir(previous_directory)


def resolve_file_run(
    test_file: str,
    image_tag: str,
    source_root: str | Path = ".",
    *,
    hw: str = "auto",
    image_repo: str = "radixark/miles",
) -> dict[str, str]:
    return plan_file_run(collect_snapshot_tests(source_root), test_file, image_tag, hw=hw, image_repo=image_repo)


def main() -> int:
    hw = os.environ.get("CI_HW") or "auto"
    image_tag = os.environ.get("CI_IMAGE_TAG") or ("dev" if hw == "auto" else "")
    try:
        plan = resolve_file_run(
            os.environ["TEST_FILE"],
            image_tag,
            os.environ.get("CI_SOURCE_ROOT") or ".",
            hw=hw,
            image_repo=os.environ["GITHUB_REPOSITORY"],
        )
    except FileRunError as error:
        if isinstance(error, RocmFileTimeoutError):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                output.write(f"rocm_timeout={error.timeout}/{ROCM_FILE_RUN_LIMIT_SECONDS}\n")
        print(f"::error::{error}", file=sys.stderr)
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for key, value in plan.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
