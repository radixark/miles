"""Resolve one explicitly requested test file to its CI execution plan.

Consumed by the `resolve-file-run` job of `.github/workflows/run-ci-file.yml`,
which runs on a bare hosted runner before any dependency install; this module
may import only the stdlib and the dependency-free registry modules.
"""

import json
import os
import re
import sys

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files

# Runner labels per CUDA suite, mirroring the pr-test.yml job wiring.
# `tests/ci/test/test_file_run.py` locks the key set to
# `run_suite.CI_SUITES[HWBackend.CUDA]` so a new suite cannot ship unmapped.
CUDA_SUITE_RUNS_ON = {
    "stage-b-2-gpu-h200": ["h200", "2gpu"],
    "stage-c-8-gpu-h100": ["h100", "8gpu"],
    "stage-c-8-gpu-h200": ["h200", "8gpu"],
    "stage-c-4-gpu-h200": ["h200", "4gpu"],
    "stage-c-2-gpu-h200": ["h200", "2gpu"],
}

# Same shape the pr-test.yml resolve-ci-image step enforces for a Docker tag.
_IMAGE_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


class FileRunError(Exception):
    pass


def plan_file_run(all_tests, test_file: str, image_tag: str) -> dict[str, str]:
    """Map one registered test file to the hw/suite/runner/image that runs it.

    An explicit file request is the selection, so domain labels and the
    nightly cadence gate do not apply; only `disabled` still blocks the run.
    Anything that cannot resolve to exactly one enabled CPU or CUDA
    registration is a hard error rather than a silent no-op.
    """
    if _IMAGE_TAG_PATTERN.fullmatch(image_tag) is None:
        raise FileRunError(f"invalid CI image tag: {image_tag!r}")
    registrations = [t for t in all_tests if t.filename == test_file]
    if not registrations:
        raise FileRunError(f"{test_file} has no CI registration; /run-ci runs only registered test files")
    supported = [t for t in registrations if t.backend in (HWBackend.CPU, HWBackend.CUDA)]
    if not supported:
        suites = ", ".join(sorted(t.suite for t in registrations))
        raise FileRunError(f"{test_file} is registered only for ROCm ({suites}); /run-ci supports CPU and CUDA")
    if len(supported) > 1:
        suites = ", ".join(sorted(f"{t.backend.name}:{t.suite}" for t in supported))
        raise FileRunError(f"{test_file} has multiple CPU/CUDA registrations ({suites}); expected exactly one")
    registration = supported[0]
    if registration.disabled is not None:
        raise FileRunError(f"{test_file} is disabled: {registration.disabled}")

    if registration.backend is HWBackend.CUDA:
        runs_on = CUDA_SUITE_RUNS_ON.get(registration.suite)
        if runs_on is None:
            raise FileRunError(f"CUDA suite {registration.suite} has no runner mapping in CUDA_SUITE_RUNS_ON")
        hw = "cuda"
        runs_on_json = json.dumps(runs_on)
    else:
        # CPU suites run on the hosted runner fixed by _run-cpu-ci.yml.
        hw = "cpu"
        runs_on_json = ""
    return {
        "hw": hw,
        "suite": registration.suite,
        "runs_on": runs_on_json,
        "container_image": f"radixark/miles:{image_tag}",
    }


def resolve_file_run(test_file: str, image_tag: str) -> dict[str, str]:
    return plan_file_run(collect_tests(discover_ci_files(), sanity_check=True), test_file, image_tag)


def main() -> int:
    try:
        plan = resolve_file_run(os.environ["TEST_FILE"], os.environ.get("CI_IMAGE_TAG") or "dev")
    except FileRunError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for key, value in plan.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
