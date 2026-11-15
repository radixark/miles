"""ROCm suite allocations and image policy shared by stage and file CI.

Stdlib only: the trusted file resolver runs before installing dependencies.
"""

import os
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RocmStage:
    num_gpus: int
    runs_on: tuple[str, ...] = ("linux-miles-mi35x-spur",)


# The unprefixed suite serves pr-test-rocm.yml. The nightly suites serve
# SGLang's external AMD nightly workflow. Both are available to file runs.
ROCM_PR_SUITE = "stage-c-4-gpu-mi350"
ROCM_STAGES = {
    ROCM_PR_SUITE: RocmStage(4),
    "nightly-stage-c-8-gpu-mi350": RocmStage(8),
    "nightly-stage-c-4-gpu-mi350": RocmStage(4),
    "nightly-stage-c-2-gpu-mi350": RocmStage(2),
}
DEFAULT_IMAGE_TAG = "miles-rocm10-mi35x"
ROCM_CHANGED_FILE_LIMIT = 4
# Spur's default TIME_LIMIT leaves 15 minutes for the image pull and setup.
ROCM_FILE_RUN_LIMIT_SECONDS = 3 * 3600 - 15 * 60


def container_image(tag: str = "") -> str:
    tag = tag or DEFAULT_IMAGE_TAG
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag) is None:
        raise ValueError(f"invalid ROCm CI image tag: {tag!r}")
    return f"rocm/sgl-dev:{tag}"


def main() -> int:
    try:
        image = container_image(os.environ.get("INPUT_CI_IMAGE_TAG", ""))
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"container_image={image}\n")
    print(f"Resolved CI image: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
