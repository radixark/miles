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


# Run by pr-test-rocm.yml and, with --cadence nightly, by the external
# sgl-project/sglang MI350 nightly. File runs use the same suites.
ROCM_STAGES = {
    "stage-c-2-gpu-mi350": RocmStage(2),
    "stage-c-4-gpu-mi350": RocmStage(4),
    "stage-c-8-gpu-mi350": RocmStage(8),
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


def pr_image_tag(pr_body: str) -> str:
    # ci-image-tag: names a radixark/miles (CUDA) tag, so ROCm reads its own directive.
    match = re.search(r"^ci-rocm-image-tag:[ \t]+(\S+)", pr_body, flags=re.MULTILINE)
    return match.group(1) if match else ""


def main() -> int:
    # pr-test.yml's order without its pr-<num> step, since PRs build no ROCm image:
    # the called workflow's image_tag or the dispatch input, the PR description,
    # then DEFAULT_IMAGE_TAG.
    tag = os.environ.get("INPUT_CI_IMAGE_TAG", "") or pr_image_tag(os.environ.get("PR_BODY", ""))
    try:
        image = container_image(tag)
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"container_image={image}\n")
    print(f"Resolved CI image: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
