"""GPU generations a CUDA test can run on, and which generation each stage is.

* An *arch* is a GPU generation, not a SKU: `h100` and `h200` differ in memory,
  which the suite name already encodes, not in the kernels they support.
* `register_cuda_ci(hardware=[...])` declares the arches a test's kernel and
  precision paths support. Which stage runs it is a separate question.
* Stdlib only. `tests/ci/file_run.py` imports this chain on a bare hosted runner
  before any dependency install, so it cannot reach into `miles.*`.
"""

__all__ = ["KNOWN_ARCHES", "CUDA_STAGE_ARCH", "auto_arch"]

# Order is the auto-dispatch preference: a test supporting several arches runs
# on the first one listed unless a PR asks otherwise.
KNOWN_ARCHES: tuple[str, ...] = ("hopper", "blackwell")

CUDA_STAGE_ARCH: dict[str, str] = {
    "stage-b-2-gpu-h200": "hopper",
    "stage-c-2-gpu-h200": "hopper",
    "stage-c-4-gpu-h200": "hopper",
    "stage-c-8-gpu-h100": "hopper",
    "stage-c-8-gpu-h200": "hopper",
    "stage-c-8-gpu-b200": "blackwell",
}


def auto_arch(hardware: list[str]) -> str:
    """The arch a test runs on when nothing asks for a different one."""
    supported = set(hardware)
    for arch in KNOWN_ARCHES:
        if arch in supported:
            return arch
    raise ValueError(f"no known arch in {sorted(supported)}")
