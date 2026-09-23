"""GPU generations a CUDA test can run on, and what each CUDA stage provides.

* An *arch* is a GPU generation, not a SKU: `h100` and `h200` differ in memory,
  which the suite name already encodes, not in the kernels they support.
* `register_cuda_ci(hardware=[...])` declares the arches a test's kernel and
  precision paths support. Which stage runs it is a separate question, answered
  by `dispatch_targets` from the test's home suite plus the run's dispatch policy.
* `CUDA_STAGES` is the single source of truth for the CUDA stage taxonomy;
  `run_suite.CI_SUITES` and `file_run.CUDA_SUITE_RUNS_ON` derive from it.
* Stdlib only. `tests/ci/file_run.py` imports this chain on a bare hosted runner
  before any dependency install, so it cannot reach into `miles.*`.
"""

from dataclasses import dataclass

__all__ = [
    "KNOWN_ARCHES",
    "CUDA_STAGES",
    "CudaStage",
    "auto_arch",
    "dispatch_targets",
    "target_stage",
]

# Order is the auto-dispatch preference: a test supporting several arches runs
# on the first one listed unless a PR asks otherwise. Hopper leads because one
# Blackwell host against four-plus Hopper hosts makes any balanced split
# saturate the Blackwell queue.
KNOWN_ARCHES: tuple[str, ...] = ("hopper", "blackwell")


@dataclass(frozen=True)
class CudaStage:
    arch: str
    num_gpus: int
    runs_on: tuple[str, ...]


CUDA_STAGES: dict[str, CudaStage] = {
    "stage-b-2-gpu-h200": CudaStage("hopper", 2, ("h200", "2gpu")),
    "stage-c-8-gpu-h100": CudaStage("hopper", 8, ("h100", "8gpu")),
    "stage-c-8-gpu-h200": CudaStage("hopper", 8, ("h200", "8gpu")),
    "stage-c-4-gpu-h200": CudaStage("hopper", 4, ("h200", "4gpu")),
    "stage-c-2-gpu-h200": CudaStage("hopper", 2, ("h200", "2gpu")),
    "stage-c-8-gpu-b200": CudaStage("blackwell", 8, ("b200", "8gpu")),
}


def auto_arch(hardware: list[str]) -> str:
    """The arch a test runs on when nothing asks for a different one."""
    supported = set(hardware)
    for arch in KNOWN_ARCHES:
        if arch in supported:
            return arch
    raise ValueError(f"no known arch in {sorted(supported)}")


def target_stage(home_suite: str, arch: str) -> str | None:
    """The stage on `arch` that runs a test homed at `home_suite`.

    On the test's own arch that is its home stage. On another arch it is the
    smallest stage with enough GPUs: a test declares its own budget through
    `ray start --num-gpus` / `torchrun --nproc-per-node` rather than reading the
    devices it can see, so a larger stage just leaves the surplus idle. None
    when that arch has no stage big enough.

    Derived rather than tabulated. A hand-written home -> destination map would
    encode a fleet shape that does not exist yet; this rule gives today's
    degenerate answer (one Blackwell stage absorbs everything) and a 1:1 mirror
    once the Blackwell fleet is partitioned, from the same three lines.
    """
    home = CUDA_STAGES[home_suite]
    if home.arch == arch:
        return home_suite
    fits = [name for name, stage in CUDA_STAGES.items() if stage.arch == arch and stage.num_gpus >= home.num_gpus]
    # Tie-break on the name so two stages of equal width choose deterministically.
    return min(fits, key=lambda name: (CUDA_STAGES[name].num_gpus, name), default=None)


def dispatch_targets(
    home_suite: str,
    hardware: list[str],
    *,
    dispatch_arches: frozenset[str],
    absorb: bool,
) -> set[str]:
    """Every CUDA stage that runs this registration under one dispatch policy.

    Empty `dispatch_arches` is AUTO: the test runs once, on `auto_arch`, which
    the home-stage invariant makes equal to its home stage's arch. `absorb` is
    what allows a test to leave its home stage at all; without it a requested
    arch that is not the test's own simply selects nothing.
    """
    supported = set(hardware)
    arches = (supported & dispatch_arches) if dispatch_arches else {auto_arch(supported)}

    targets: set[str] = set()
    for arch in arches:
        if CUDA_STAGES[home_suite].arch == arch:
            targets.add(home_suite)
        elif absorb:
            destination = target_stage(home_suite, arch)
            if destination is not None:
                targets.add(destination)
    return targets
