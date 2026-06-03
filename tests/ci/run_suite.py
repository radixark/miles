import argparse
import subprocess
import sys
import warnings
from collections.abc import Iterable

from tests.ci.ci_register import CIRegistry, HWBackend, collect_tests, discover_ci_files
from tests.ci.ci_utils import run_unittest_files

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
}

# PR-side label prefix the workflow attaches to every domain label and passes
# verbatim to `--labels`. Stripping is done here (not in YAML) so the filter
# is unit-testable and the workflow stays a thin pass-through.
_RUN_CI_PREFIX = "run-ci-"

# Per-commit test suites (run on every PR; per-domain selection is done at
# runtime by `filter_tests` via the `--labels` arg, not via per-suite jobs).
#
# CUDA suites: each is served by a matching workflow job in
# .github/workflows/pr-test.yml. `stage-c-8-gpu-h100` runs on the full-node
# 8-GPU H100 host; the H200 fleet is one 8-GPU node split into 2+2+4 workers
# via per-runner CUDA_VISIBLE_DEVICES (see pr-test.yml stage-c-4-gpu-h200 /
# stage-b-2-gpu-h200 / stage-c-2-gpu-h200 job comments).
PER_COMMIT_SUITES = {
    HWBackend.CPU: [
        "stage-a-cpu",
        "stage-b-cpu",
    ],
    HWBackend.CUDA: [
        "stage-b-2-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-4-gpu-h200",
        "stage-c-2-gpu-h200",
    ],
}

# Nightly test suites (placeholder for future use)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [],
}


def strip_run_ci_prefix(raw_labels: Iterable[str]) -> set[str]:
    """Strip the `run-ci-` prefix from each PR-side label.

    Inputs come straight from the workflow (e.g. `["run-ci-megatron",
    "run-ci-fsdp"]`). Empty input yields an empty set. Items missing the
    `run-ci-` prefix are skipped after emitting a `warnings.warn(...)` --
    the workflow contract requires every passed label to be a raw
    `run-ci-<X>` string, and silently including a non-prefixed item would
    risk matching the wrong domain label (e.g. bare `"megatron"` colliding
    with a test's domain label by accident).
    """
    stripped: set[str] = set()
    for raw in raw_labels:
        if not raw:
            continue
        if raw.startswith(_RUN_CI_PREFIX):
            stripped.add(raw[len(_RUN_CI_PREFIX) :])
        else:
            warnings.warn(
                f"--labels entry {raw!r} is missing the expected {_RUN_CI_PREFIX!r} "
                f"prefix; ignoring. The workflow must pass raw `run-ci-<X>` labels.",
                stacklevel=2,
            )
    return stripped


def filter_tests(
    ci_tests: list[CIRegistry],
    hw: HWBackend,
    suite: str,
    nightly: bool = False,
    labels: set[str] | None = None,
    match_all_labels: bool = False,
) -> tuple[list[CIRegistry], list[CIRegistry]]:
    """Filter registered tests down to the set that should run.

    The base predicate (hw / suite / nightly / disabled) is applied first.
    Label selection then narrows further, with two modes:

    * `match_all_labels=True`: ignore labels entirely -- every enabled test
      that matches hw/suite/nightly runs. Used for the `run-ci-image` /
      `run-ci-all` meta-labels and for `workflow_dispatch`. Precedence: this
      mode wins even when `labels` is also passed.
    * `match_all_labels=False` (default): include only tests where
      `not test.labels or (set(test.labels) & labels)`. `labels` here is
      the already-stripped domain-label set produced by
      `strip_run_ci_prefix`. A test registered with `labels=[]` (or
      omitted) is treated as always-run: it survives an empty PR-label
      set; a test with non-empty `labels` survives only when its labels
      intersect the PR-supplied set.
    """
    ci_tests = [t for t in ci_tests if t.backend == hw and t.suite == suite and t.nightly == nightly]

    valid_suites = NIGHTLY_SUITES.get(hw, []) if nightly else PER_COMMIT_SUITES.get(hw, [])

    if suite not in valid_suites:
        print(f"Warning: Unknown suite {suite} for backend {hw.name}, nightly={nightly}")

    if not match_all_labels:
        label_set: set[str] = labels or set()
        ci_tests = [t for t in ci_tests if not t.labels or (set(t.labels) & label_set)]

    enabled_tests = [t for t in ci_tests if t.disabled is None]
    skipped_tests = [t for t in ci_tests if t.disabled is not None]

    return enabled_tests, skipped_tests


def auto_partition(files: list[CIRegistry], rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using a greedy algorithm (LPT heuristic), and return the partition for the specified rank.
    """
    if not files or size <= 0:
        return []

    # Sort files by estimated_time in descending order (LPT heuristic).
    # Use filename as tie-breaker to ensure deterministic partitioning
    # regardless of glob ordering.
    sorted_files = sorted(files, key=lambda f: (-f.est_time, f.filename))

    partitions = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each file to the partition with the smallest current total time
    for file in sorted_files:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(file)
        partition_sums[min_sum_idx] += file.est_time

    if rank < size:
        return partitions[rank]
    return []


def pretty_print_tests(args, ci_tests: list[CIRegistry], skipped_tests: list[CIRegistry]):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} " f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    msg = f"\n{'='*60}\n"
    msg += f"Hardware: {hw.name}  Suite: {suite}  Nightly: {nightly}  Partition: {partition_info}\n"
    msg += f"{'='*60}\n"

    if skipped_tests:
        msg += f"Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}\n"
        msg += "This is expected during incremental migration. Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += f"Enabled {len(ci_tests)} test(s) (est total {total_est_time:.0f}s):\n"
        for t in ci_tests:
            suffix = " [implicit]" if t.implicit else ""
            msg += f"  - {t.filename} (est_time={t.est_time}s){suffix}\n"

    print(msg, flush=True)


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    files = discover_ci_files()
    all_tests = collect_tests(files, sanity_check=True)
    stripped_labels = strip_run_ci_prefix(args.labels or [])
    ci_tests, skipped_tests = filter_tests(
        all_tests,
        hw,
        suite,
        nightly,
        labels=stripped_labels,
        match_all_labels=args.match_all_labels,
    )

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, ci_tests, skipped_tests)

    if len(ci_tests) == 0:
        print("No tests to run. Exiting with success.", flush=True)
        return 0

    if args.list_only:
        return 0

    # CPU tests (fast/) use pytest; CUDA tests use python3 per-file
    if hw == HWBackend.CPU:
        cmd = ["pytest"] + [t.filename for t in ci_tests] + ["-x", "-v"]
        print(f"Running: {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd)

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description="Run CI test suites from tests/e2e/")
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Run nightly tests instead of per-commit tests.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="The time limit for running one file in seconds (default: 1800).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails.",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2).",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60).",
    )
    parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        default=False,
        help="Only list tests that would be run, do not execute them.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help=(
            "Raw PR-side labels (e.g. `run-ci-megatron run-ci-fsdp`). The "
            "`run-ci-` prefix is stripped on the Python side; the resulting "
            "domain-label set is intersected with each test's `labels` to "
            "decide what runs. An empty list keeps only `always_on=True` "
            "tests for the suite."
        ),
    )
    parser.add_argument(
        "--match-all-labels",
        action="store_true",
        default=False,
        help=(
            "Bypass the labels filter and run every enabled test in the "
            "suite (subject to hw/suite/nightly/disabled). Set by the "
            "workflow when the PR carries `run-ci-image` or `run-ci-all`, "
            "and equivalently on `workflow_dispatch`."
        ),
    )
    args = parser.parse_args()

    # Validate auto-partition arguments
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error("--auto-partition-id and --auto-partition-size must be specified together.")
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
