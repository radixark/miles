import argparse
import subprocess
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

from tests.ci.ci_register import CIRegistry, HWBackend, collect_tests, discover_ci_files
from tests.ci.ci_utils import run_unittest_files
from tests.ci.labels import KNOWN_LABELS

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "rocm": HWBackend.ROCM,
}

# PR-side label prefix attached to every domain label. The workflow forwards
# only canonical, shell-safe CI labels; stripping stays here so selection is
# unit-testable.
_RUN_CI_PREFIX = "run-ci-"

# CI suites by hardware backend. Cadence is an eligibility filter within a
# suite, not a second suite inventory.
#
# CUDA suites: each is served by a matching workflow job in
# .github/workflows/pr-test.yml. `stage-c-8-gpu-h100` and `stage-c-8-gpu-h200`
# run on full-node 8-GPU hosts; the split H200 fleet is one 8-GPU node divided
# into 2+2+4 workers via per-runner CUDA_VISIBLE_DEVICES (see pr-test.yml
# stage-c-4-gpu-h200 / stage-b-2-gpu-h200 / stage-c-2-gpu-h200 job comments).
CI_SUITES = {
    HWBackend.CPU: [
        "stage-a-cpu",
        "stage-b-cpu",
    ],
    HWBackend.CUDA: [
        "stage-b-2-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-8-gpu-h200",
        "stage-c-4-gpu-h200",
        "stage-c-2-gpu-h200",
    ],
    HWBackend.ROCM: [
        "stage-c-8-gpu-mi350",
        "stage-c-4-gpu-mi350",
        "stage-c-2-gpu-mi350",
    ],
}

# PR labels that are workflow switches, not domain-label selectors: `nightly`
# is adapted to an explicit cadence by pr-test.yml; `bypass-fastfail` feeds
# the resolved run policy.
# `strip_run_ci_prefix` skips them without warning.
_WORKFLOW_ONLY_LABELS = {"nightly", "bypass-fastfail"}

REGULAR_CADENCE = "regular"
NIGHTLY_CADENCE = "nightly"
CI_CADENCES = frozenset({REGULAR_CADENCE, NIGHTLY_CADENCE})


@dataclass(frozen=True)
class RunPolicy:
    cadence: str
    include_labels: frozenset[str]
    bypass_fastfail: bool

    @property
    def is_nightly(self) -> bool:
        return self.cadence == NIGHTLY_CADENCE


def strip_run_ci_prefix(raw_labels: Iterable[str]) -> set[str]:
    """Strip the `run-ci-` prefix from each PR-side label.

    Inputs are the canonical PR-side CI label names forwarded by the workflow
    (e.g. `["run-ci-megatron", "nightly"]`). Empty input yields an empty set.
    Known workflow-only labels (`_WORKFLOW_ONLY_LABELS`) are consumed
    elsewhere and skipped silently; any other item missing the `run-ci-`
    prefix is skipped after a `warnings.warn(...)`, because silently
    including it would risk matching the wrong domain label (e.g. bare
    `"megatron"` colliding with a test's domain label by accident).
    """
    stripped: set[str] = set()
    for raw in raw_labels:
        if not raw or raw in _WORKFLOW_ONLY_LABELS:
            continue
        if raw.startswith(_RUN_CI_PREFIX):
            stripped.add(raw[len(_RUN_CI_PREFIX) :])
        else:
            warnings.warn(
                f"--labels entry {raw!r} is missing the expected {_RUN_CI_PREFIX!r} "
                f"prefix; ignoring. Domain labels must be raw `run-ci-<X>` strings.",
                stacklevel=2,
            )
    return stripped


def resolve_policy(cadence: str, raw_labels: set[str]) -> RunPolicy:
    """Resolve selection and within-stage failure behavior from explicit inputs.

    `pr-test.yml` adapts trigger-specific facts into a cadence and raw labels;
    this function never infers policy from a GitHub event name. A test runs iff
    it is cadence-eligible and declares no labels (always-run) or any of its
    labels is in the effective include set.

    Broad scopes are large include sets: `run-ci-all` includes every registered
    label, nightly cadence everything except `ft-long`, and `run-ci-image`
    everything except `ft-short` and `ft-long`. Branch order encodes the
    precedence `run-ci-all` > nightly > `run-ci-image`.

    Explicitly requested `run-ci-<x>` labels are unioned in last, so an
    explicit request always wins over a scope subtraction. A subtraction is
    not a per-test veto: a test carrying a subtracted label still runs when
    another of its labels is included.
    """
    if cadence not in CI_CADENCES:
        raise ValueError(f"Unknown CI cadence {cadence!r}; expected one of {sorted(CI_CADENCES)}")
    if "nightly" in raw_labels and cadence != NIGHTLY_CADENCE:
        raise ValueError("The nightly workflow label requires cadence='nightly'")

    requested = strip_run_ci_prefix(raw_labels) & set(KNOWN_LABELS)
    if "run-ci-all" in raw_labels:
        scope = set(KNOWN_LABELS)
    elif cadence == NIGHTLY_CADENCE:
        scope = set(KNOWN_LABELS) - {"ft-long"}
    elif "run-ci-image" in raw_labels:
        scope = set(KNOWN_LABELS) - {"ft-short", "ft-long"}
    else:
        scope = set()
    return RunPolicy(
        cadence=cadence,
        include_labels=frozenset(scope | requested),
        bypass_fastfail=cadence == NIGHTLY_CADENCE or "bypass-fastfail" in raw_labels,
    )


def filter_tests(
    ci_tests: list[CIRegistry],
    hw: HWBackend,
    suite: str,
    nightly: bool = False,
    labels: set[str] | None = None,
) -> tuple[list[CIRegistry], list[CIRegistry]]:
    """Filter registered tests down to the set that should run.

    The base predicate (hw / suite / cadence eligibility / disabled) is applied first.
    Label selection then keeps a test iff it declares no labels (always-run)
    or any of its labels is in `labels` -- the effective include set from
    `resolve_policy` (the requested domain labels for a plain PR, near-total
    registry sets for broad scopes). There is no separate exclusion pass: a
    label a scope subtracted simply grants no inclusion, so a test whose
    only labels were subtracted drops out (including from the skip report),
    while a test that also carries an included label still runs.
    """
    valid_suites = CI_SUITES.get(hw, [])
    if suite not in valid_suites:
        raise ValueError(f"Unknown suite {suite} for backend {hw.name}")

    ci_tests = [t for t in ci_tests if t.backend == hw and t.suite == suite and (not t.nightly or nightly)]

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


def pretty_print_tests(
    args,
    policy: RunPolicy,
    continue_on_error: bool,
    ci_tests: list[CIRegistry],
    skipped_tests: list[CIRegistry],
):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} " f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    msg = f"\n{'='*60}\n"
    msg += (
        f"Hardware: {hw.name}  Suite: {suite}  Cadence: {policy.cadence}  "
        f"Continue on error: {continue_on_error}  Partition: {partition_info}\n"
    )
    msg += f"{'='*60}\n"

    if skipped_tests:
        msg += f"Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for hw={hw.name}, suite={suite}, cadence={policy.cadence}\n"
        msg += "This is expected during incremental migration. Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += f"Enabled {len(ci_tests)} test(s) (est total {total_est_time:.0f}s):\n"
        for t in ci_tests:
            suffix = " [implicit]" if t.implicit else ""
            msg += f"  - {t.filename} (est_time={t.est_time}s){suffix}\n"

    print(msg, flush=True)


def build_cpu_pytest_cmd(filenames: list[str], continue_on_error: bool) -> list[str]:
    """Build the single pytest invocation for a CPU suite.

    `-x` (stop at first failure) is the default regular-run behavior. With
    continue_on_error -- e.g. a PR carrying the `bypass-fastfail` label -- drop
    `-x` so every file runs; pytest still exits non-zero if any failed, so the
    stage stays red.
    """
    cmd = ["pytest", *filenames, "-v"]
    if not continue_on_error:
        cmd.append("-x")
    return cmd


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    files = discover_ci_files()
    all_tests = collect_tests(files, sanity_check=True)
    policy = resolve_policy(args.cadence, set(args.labels or []))
    include_labels = set(policy.include_labels)
    if args.match_all_labels:
        include_labels |= set(KNOWN_LABELS)
    continue_on_error = args.continue_on_error or policy.bypass_fastfail
    print(
        f"Policy: cadence={policy.cadence!r} bypass_fastfail={policy.bypass_fastfail} "
        f"include_labels={sorted(include_labels)}",
        flush=True,
    )
    ci_tests, skipped_tests = filter_tests(
        all_tests,
        hw,
        suite,
        policy.is_nightly,
        labels=include_labels,
    )

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, policy, continue_on_error, ci_tests, skipped_tests)

    if len(ci_tests) == 0:
        print("No tests to run. Exiting with success.", flush=True)
        return 0

    if args.list_only:
        return 0

    # CPU tests (fast/) use pytest; CUDA tests use python3 per-file
    if hw == HWBackend.CPU:
        cmd = build_cpu_pytest_cmd([t.filename for t in ci_tests], continue_on_error)
        print(f"Running: {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd)

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=continue_on_error,
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
    cadence_group = parser.add_mutually_exclusive_group()
    cadence_group.add_argument(
        "--cadence",
        choices=sorted(CI_CADENCES),
        default=REGULAR_CADENCE,
        help="Explicit CI cadence resolved by the workflow (default: regular).",
    )
    cadence_group.add_argument(
        "--nightly",
        dest="cadence",
        action="store_const",
        const=NIGHTLY_CADENCE,
        help="Local alias for --cadence nightly; matches the nightly tag's selection and failure policy.",
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
            "decide what runs. An empty list keeps only registrations with "
            "no domain labels."
        ),
    )
    parser.add_argument(
        "--match-all-labels",
        action="store_true",
        default=False,
        help=(
            "Include every registered label, running every enabled test in "
            "the suite (subject to hw/suite/cadence/disabled). Manual "
            "override for local runs; the workflow passes resolved cadence "
            "and labels instead."
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
