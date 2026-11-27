import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from tests.ci.ci_policy import (
    CI_CADENCES,
    NIGHTLY_CADENCE,
    REGULAR_CADENCE,
    RunPolicy,
    registration_matches_selection,
    resolve_policy,
)
from tests.ci.ci_register import (
    CPU_FAST_EST_TIME_LIMIT_S,
    CIRegistry,
    HWBackend,
    collect_tests,
    discover_ci_files,
    file_timeout,
)
from tests.ci.ci_utils import (
    CI_GATE_RECORD_DIR_ENV,
    build_store_from_env,
    gate_provenance_from_env,
    reaping_is_isolated,
    run_unittest_files,
)
from tests.ci.hardware import CUDA_STAGES, dispatch_targets
from tests.ci.labels import KNOWN_LABELS, PERF_LABEL
from tests.ci.rocm import LEGACY_NIGHTLY_SUITES, ROCM_STAGES

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "rocm": HWBackend.ROCM,
}

# CI suites by hardware backend. Cadence is an eligibility filter within a
# suite, not a second suite inventory.
#
# CUDA suites derive from `hardware.CUDA_STAGES`, which also carries each
# stage's arch, GPU count and runner labels; each has a matching workflow job in
# .github/workflows/pr-test.yml or its B200 child workflow.
CI_SUITES = {
    HWBackend.CPU: [
        "stage-a-cpu",
        "stage-b-cpu",
    ],
    HWBackend.CUDA: list(CUDA_STAGES),
    HWBackend.ROCM: list(ROCM_STAGES),
}


def snapshot_hardware(suite: str) -> str:
    if suite in CI_SUITES[HWBackend.CPU]:
        return "cpu"
    if suite in CUDA_STAGES:
        return CUDA_STAGES[suite].runs_on[0]
    if suite in CI_SUITES[HWBackend.ROCM]:
        return suite.rsplit("-", maxsplit=1)[-1]
    raise ValueError(f"Unknown snapshot execution suite: {suite!r}")


def runs_in_stage(
    registration: CIRegistry,
    suite: str,
    *,
    dispatch_arches: frozenset[str] = frozenset(),
    absorb: bool = False,
) -> bool:
    """Whether `registration` executes in `suite`, ignoring label selection.

    CPU and ROCm registrations have no arch, so their home suite is the only
    answer, except that `run-ci-blackwell-only` (an arch restriction without
    `run-on-*`) selects no ROCm test. A CUDA registration's home stage is where
    AUTO sends it; an explicit `run-on-*` can send it to another arch's stage
    instead. Shared with `stage_selection` so runner allocation and test
    collection cannot disagree.
    """
    if registration.backend is HWBackend.ROCM:
        return registration.suite == suite and (absorb or not dispatch_arches)
    if registration.backend is not HWBackend.CUDA:
        return registration.suite == suite
    return suite in dispatch_targets(
        registration.suite,
        registration.hardware,
        dispatch_arches=dispatch_arches,
        absorb=absorb,
    )


def filter_tests(
    ci_tests: list[CIRegistry],
    hw: HWBackend,
    suite: str,
    admit_nightly_tests: bool = False,
    labels: set[str] | None = None,
    dispatch_arches: frozenset[str] = frozenset(),
    absorb: bool = False,
) -> tuple[list[CIRegistry], list[CIRegistry]]:
    """Filter registered tests down to the set that should run.

    The base predicate (hw / suite / cadence eligibility / disabled) is applied first.
    Label selection then keeps a test iff it declares no labels (the CPU
    always-on case) or any of its labels is in `labels` -- the effective
    include set from `resolve_policy` (the requested domain labels for a plain
    PR, near-total registry sets for broad scopes). GPU registrations require
    at least one label. There is no separate exclusion pass: a label a scope
    subtracted simply grants no inclusion, so a test whose only labels were
    subtracted drops out (including from the skip report), while a test that
    also carries an included label still runs.
    """
    valid_suites = CI_SUITES.get(hw, [])
    if suite not in valid_suites:
        raise ValueError(f"Unknown suite {suite} for backend {hw.name}")

    label_set: set[str] = labels or set()
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw
        and runs_in_stage(t, suite, dispatch_arches=dispatch_arches, absorb=absorb)
        and registration_matches_selection(
            t.labels,
            t.nightly,
            admit_nightly_tests=admit_nightly_tests,
            include_labels=label_set,
        )
    ]

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
    # sorted so every directory is named in one run of arguments: pytest loads a directory's
    # conftest once, and an argument list that returns to a directory after naming its parent
    # leaves the tests of that second visit without the fixtures their own conftest defines
    cmd = ["pytest", *sorted(filenames), "-v"]
    if not continue_on_error:
        cmd.append("-x")
    return cmd


def _warn_slow_cpu_files(
    junit_xml_path: Path,
    ci_tests: list[CIRegistry],
    threshold_s: float,
) -> None:
    """Emit non-failing GitHub Actions warnings for CPU test files exceeding `threshold_s`."""
    if not junit_xml_path.is_file():
        return
    try:
        root = ET.parse(junit_xml_path).getroot()
    except (ET.ParseError, OSError):
        return

    durations_by_file: dict[str, float] = defaultdict(float)
    for testcase in root.iter("testcase"):
        raw_file = testcase.attrib.get("file", "")
        raw_time = testcase.attrib.get("time", "")
        if not raw_file or not raw_time:
            continue
        try:
            elapsed = float(raw_time)
        except ValueError:
            continue
        norm_file = Path(raw_file).as_posix().removeprefix("./")
        durations_by_file[norm_file] += elapsed

    regs_by_file = {Path(t.filename).as_posix().removeprefix("./"): t for t in ci_tests}
    slow_files = sorted(
        ((filename, elapsed) for filename, elapsed in durations_by_file.items() if elapsed > threshold_s),
        key=lambda item: (-item[1], item[0]),
    )
    for filename, elapsed in slow_files:
        reg = regs_by_file.get(filename)
        if reg is not None:
            est_note = f"est_time={reg.est_time:g}s{' (implicit)' if reg.implicit else ''}"
        else:
            est_note = "unmapped registration"
        print(
            f"::warning file={filename}::{filename} took {elapsed:.1f}s in stage-a-cpu "
            f"(budget: {threshold_s:g}s, {est_note}). "
            f"Speed it up or register it in stage-b-cpu with domain labels "
            f"(see .claude/rules/ci-test-registration.md).",
            flush=True,
        )


def run_cpu_tests(
    ci_tests: list[CIRegistry],
    timeout_per_file: float,
    continue_on_error: bool,
    *,
    warn_slow_threshold_s: float | None = None,
) -> int:
    """Run a CPU suite in one pytest, except that each `perf` file runs alone within its file timeout."""
    shared = [t.filename for t in ci_tests if PERF_LABEL not in t.labels]
    junit_dir = (
        tempfile.TemporaryDirectory(prefix="miles-cpu-ci-junit-") if (shared and warn_slow_threshold_s) else None
    )
    junit_path = Path(junit_dir.name) / "junit.xml" if junit_dir is not None else None
    try:
        runs: list[tuple[list[str], tuple[str, int] | None]] = []
        if shared:
            shared_cmd = build_cpu_pytest_cmd(shared, continue_on_error)
            if junit_path is not None:
                shared_cmd.extend(["-o", "junit_family=xunit1", f"--junitxml={junit_path}"])
            runs.append((shared_cmd, None))
        for test in ci_tests:
            if PERF_LABEL in test.labels:
                timeout = file_timeout(test.est_time, test.labels, timeout_per_file)
                # The bound of a /rerun-test file run: TERM the process group, KILL it 30 s later.
                bound = ["timeout", "--signal=TERM", "--kill-after=30s", f"{timeout}s"]
                runs.append(
                    (bound + build_cpu_pytest_cmd([test.filename], continue_on_error), (test.filename, timeout))
                )
        status = 0
        for cmd, bounded in runs:
            print(f"Running: {' '.join(cmd)}", flush=True)
            returncode = subprocess.call(cmd)
            if bounded is not None and returncode == 124:  # coreutils `timeout` expired
                print(f"TIMEOUT: {bounded[0]} after {bounded[1]} seconds", flush=True)
            status = status or returncode
            if status and not continue_on_error:
                break
        if junit_path is not None and warn_slow_threshold_s is not None:
            _warn_slow_cpu_files(junit_path, ci_tests, warn_slow_threshold_s)
        return status
    finally:
        if junit_dir is not None:
            junit_dir.cleanup()


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
        policy.admit_nightly_tests,
        labels=include_labels,
        dispatch_arches=policy.dispatch_arches,
        absorb=policy.absorb,
    )

    if args.test_file:
        ci_tests = [test for test in ci_tests if test.filename == args.test_file]
        assert len(ci_tests) == 1, f"Selected test is not enabled in this suite: {args.test_file}"

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, policy, continue_on_error, ci_tests, skipped_tests)

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as output:
            output.write(f"has_tests={str(bool(ci_tests)).lower()}\n")

    if len(ci_tests) == 0:
        print("No tests to run. Exiting with success.", flush=True)
        return 0

    if args.list_only:
        return 0

    # CPU tests (fast/) use pytest; CUDA tests use python3 per-file
    if hw == HWBackend.CPU:
        warn_slow_threshold_s = CPU_FAST_EST_TIME_LIMIT_S if suite == "stage-a-cpu" else None
        return run_cpu_tests(
            ci_tests,
            args.timeout_per_file,
            continue_on_error,
            warn_slow_threshold_s=warn_slow_threshold_s,
        )

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    # Each backend uses its own optional Neon URL. Missing/empty URLs skip the
    # history gate. The resolved cadence supplies the baseline-writing signal;
    # provenance comes from the GitHub environment.
    gate_store = build_store_from_env(backend=hw)
    gate_provenance = gate_provenance_from_env()

    # The gate collects only when a record directory exists. CI does not set
    # MILES_CI_GATE_RECORD_DIR, so allocate a job-local one whenever a store is
    # configured (CUDA and ROCm suites; ROCm only records, see run_gate_hook). The training
    # subprocesses' CiHistoryBackend and the merge/gate steps read it from env.
    if (
        gate_store is not None
        and hw in (HWBackend.CUDA, HWBackend.ROCM)
        and not os.environ.get(CI_GATE_RECORD_DIR_ENV)
    ):
        os.environ[CI_GATE_RECORD_DIR_ENV] = tempfile.mkdtemp(prefix="miles-ci-gate-")

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
        gate_store=gate_store,
        gate_executing_suite=suite,
        snapshot_hardware=snapshot_hardware(suite) if SNAPSHOT_TESTING_ENABLED else None,
        gate_write_baseline=policy.write_baseline,
        gate_provenance=gate_provenance,
        reap_leftovers=reaping_is_isolated(),
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
    parser.add_argument("--test-file", help="Execute one file from the selected suite and policy.")
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
            "decide what runs. An empty list keeps only CPU registrations "
            "with no domain labels; it selects no GPU tests."
        ),
    )
    parser.add_argument(
        "--github-output",
        help="Append has_tests for the selected shard to a GitHub Actions output file; requires --list-only.",
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

    if args.github_output and not args.list_only:
        parser.error("--github-output requires --list-only.")
    if args.hw == "rocm":
        args.suite = LEGACY_NIGHTLY_SUITES.get(args.suite, args.suite)

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
