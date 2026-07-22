import datetime
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from tests.ci.ci_register import CIRegistry, HWBackend
from tests.ci.metric_history import (
    NEON_DATABASE_URL_ENV,
    MetricSample,
    NeonMetricHistoryStore,
    RunIdentity,
    RunProvenance,
)
from tests.ci.metric_history.gate import evaluate_gate

# Env var the training process reads to find the per-attempt record directory; kept
# in sync with miles.utils.tracking_utils.ci_history.RECORD_DIR_ENV.
CI_GATE_RECORD_DIR_ENV = "MILES_CI_GATE_RECORD_DIR"


def _sanitize_for_path(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _attempt_record_dir(base_dir: str, filename: str, attempt: int) -> str:
    """Per-test, per-attempt subdir for CI metric-history records."""
    record_key = f"{_sanitize_for_path(filename)}-{hashlib.sha1(filename.encode()).hexdigest()[:10]}"
    return os.path.join(base_dir, record_key, f"attempt-{attempt}")


def _merge_attempt_records(attempt_dir: str, merged_path: str) -> None:
    """Merge every per-process JSONL file under `attempt_dir` into one record.

    Each per-process file holds lines of `{"metric": key, "series": [[step, value], ...]}`.
    The same metric key may appear in more than one file; concatenate
    their series and sort by step so the merged per-run record is coherent. Runs
    only for the PASSING attempt, right before the gate hook consumes the result.
    """
    if not os.path.isdir(attempt_dir):
        return
    merged: dict[str, list[list]] = {}
    for fname in sorted(os.listdir(attempt_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(attempt_dir, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                merged.setdefault(rec["metric"], []).extend(rec["series"])

    for points in merged.values():
        points.sort(key=lambda p: (p[0] is None, p[0]))

    with open(merged_path, "w", encoding="utf-8") as f:
        for metric, points in merged.items():
            f.write(json.dumps({"metric": metric, "series": points}) + "\n")


# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


# Patterns that indicate retriable accuracy/performance failures
RETRIABLE_PATTERNS = [
    r"AssertionError:.*not greater than",
    r"AssertionError:.*not less than",
    r"AssertionError:.*not equal to",
    r"AssertionError:.*!=.*expected",
    r"accuracy",
    r"score",
    r"latency",
    r"throughput",
    r"timeout",
]

# Patterns that indicate non-retriable failures (real code errors)
NON_RETRIABLE_PATTERNS = [
    r"SyntaxError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"NameError",
    r"TypeError",
    r"AttributeError",
    r"RuntimeError",
    r"CUDA out of memory",
    r"OOM",
    r"Segmentation fault",
    r"core dumped",
    r"ConnectionRefusedError",
    r"FileNotFoundError",
]


def is_retriable_failure(output: str) -> tuple[bool, str]:
    """
    Determine if a test failure is retriable based on output patterns.

    Returns:
        tuple: (is_retriable, reason)
    """
    # Check for non-retriable patterns first
    for pattern in NON_RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return False, f"non-retriable error: {pattern}"

    # Check for retriable patterns
    for pattern in RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return True, f"retriable pattern: {pattern}"

    # If we have an AssertionError but didn't match non-retriable, assume retriable
    if re.search(r"AssertionError", output):
        return True, "AssertionError (assuming retriable)"

    # Default: not retriable
    return False, "unknown failure type"


def _kill_process_tree(pgid: int):
    """Kill a process group by its PGID."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Error killing process group {pgid}: {e}")


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []
    exception_holder = []

    def _target_func():
        try:
            ret_value.append(func(*args, **(kwargs or {})))
        except Exception as e:
            exception_holder.append(e)

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if exception_holder:
        raise exception_holder[0]

    if not ret_value:
        raise RuntimeError("Thread completed but no return value or exception was captured.")

    return ret_value[0]


def write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def _gate_pr_number_from_env() -> int | None:
    """Parse the PR number out of `GITHUB_COMMIT_NAME` (`{sha}_{pr|non-pr}`).

    The workflow sets `GITHUB_COMMIT_NAME = {github.sha}_{pr_number||'non-pr'}`.
    A push / schedule run carries the `non-pr` sentinel and yields None.
    """
    commit_name = os.environ.get("GITHUB_COMMIT_NAME")
    if not commit_name or "_" not in commit_name:
        return None
    tail = commit_name.rsplit("_", 1)[1]
    if tail == "non-pr":
        return None
    try:
        return int(tail)
    except ValueError:
        return None


def _gate_commit_sha_from_env() -> str:
    """The run's commit sha: prefer `GITHUB_SHA`; fall back to the sha encoded
    in `GITHUB_COMMIT_NAME` (`{sha}_{pr|non-pr}`); else empty string."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    commit_name = os.environ.get("GITHUB_COMMIT_NAME")
    if commit_name and "_" in commit_name:
        return commit_name.rsplit("_", 1)[0]
    return commit_name or ""


def _gate_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def gate_provenance_from_env() -> RunProvenance:
    """Build a :class:`RunProvenance` from the GitHub Actions environment.

    Reads `GITHUB_SHA` (or the sha embedded in `GITHUB_COMMIT_NAME`),
    `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, `GITHUB_EVENT_NAME`,
    `GITHUB_REF`, and the PR number embedded in `GITHUB_COMMIT_NAME`. Missing
    values become `None` (run_id / attempt / pr_number) or an empty string
    (commit_sha) rather than raising -- provenance is audit metadata, never part
    of the baseline key.
    """
    return RunProvenance(
        commit_sha=_gate_commit_sha_from_env(),
        pr_number=_gate_pr_number_from_env(),
        github_run_id=_gate_int_env("GITHUB_RUN_ID"),
        github_run_attempt=_gate_int_env("GITHUB_RUN_ATTEMPT"),
        event_name=os.environ.get("GITHUB_EVENT_NAME"),
        ref=os.environ.get("GITHUB_REF"),
    )


def build_store_from_env():
    """Return the metric-history store for this environment, or None.

    A hosted store is used only when `NEON_DATABASE_URL` is set (CI with the
    secret inherited). Locally / in dev the var is unset and this returns None,
    which disables the gate hook entirely -- no store, no evaluation, no writes.

    Construction opens a DB connection eagerly. That happens outside the gate
    hook's try/except, so a missing driver / bad DSN / DB outage is caught here
    and degraded to None -- the gate must never fail a CI job (CUDA or ROCm)
    before any test runs.
    """
    if os.environ.get(NEON_DATABASE_URL_ENV):
        try:
            return NeonMetricHistoryStore()
        except Exception as e:  # noqa: BLE001 -- never let store setup fail CI
            logger.warning("[CI Gate] store unavailable (%s: %s); gate disabled.", type(e).__name__, e)
    return None


def _shadow_verdict_line(filename: str, result) -> str:
    """One-line human-readable shadow verdict for a PR run.

    Shadow runs never write a row and never change pass/fail; this is the only
    artifact they emit besides the per-metric detail.
    """
    verdict = "TRUSTED" if result.trusted else "NOT-TRUSTED"
    return f"[CI Gate][shadow] {filename}: {verdict} ({len(result.metrics)} metric(s))"


def run_gate_hook(
    filename: str,
    merged_record_path: str,
    *,
    store,
    registry: CIRegistry,
    nightly: bool,
    provenance: RunProvenance,
    now_iso: str | None = None,
) -> None:
    """Evaluate the history gate for one passed CUDA test and act on the verdict.

    NIGHTLY-marked run -> persist the run as a trusted/untrusted baseline via
    `store.write_run`, one `metric_values` row per coordinate (specs sharing a
    coordinate collapse to one row); a file that declares no gate writes
    nothing at all. ORDINARY PR run -> never write; log a shadow verdict and
    append it to `GITHUB_STEP_SUMMARY`.

    The entire body is wrapped: any gate or store error is caught and logged and
    NEVER propagates, so the gate verdict can never change the test's pass/fail
    this round.
    """
    try:
        result = evaluate_gate(filename, merged_record_path, store, registry=registry)

        if nightly:
            if not result.metrics:
                # Every spec yields at least one per-coordinate result, so an
                # empty list means the file declares no gate: an empty runs row
                # is nothing a baseline can use, so write nothing.
                logger.info(f"[CI Gate][nightly] {filename}: no gate declared; skipping write")
                return
            identity = RunIdentity(
                test_path=result.test_path,
                backend=result.backend,
                suite=result.suite,
            )
            created_at = now_iso or datetime.datetime.now(datetime.timezone.utc).isoformat()
            # Specs sharing a coordinate (identical declaration literals,
            # differing only in policy metadata) select the same value; writing
            # one row per spec would double-weight that baseline mean.
            seen_coords: set[tuple[str, str, str, int]] = set()
            values: list[MetricSample] = []
            for m in result.metrics:
                if m.current is None:
                    continue
                coord = (m.metric_key, m.steps_key, m.constraint_key, m.step)
                if coord in seen_coords:
                    continue
                seen_coords.add(coord)
                values.append(MetricSample(m.metric_key, m.steps_key, m.constraint_key, m.step, m.current))
            store.write_run(
                identity,
                provenance,
                created_at,
                trusted=result.trusted,
                values=values,
            )
            logger.info(
                f"[CI Gate][nightly] {filename}: wrote baseline " f"(trusted={result.trusted}, {len(values)} value(s))"
            )
        else:
            line = _shadow_verdict_line(filename, result)
            logger.info(line)
            detail = "\n".join(f"  - {m.metric_key} (step={m.step}): {m.reason}" for m in result.metrics)
            write_github_step_summary(line + ("\n" + detail if detail else "") + "\n")
    except Exception as e:
        # The gate is informational this round: a gate error, a missing record,
        # or a DB error must never fail the test or CI.
        logger.warning(f"[CI Gate] hook failed for {filename}: {type(e).__name__}: {e}")


def _gha_emit_group(title: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    safe = title.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::group::{safe}", flush=True)


def _gha_emit_endgroup() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    print("::endgroup::", flush=True)


def _gha_emit_summary(
    i: int,
    n: int,
    filename: str,
    status: str,
    elapsed: float,
    exit_code: int | None = None,
    timeout_after: float | None = None,
    retry_of: int | None = None,
) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    safe_name = filename.replace("\r", "\\r").replace("\n", "\\n")
    line = f"[{i}/{n}] {safe_name}  {status}  elapsed={int(elapsed)}s"
    if exit_code is not None:
        line += f" exit={int(exit_code)}"
    if timeout_after is not None:
        line += f" timeout_after={int(timeout_after)}s"
    if retry_of is not None:
        line += f" retry_of=attempt={int(retry_of)}"
    print(line, flush=True)


def run_unittest_files(
    files: list[TestFile] | list[CIRegistry],
    timeout_per_file: float,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_attempts: int = 2,
    retry_wait_seconds: int = 60,
    gate_store=None,
    gate_nightly: bool = False,
    gate_provenance: RunProvenance | None = None,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile or CIRegistry objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        enable_retry: If True, retry failed tests that appear to be accuracy/performance
                     assertion failures (not code errors).
        max_attempts: Maximum number of attempts per file including initial run (default: 2).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
        gate_store: Metric-history store for the regression gate, or None to skip
                    the gate hook entirely. Built by `build_store_from_env`.
        gate_nightly: True when this run writes trusted baselines (nightly);
                    False for an ordinary PR run, which only logs a shadow verdict.
        gate_provenance: RunProvenance for the gate write; defaults to
                    `gate_provenance_from_env()` when None.
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    retried_tests = []  # Track which tests were retried

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            filename, estimated_time = file.name, file.estimated_time

        effective_timeout = max(timeout_per_file, int(estimated_time * 1.25))

        process = None
        output_lines = []

        def run_one_file(filename, capture_output=False, record_dir=None, _i=i, _estimated_time=estimated_time):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(f".\n.\nBegin ({_i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n")
            file_tic = time.perf_counter()

            child_env = None
            if record_dir is not None:
                # Point the training process at this attempt's own record dir.
                os.makedirs(record_dir, exist_ok=True)
                child_env = os.environ.copy()
                child_env[CI_GATE_RECORD_DIR_ENV] = record_dir

            if capture_output:
                # Capture output for retry decision
                process = subprocess.Popen(
                    ["python3", full_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="ignore",
                    start_new_session=True,
                    env=child_env,
                )
                output_lines = []
                for line in process.stdout:
                    logger.info(line.rstrip())
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(
                    ["python3", full_path],
                    stdout=None,
                    stderr=None,
                    start_new_session=True,
                    env=child_env,
                )
                process.wait()

            elapsed = time.perf_counter() - file_tic

            logger.info(f".\n.\nEnd ({_i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {_estimated_time=}\n.\n.\n")
            return process.returncode

        # Retry loop for each file
        attempt = 1
        file_passed = False
        was_retried = False
        # Merged record path of the PASSING attempt only. Failed attempts keep
        # their per-process records on disk but are never merged or gated.
        passing_record_path: str | None = None

        while attempt <= (max_attempts if enable_retry else 1):
            if attempt > 1:
                logger.info(f"\n[CI Retry] Attempt {attempt}/{max_attempts} for {filename}\n")
                was_retried = True

            attempt_tic = time.perf_counter()
            current_attempt = attempt
            group_title = f"{filename}  ({i + 1}/{len(files)} est={int(estimated_time)}s attempt={current_attempt})"
            _gha_emit_group(group_title)
            attempt_status: str | None = None
            attempt_exit_code: int | None = None
            attempt_timeout_after: float | None = None
            attempt_elapsed: float = 0.0

            gate_base_dir = os.environ.get(CI_GATE_RECORD_DIR_ENV)
            attempt_record_dir = (
                _attempt_record_dir(gate_base_dir, filename, current_attempt) if gate_base_dir else None
            )

            try:
                try:
                    ret_code = run_with_timeout(
                        run_one_file,
                        args=(filename,),
                        kwargs={"capture_output": enable_retry, "record_dir": attempt_record_dir},
                        timeout=effective_timeout,
                    )
                    attempt_elapsed = time.perf_counter() - attempt_tic

                    if ret_code == 0:
                        attempt_status = "PASS"
                        file_passed = True
                        if attempt_record_dir is not None:
                            # The training process has exited, so its per-process
                            # JSONL records are complete. Merge the passing
                            # attempt's records into the one per-run record the
                            # gate consumes. Gate infrastructure: a merge I/O
                            # error must never propagate and change the test's
                            # pass/fail, so it is caught and logged and only
                            # skips the gate hook.
                            merged_path = f"{attempt_record_dir}.merged.jsonl"
                            try:
                                _merge_attempt_records(attempt_record_dir, merged_path)
                                passing_record_path = merged_path
                            except Exception as e:  # noqa: BLE001 -- gate infra must not affect pass/fail
                                logger.warning(
                                    "[CI Gate] record merge failed for %s: %s: %s", filename, type(e).__name__, e
                                )
                        if was_retried:
                            logger.info(f"\nPASSED on retry (attempt {attempt}): {filename}\n")
                            retried_tests.append((filename, attempt, "passed"))
                        passed_tests.append(filename)
                        break
                    else:
                        attempt_status = "FAIL"
                        attempt_exit_code = ret_code
                        # Check if we should retry
                        if enable_retry and attempt < max_attempts:
                            output = "".join(output_lines)
                            is_retriable, reason = is_retriable_failure(output)

                            if is_retriable:
                                logger.info(f"\n[CI Retry] {filename} failed with {reason}")
                                logger.info(f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n")
                                time.sleep(retry_wait_seconds)
                                attempt += 1
                                continue
                            else:
                                logger.info(f"\n[CI Retry] {filename} failed with {reason} - not retrying\n")

                        # No retry or not retriable
                        logger.info(f"\nFAILED: {filename} returned exit code {ret_code}\n")
                        if was_retried:
                            retried_tests.append((filename, attempt, "failed"))
                        failed_tests.append((filename, f"exit code {ret_code}"))
                        break

                except TimeoutError:
                    attempt_elapsed = time.perf_counter() - attempt_tic
                    attempt_status = "TIMEOUT"
                    attempt_timeout_after = effective_timeout
                    _kill_process_tree(process.pid)
                    time.sleep(5)
                    logger.info(f"\nTIMEOUT: {filename} after {effective_timeout} seconds\n")
                    if was_retried:
                        retried_tests.append((filename, attempt, "timeout"))
                    failed_tests.append((filename, f"timeout after {effective_timeout}s"))
                    break
                except Exception:
                    attempt_elapsed = time.perf_counter() - attempt_tic
                    attempt_status = "FAIL"
                    raise
            finally:
                _gha_emit_endgroup()
                if attempt_status is not None:
                    _gha_emit_summary(
                        i + 1,
                        len(files),
                        filename,
                        attempt_status,
                        elapsed=attempt_elapsed,
                        exit_code=attempt_exit_code,
                        timeout_after=attempt_timeout_after,
                        retry_of=(current_attempt - 1) if current_attempt >= 2 else None,
                    )

        # Gate hook (CUDA path only): only run_unittest_files dispatches CUDA
        # suites; CPU suites go through pytest in run_a_suite and never reach
        # here. Fire only on a PASS with a selected passing-attempt record and a
        # configured store. The hook never affects file_passed / success.
        if (
            file_passed
            and gate_store is not None
            and passing_record_path is not None
            and isinstance(file, CIRegistry)
            and file.backend == HWBackend.CUDA
        ):
            run_gate_hook(
                filename,
                passing_record_path,
                store=gate_store,
                registry=file,
                nightly=gate_nightly,
                provenance=gate_provenance or gate_provenance_from_env(),
            )

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Summary: {len(passed_tests)}/{len(files)} passed")
    if enable_retry and retried_tests:
        logger.info(f"Retries: {len(retried_tests)} test(s) were retried")
    logger.info(f"{'='*60}")
    if passed_tests:
        logger.info("PASSED:")
        for test in passed_tests:
            logger.info(f"  {test}")
    if failed_tests:
        logger.info("\nFAILED:")
        for test, reason in failed_tests:
            logger.info(f"  {test} ({reason})")
    if retried_tests:
        logger.info("\nRETRIED:")
        for test, attempts, result in retried_tests:
            logger.info(f"  {test} ({attempts} attempts, {result})")
    logger.info(f"{'='*60}\n")

    # Write GitHub Step Summary only if retries occurred
    if retried_tests:
        passed_on_retry = [t for t, _, r in retried_tests if r == "passed"]
        failed_after_retry = [t for t, _, r in retried_tests if r != "passed"]
        summary = f"**Retried {len(retried_tests)} test(s):**\n"
        if passed_on_retry:
            summary += f"- Passed on retry: {', '.join(passed_on_retry)}\n"
        if failed_after_retry:
            summary += f"- Still failed: {', '.join(failed_after_retry)}\n"
        write_github_step_summary(summary)

    return 0 if success else -1
