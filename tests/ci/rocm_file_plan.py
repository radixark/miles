"""Plan the changed ROCm file jobs of an opted-in pull request."""

import json
import os

from tests.ci.ci_policy import RunPolicy, resolve_policy
from tests.ci.ci_register import CIRegistry, HWBackend, collect_tests, discover_ci_files
from tests.ci.file_run import FileRunError, RocmFileTimeoutError, plan_file_run
from tests.ci.rocm import ROCM_CHANGED_FILE_LIMIT, ROCM_STAGES, pr_image_tag
from tests.ci.run_suite import filter_tests
from tests.ci.stage_selection import ChangedFile, read_changed_files

_OPT_IN_LABELS = frozenset({"run-ci-amd", "run-ci-all", "nightly"})
_LONG_LABELS = frozenset({"long", "ft-long"})
_RERUN_HINT = "Use `/rerun-test <file> --hw rocm` to request an individual file."


def is_opted_in(raw_labels: set[str]) -> bool:
    return bool(raw_labels & _OPT_IN_LABELS)


def plan_jobs(
    changed_files: tuple[ChangedFile, ...],
    registrations: list[CIRegistry],
    policy: RunPolicy,
    raw_labels: set[str],
    skipped_stages: set[str],
    image_tag: str = "",
) -> tuple[list[dict[str, str]], list[str]]:
    # For renames and copies, only the destination exists in this plan's scope.
    paths = {change.paths[-1] for change in changed_files if change.status[0] in {"A", "M", "R", "C", "T"}}
    for suite in ROCM_STAGES:
        if suite not in skipped_stages:
            selected, _ = filter_tests(
                registrations, HWBackend.ROCM, suite, policy.admit_nightly_tests, set(policy.include_labels)
            )
            paths.difference_update(test.filename for test in selected)

    by_file: dict[str, list[CIRegistry]] = {}
    warnings = []
    for test in registrations:
        if test.filename not in paths or test.backend is not HWBackend.ROCM or test.disabled is not None:
            continue
        missing = {f"run-ci-{label}" for label in _LONG_LABELS.intersection(test.labels)} - raw_labels
        if missing:
            warnings.append(f"{test.filename} ({test.suite}) requires {', '.join(sorted(missing))}. {_RERUN_HINT}")
            continue
        by_file.setdefault(test.filename, []).append(test)

    if len(by_file) > ROCM_CHANGED_FILE_LIMIT:
        listed = "\n".join(f"  - `{path}`" for path in sorted(by_file))
        warnings.append(
            f"{len(by_file)} changed ROCm files exceed the limit of {ROCM_CHANGED_FILE_LIMIT}. "
            f"{_RERUN_HINT}\n{listed}"
        )
        return [], warnings

    jobs = []
    for path, tests in sorted(by_file.items()):
        try:
            jobs.append({"test_file": path, **plan_file_run(tests, path, image_tag, hw="rocm")})
        except RocmFileTimeoutError as error:
            warnings.append(f"Skipped {error}. Comment reruns share this file budget.")
        except FileRunError as error:
            warnings.append(f"Skipped {error}.")
    return jobs, warnings


def main() -> int:
    labels = set(os.environ.get("RAW_LABELS", "").split())
    changed_files = read_changed_files(os.environ.get("CHANGED_FILES_PATH", ""))
    jobs, warnings = [], []
    if changed_files and is_opted_in(labels):
        jobs, warnings = plan_jobs(
            changed_files,
            collect_tests(discover_ci_files(), sanity_check=True),
            resolve_policy(os.environ["CADENCE"], labels),
            labels,
            set(json.loads(os.environ["SKIPPED_STAGES"])),
            pr_image_tag(os.environ.get("PR_BODY", "")),
        )

    for warning in warnings:
        message = warning.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::warning::{message}")
    if warnings and (summary_path := os.environ.get("GITHUB_STEP_SUMMARY")):
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("### ROCm file planning\n\n" + "".join(f"- {warning}\n" for warning in warnings))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"rocm_file_jobs={json.dumps(jobs, separators=(',', ':'))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
