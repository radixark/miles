import argparse
import json
import math
import os
import shlex

from tests.ci.ci_policy import resolve_policy
from tests.ci.ci_register import HWBackend
from tests.ci.file_run import collect_snapshot_tests
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import filter_tests


# Leave headroom below GITHUB_TOKEN's 24-hour lifetime (and the 5-day job limit).
MAX_BATCH_MINUTES = 23 * 60


def plan_jobs(registrations, suites, cadence, labels, match_all_labels=False):
    policy = resolve_policy(cadence, set(labels))
    selected_labels = set(policy.include_labels)
    if match_all_labels:
        selected_labels.update(KNOWN_LABELS)
    jobs = []
    for suite in suites:
        assert suite in ("stage-c-4-gpu-b200", "stage-c-8-gpu-b200"), suite
        selected, _ = filter_tests(
            registrations,
            HWBackend.CUDA,
            suite,
            policy.admit_nightly_tests,
            selected_labels,
            policy.dispatch_arches,
            policy.absorb,
        )
        for test in selected:
            jobs.append(
                {
                    "file": test.filename,
                    "shell_file": shlex.quote(test.filename),
                    "suite": suite,
                    "num_gpus": test.required_gpus,
                    "runs_on": json.dumps(["b200", f"{test.required_gpus}gpu"]),
                    "timeout_minutes": math.ceil((max(1800, test.est_time * 1.25) + 1200) / 60),
                    "est_time": test.est_time,
                }
            )
    assert len(jobs) <= 256, "B200 test plan exceeds GitHub's 256-job matrix limit"
    assert len({job["file"] for job in jobs}) == len(jobs), "B200 test selected more than once"
    return sorted(jobs, key=lambda job: (-job["est_time"], job["file"]))


def batch_jobs(jobs):
    batches = []
    current = []
    minutes = 0
    for job in jobs:
        budget = job["timeout_minutes"]
        assert budget <= MAX_BATCH_MINUTES, f"B200 test exceeds batch budget: {job['file']} ({budget} minutes)"
        if current and minutes + budget > MAX_BATCH_MINUTES:
            batches.append(current)
            current = []
            minutes = 0
        current.append(job)
        minutes += budget
    if current:
        batches.append(current)

    result = []
    for index, batch in enumerate(batches):
        # Any file can run last: cover all peers' execution/setup/cleanup, even without overlap.
        timeout = sum(job["timeout_minutes"] for job in batch)
        matrix = {"include": [{**job, "timeout_minutes": timeout} for job in batch]}
        result.append({"batch": index + 1, "jobs": json.dumps(matrix)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--cadence", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--suites", nargs="*", required=True)
    parser.add_argument("--match-all-labels", action="store_true")
    args = parser.parse_args()
    jobs = plan_jobs(
        collect_snapshot_tests(args.source_root),
        args.suites,
        args.cadence,
        args.labels,
        args.match_all_labels,
    )
    matrix = json.dumps({"include": batch_jobs(jobs)})
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={matrix}\nhas_tests={str(bool(jobs)).lower()}\n")
    print(matrix)


if __name__ == "__main__":
    main()
