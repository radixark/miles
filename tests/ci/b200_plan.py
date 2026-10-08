import argparse
import json
import math
import os

from tests.ci.ci_policy import resolve_policy
from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import filter_tests


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadence", required=True)
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--suites", nargs="*", required=True)
    parser.add_argument("--match-all-labels", action="store_true")
    args = parser.parse_args()
    jobs = plan_jobs(
        collect_tests(discover_ci_files(), sanity_check=True),
        args.suites,
        args.cadence,
        args.labels,
        args.match_all_labels,
    )
    matrix = json.dumps({"include": jobs})
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={matrix}\nhas_tests={str(bool(jobs)).lower()}\n")
    print(matrix)


if __name__ == "__main__":
    main()
