import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WORKFLOW = "run-ci-file.yml"
ACTIVE_STATUSES = ("requested", "waiting", "pending", "queued", "in_progress")


def github_request(path, *, method="GET"):
    request = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if method == "POST":
            if response.status != 202:
                raise RuntimeError(f"Cancellation returned HTTP {response.status}, expected 202")
            return None
        return json.load(response)


def cancel_merged_file_runs(event, request=github_request):
    if event["action"] != "closed" or event["pull_request"]["merged"] is not True:
        return []

    repo = event["repository"]["full_name"]
    pull_number = event["number"]
    # Dispatch runs belong to main, so head_sha/pull_requests cannot identify
    # their PR. run-ci-file.yml puts the dispatch input in this exact run name.
    title = re.compile(rf"/rerun-test \S+ \(PR #{pull_number}\)")
    runs = {}
    for status in ACTIVE_STATUSES:
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {"event": "workflow_dispatch", "status": status, "per_page": 100, "page": page}
            )
            result = request(f"repos/{repo}/actions/workflows/{WORKFLOW}/runs?{query}")
            if result["total_count"] > 1000:
                raise RuntimeError(f"File-run listing exceeds GitHub's 1000-run limit for {status}")
            for run in result["workflow_runs"]:
                if (
                    run["path"] == f".github/workflows/{WORKFLOW}"
                    and run["event"] == "workflow_dispatch"
                    and run["status"] != "completed"
                    and title.fullmatch(run["display_title"])
                ):
                    runs[run["id"]] = run
            if len(result["workflow_runs"]) < 100:
                break

    # Finish pagination before cancelling: removals from active-status lists
    # would otherwise shift later pages and leave queued runs behind.
    cancelled = []
    failed = []
    for run_id in runs:
        path = f"repos/{repo}/actions/runs/{run_id}"
        try:
            try:
                request(f"{path}/cancel", method="POST")
            except urllib.error.HTTPError as error:
                if error.code != 409 or request(path)["status"] != "completed":
                    raise
                print(f"File run {run_id} already completed")
            else:
                print(f"Requested cancellation of file run {run_id} for PR #{pull_number}")
                cancelled.append(run_id)
        except Exception as error:
            print(f"Failed to cancel file run {run_id}: {error}")
            failed.append(run_id)
    if failed:
        raise RuntimeError(f"Failed to cancel file runs for PR #{pull_number}: {failed}")
    return cancelled


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    cancel_merged_file_runs(event)
