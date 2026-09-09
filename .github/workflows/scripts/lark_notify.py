#!/usr/bin/env python3
"""
Post Miles CI health cards to a Lark group via an incoming webhook.

Used by .github/workflows/ci-lark-notify.yml. Needs GITHUB_TOKEN and
LARK_WEBHOOK, or --dry-run to print the card JSON instead of posting.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_REPO = "radixark/miles"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
GITHUB_API = "https://api.github.com"

FAILED_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}
# Aggregator jobs fail whenever any other job fails; listing them is noise.
AGGREGATOR_JOB_RE = re.compile(r"^(check-all-jobs|pr-test-finish)$")
MAX_LISTED_JOBS = 15


# --------------------------------------------------------------------------
# GitHub API
# --------------------------------------------------------------------------


class GitHub:
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    def get(self, path: str, params: dict | None = None, retries: int = 5) -> Any:
        url = f"{GITHUB_API}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                transient = e.code in (429, 502, 503, 504) or (e.code == 403 and "rate limit" in body.lower())
                if not transient or attempt == retries - 1:
                    raise RuntimeError(f"GET {url} -> {e.code}: {body[:300]}") from e
            except urllib.error.URLError as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"GET {url} failed: {e}") from e
            time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def paginate(self, path: str, key: str, params: dict | None = None, max_pages: int = 30) -> list:
        params = dict(params or {})
        params.setdefault("per_page", 100)
        items: list = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self.get(path, params)
            chunk = data.get(key, [])
            items.extend(chunk)
            if len(chunk) < params["per_page"]:
                break
        return items

    def run(self, run_id: int) -> dict:
        return self.get(f"repos/{self.repo}/actions/runs/{run_id}")

    def run_jobs(self, run_id: int) -> list:
        return self.paginate(
            f"repos/{self.repo}/actions/runs/{run_id}/jobs",
            "jobs",
            # latest attempt per job; jobs not rerun keep their earlier result
            {"filter": "latest"},
        )

    def run_attempt_jobs(self, run_id: int, attempt: int) -> list:
        return self.paginate(f"repos/{self.repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")


# --------------------------------------------------------------------------
# Lark card (schema 2.0)
# --------------------------------------------------------------------------


def md(text: str) -> dict:
    return {"tag": "markdown", "content": text}


def grey(text: str) -> str:
    return f"<font color='grey'>{text}</font>"


def kv_columns(pairs: list) -> dict:
    return {
        "tag": "column_set",
        "flex_mode": "flow",
        "horizontal_spacing": "default",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [md(f"{grey(k)}\n**{v}**")],
            }
            for k, v in pairs
        ],
    }


def button(text: str, url: str) -> dict:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": "default",
        "behaviors": [{"type": "open_url", "default_url": url}],
    }


HR = {"tag": "hr"}


def build_card(title: str, color: str, elements: list, buttons: list) -> dict:
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,  # red | orange | green | blue | grey
            },
            "body": {"elements": elements + [button(t, u) for t, u in buttons]},
        },
    }


def post_card(card: dict, webhook: str, dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(card, indent=2))
        return
    req = urllib.request.Request(
        webhook,
        data=json.dumps(card).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("code", body.get("StatusCode")) not in (0, None):
        raise RuntimeError(f"Lark webhook rejected message: {body}")
    print(f"posted: {card['card']['header']['title']['content']}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_local(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")


def ci_display_name(run: dict) -> str:
    if run["event"] != "schedule":
        return run["name"]
    scheduled_at = parse_time(run["created_at"])
    assert scheduled_at is not None
    # PR Test runs weekly on Saturday UTC and nightly Sunday through Friday.
    return "Weekly Test" if scheduled_at.weekday() == 5 else "Nightly Test"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def list_jobs_md(jobs: list, limit: int = MAX_LISTED_JOBS) -> str:
    lines = [f"- [{j['name']}]({j['html_url']})" for j in jobs[:limit]]
    if len(jobs) > limit:
        lines.append(f"- ... and {len(jobs) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# ci-status
# --------------------------------------------------------------------------


def is_reportable_job(job: dict) -> bool:
    return job.get("conclusion") not in (
        None,
        "skipped",
    ) and not AGGREGATOR_JOB_RE.match(job["name"])


def failed_job_names(jobs: list) -> dict:
    return {j["name"]: j for j in jobs if is_reportable_job(j) and j.get("conclusion") in FAILED_CONCLUSIONS}


def diff_attempts(current: dict, previous: dict) -> dict:
    return {
        "fixed": [j for n, j in previous.items() if n not in current],
        "still": [j for n, j in current.items() if n in previous],
        "new": [j for n, j in current.items() if n not in previous],
    }


def render_ci_status(run: dict, jobs: list, prev_failed: dict | None) -> dict:
    name = ci_display_name(run)
    attempt = run.get("run_attempt", 1)
    conclusion = run.get("conclusion") or "unknown"
    counted = [j for j in jobs if is_reportable_job(j)]
    failed = failed_job_names(jobs)
    cancelled = [j for j in jobs if j.get("conclusion") == "cancelled"]
    started = parse_time(run.get("run_started_at"))
    updated = parse_time(run.get("updated_at"))
    duration = fmt_duration((updated - started).total_seconds()) if started and updated else "-"

    repo_url = run["html_url"].split("/actions/")[0]
    sha = run["head_sha"]
    subject = ((run.get("head_commit") or {}).get("message") or "").splitlines()
    commit_md = f"[`{sha[:9]}`]({repo_url}/commit/{sha}) {subject[0] if subject else ''}"
    rerun_prefix = f"Rerun #{attempt} - " if attempt > 1 else ""

    if conclusion == "cancelled":
        title = f"{rerun_prefix}{name}: CANCELLED"
        color = "grey"
    elif failed:
        title = f"{rerun_prefix}{name}: FAILED ({len(failed)} of {plural(len(counted), 'job')})"
        color = "red"
    else:
        title = f"{rerun_prefix}{name}: PASSED ({plural(len(counted), 'job')})"
        color = "green"

    jobs_summary = f"{len(counted)} total, {len(failed)} failed"
    if cancelled:
        jobs_summary += f", {len(cancelled)} cancelled"
    commit_label = "Tested main commit" if run["event"] == "schedule" else "Commit"
    elements = [
        md(f"{grey(commit_label)}  {commit_md}"),
        kv_columns(
            [
                ("Started", fmt_local(started)),
                ("Finished", fmt_local(updated)),
                ("Duration", duration),
                ("Jobs", jobs_summary),
            ]
        ),
    ]

    sections = []
    # None: first attempt, nothing to compare against
    if prev_failed is None:
        if failed:
            sections.append(f"**Failed jobs ({len(failed)})**\n{list_jobs_md(list(failed.values()))}")
    else:
        diff = diff_attempts(failed, prev_failed)
        for key, heading in (
            ("fixed", "Fixed by rerun"),
            ("still", "Still failing"),
            ("new", "New failures"),
        ):
            if diff[key]:
                sections.append(f"**{heading} ({len(diff[key])})**\n{list_jobs_md(diff[key])}")
    if sections:
        elements.append(HR)
        elements.append(md("\n\n".join(sections)))

    buttons = [("View run on GitHub", run["html_url"])]
    if attempt > 1:
        buttons.append((f"View attempt {attempt - 1}", f"{run['html_url']}/attempts/{attempt - 1}"))
    return build_card(title, color, elements, buttons)


def cmd_ci_status(args: argparse.Namespace, gh: GitHub) -> None:
    run = gh.run(args.run_id)
    if run["event"] != "schedule" and not args.any_event:
        print(f"run {args.run_id} event={run['event']} is not a scheduled run; skipping")
        return
    if run.get("status") != "completed":
        print(f"run {args.run_id} status={run.get('status')} is not completed; skipping")
        return
    jobs = gh.run_jobs(run["id"])
    attempt = run.get("run_attempt", 1)
    prev_failed = None
    if attempt > 1:
        prev_failed = failed_job_names(gh.run_attempt_jobs(run["id"], attempt - 1))
    post_card(render_ci_status(run, jobs, prev_failed), args.webhook, args.dry_run)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--webhook", default=os.environ.get("LARK_WEBHOOK"))
    parser.add_argument("--dry-run", action="store_true", help="print card JSON instead of posting")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ci-status", help="summarize a finished scheduled run")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--any-event", action="store_true", help="also report non-schedule runs")

    args = parser.parse_args()
    if not args.token:
        print("GITHUB_TOKEN (or --token) is required", file=sys.stderr)
        return 2
    if not args.webhook and not args.dry_run:
        print("LARK_WEBHOOK (or --webhook) is required unless --dry-run", file=sys.stderr)
        return 2

    gh = GitHub(args.token, args.repo)
    cmd_ci_status(args, gh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
