#!/usr/bin/env python3

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_ROOT = "https://api.github.com"
REPOSITORY = "radixark/miles"
REPOSITORY_ID = 1072725553
CLEAR_COMMAND = "clear-labels"
RERUN_COMMAND = "rerun-failed-ci"
CLEAR_EXACT_LABELS = frozenset({"nightly", "bypass-fastfail"})
RERUN_WORKFLOWS = (
    ("pre-commit.yml", ".github/workflows/pre-commit.yml"),
    ("pr-test.yml", ".github/workflows/pr-test.yml"),
    ("pr-test-rocm.yml", ".github/workflows/pr-test-rocm.yml"),
)
COMMAND_PATTERN = re.compile(r"/(run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*)")
LABEL_PATTERN = re.compile(r"run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
POLICY_PERMISSIONS = frozenset({"write", "admin"})


class CommentLabelError(Exception):
    pass


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CommentLabelError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise CommentLabelError(f"non-standard JSON number: {value}")


def load_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommentLabelError(f"cannot read JSON from {path}: {error}") from error


def _validate_permissions(name, values):
    if not isinstance(values, list) or not values:
        raise CommentLabelError(f"{name} must be a non-empty array")
    if any(type(value) is not str or value not in POLICY_PERMISSIONS for value in values):
        raise CommentLabelError(f"{name} must contain only write or admin")
    if len(set(values)) != len(values):
        raise CommentLabelError(f"{name} contains duplicate permissions")
    return frozenset(values)


def load_policy(path):
    raw = load_json(path)
    expected_keys = {"version", "labels", "clear_permissions", "rerun_permissions"}
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise CommentLabelError("policy must contain only version, labels, clear_permissions, and rerun_permissions")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise CommentLabelError("policy version must be 1")
    if not isinstance(raw["labels"], dict) or not raw["labels"]:
        raise CommentLabelError("policy labels must be a non-empty object")

    labels = {}
    for label, permissions in raw["labels"].items():
        if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
            raise CommentLabelError(f"invalid exact CI label: {label!r}")
        labels[label] = _validate_permissions(f"labels.{label}", permissions)

    return {
        "clear_permissions": _validate_permissions("clear_permissions", raw["clear_permissions"]),
        "labels": labels,
        "rerun_permissions": _validate_permissions("rerun_permissions", raw["rerun_permissions"]),
    }


def parse_command(body):
    if not isinstance(body, str):
        raise CommentLabelError("comment body must be a string")
    command = body.strip()
    if command == f"/{CLEAR_COMMAND}":
        return CLEAR_COMMAND
    if command == f"/{RERUN_COMMAND}":
        return RERUN_COMMAND
    match = COMMAND_PATTERN.fullmatch(command)
    if match is None:
        raise CommentLabelError("comment must contain only /run-ci-<key>, /clear-labels, or /rerun-failed-ci")
    return match.group(1)


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise CommentLabelError(f"{name} must be a positive integer")
    return value


def parse_event(event):
    if not isinstance(event, dict) or event.get("action") != "created":
        raise CommentLabelError("event must be issue_comment.created")

    repository = event.get("repository")
    if not isinstance(repository, dict):
        raise CommentLabelError("event repository is missing")
    if repository.get("id") != REPOSITORY_ID or repository.get("full_name") != REPOSITORY:
        raise CommentLabelError("event repository does not match radixark/miles")

    issue = event.get("issue")
    if not isinstance(issue, dict) or not isinstance(issue.get("pull_request"), dict):
        raise CommentLabelError("comment is not attached to a pull request")
    pull_number = _positive_int(issue.get("number"), "pull request number")

    comment = event.get("comment")
    if not isinstance(comment, dict) or not isinstance(comment.get("user"), dict):
        raise CommentLabelError("comment author is missing")
    if comment["user"].get("type") != "User":
        raise CommentLabelError("comment author must be a human GitHub user")
    actor_id = _positive_int(comment["user"].get("id"), "comment author ID")
    actor_login = comment["user"].get("login")
    if not isinstance(actor_login, str) or not actor_login:
        raise CommentLabelError("comment author login is missing")

    sender = event.get("sender")
    if not isinstance(sender, dict) or sender.get("id") != actor_id:
        raise CommentLabelError("event sender does not match the comment author")

    return pull_number, actor_id, actor_login, parse_command(comment.get("body"))


class GitHubAPI:
    def __init__(self, token):
        if not token:
            raise CommentLabelError("GitHub API token is missing")
        self.token = token

    def _request(self, path, *, method="GET", payload=None, expected_status=None, expect_json=True):
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if expected_status is not None and response.status != expected_status:
                    raise CommentLabelError(f"GitHub API returned HTTP {response.status}; expected {expected_status}")
                body = response.read()
                if not expect_json:
                    if body.strip():
                        raise CommentLabelError("GitHub API returned an unexpected response body")
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise CommentLabelError(f"GitHub API returned HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise CommentLabelError(f"GitHub API request failed: {error.reason}") from error
        except TimeoutError as error:
            raise CommentLabelError("GitHub API request timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CommentLabelError("GitHub API returned invalid JSON") from error

    def get_pull(self, pull_number):
        return self._request(f"/repos/{REPOSITORY}/pulls/{pull_number}")

    def get_permission(self, actor_login):
        encoded_login = urllib.parse.quote(actor_login, safe="")
        return self._request(f"/repos/{REPOSITORY}/collaborators/{encoded_login}/permission")

    def add_label(self, pull_number, label):
        return self._request(
            f"/repos/{REPOSITORY}/issues/{pull_number}/labels",
            method="POST",
            payload={"labels": [label]},
        )

    def remove_label(self, pull_number, label):
        encoded_label = urllib.parse.quote(label, safe="")
        return self._request(
            f"/repos/{REPOSITORY}/issues/{pull_number}/labels/{encoded_label}",
            method="DELETE",
        )

    def list_workflow_runs(self, workflow_file, head_sha):
        encoded_workflow = urllib.parse.quote(workflow_file, safe="")
        runs = []
        total_count = None
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {
                    "event": "pull_request",
                    "head_sha": head_sha,
                    "page": page,
                    "per_page": 100,
                }
            )
            result = self._request(f"/repos/{REPOSITORY}/actions/workflows/{encoded_workflow}/runs?{query}")
            if not isinstance(result, dict):
                raise CommentLabelError("GitHub API returned an invalid workflow-run page")
            page_total = result.get("total_count")
            page_runs = result.get("workflow_runs")
            if type(page_total) is not int or page_total < 0 or page_total > 1000:
                raise CommentLabelError("GitHub API returned an invalid workflow-run count")
            if not isinstance(page_runs, list):
                raise CommentLabelError("GitHub API returned invalid workflow runs")
            if total_count is None:
                total_count = page_total
            elif page_total != total_count:
                raise CommentLabelError("workflow-run count changed during pagination")
            runs.extend(page_runs)
            if len(runs) >= total_count:
                if len(runs) != total_count:
                    raise CommentLabelError("GitHub API returned too many workflow runs")
                return runs
            if not page_runs:
                raise CommentLabelError("GitHub API returned an incomplete workflow-run listing")
        raise CommentLabelError("workflow-run listing exceeded the 1000-run API limit")

    def list_pulls_for_head(self, owner_login, head_ref):
        pulls = []
        for page in range(1, 11):
            query = urllib.parse.urlencode(
                {
                    "head": f"{owner_login}:{head_ref}",
                    "page": page,
                    "per_page": 100,
                    "state": "all",
                }
            )
            result = self._request(f"/repos/{REPOSITORY}/pulls?{query}")
            if not isinstance(result, list):
                raise CommentLabelError("GitHub API returned an invalid head pull-request page")
            pulls.extend(result)
            if len(result) < 100:
                return pulls
        raise CommentLabelError("head pull-request listing exceeded the 1000-result limit")

    def rerun_failed_jobs(self, run_id):
        self._request(
            f"/repos/{REPOSITORY}/actions/runs/{run_id}/rerun-failed-jobs",
            method="POST",
            expected_status=201,
            expect_json=False,
        )


def _validate_live_pull(pull, pull_number):
    if not isinstance(pull, dict) or pull.get("number") != pull_number:
        raise CommentLabelError("GitHub API returned the wrong pull request")
    if pull.get("state") != "open":
        raise CommentLabelError("pull request is not open")

    base = pull.get("base")
    base_repository = base.get("repo") if isinstance(base, dict) else None
    if not isinstance(base_repository, dict):
        raise CommentLabelError("pull request base repository is missing")
    if base_repository.get("id") != REPOSITORY_ID or base_repository.get("full_name") != REPOSITORY:
        raise CommentLabelError("pull request base repository does not match radixark/miles")

    head = pull.get("head")
    head_repository = head.get("repo") if isinstance(head, dict) else None
    if not isinstance(head_repository, dict):
        raise CommentLabelError("pull request head repository is missing")
    head_repository_id = _positive_int(head_repository.get("id"), "pull request head repository ID")
    head_repository_owner = head_repository.get("owner")
    head_owner_login = head_repository_owner.get("login") if isinstance(head_repository_owner, dict) else None
    if not isinstance(head_owner_login, str) or not head_owner_login:
        raise CommentLabelError("pull request head repository owner is missing")
    head_ref = head.get("ref") if isinstance(head, dict) else None
    if not isinstance(head_ref, str) or not head_ref:
        raise CommentLabelError("pull request head ref is missing")

    labels = pull.get("labels")
    if not isinstance(labels, list) or any(not isinstance(item, dict) for item in labels):
        raise CommentLabelError("pull request labels are invalid")
    names = []
    for item in labels:
        name = item.get("name")
        if not isinstance(name, str):
            raise CommentLabelError("pull request label name is invalid")
        names.append(name)

    head_sha = head.get("sha") if isinstance(head, dict) else None
    if not isinstance(head_sha, str) or SHA_PATTERN.fullmatch(head_sha) is None:
        raise CommentLabelError("pull request head SHA is invalid")

    return frozenset(names), head_sha, head_repository_id, head_owner_login, head_ref


def resolve_policy(event, policy):
    pull_number, actor_id, actor_login, target = parse_event(event)
    if target == CLEAR_COMMAND:
        allowed_permissions = policy["clear_permissions"]
    elif target == RERUN_COMMAND:
        allowed_permissions = policy["rerun_permissions"]
    else:
        allowed_permissions = policy["labels"].get(target)
        if allowed_permissions is None:
            raise CommentLabelError("requested label is not exposed by policy")
    return pull_number, actor_id, actor_login, target, allowed_permissions


def require_permission(api, actor_id, actor_login, allowed_permissions):
    permission_result = api.get_permission(actor_login)
    if not isinstance(permission_result, dict):
        raise CommentLabelError("GitHub API returned an invalid repository permission")
    permission_user = permission_result.get("user")
    permission_user_id = permission_user.get("id") if isinstance(permission_user, dict) else None
    if type(permission_user_id) is not int or permission_user_id <= 0:
        raise CommentLabelError("GitHub API returned an invalid repository permission identity")
    if permission_user_id != actor_id:
        raise CommentLabelError("repository permission identity does not match the comment author")
    permission = permission_result.get("permission")
    if not isinstance(permission, str):
        raise CommentLabelError("GitHub API returned an invalid repository permission")
    if permission not in allowed_permissions:
        raise CommentLabelError("comment author is not authorized for the requested operation")


def _is_ci_control_label(label):
    return label.startswith("run-ci") or label in CLEAR_EXACT_LABELS


def _latest_failed_run(
    runs,
    workflow_path,
    pull_number,
    head_sha,
    head_repository_id,
    head_ref,
    allow_empty_pull_requests,
):
    associated = []
    seen_ids = set()
    for run in runs:
        if not isinstance(run, dict):
            raise CommentLabelError("GitHub API returned an invalid workflow run")
        run_id = _positive_int(run.get("id"), "workflow run ID")
        if run_id in seen_ids:
            raise CommentLabelError("GitHub API returned a duplicate workflow run")
        seen_ids.add(run_id)
        run_number = _positive_int(run.get("run_number"), "workflow run number")
        if run.get("path") != workflow_path:
            raise CommentLabelError("GitHub API returned a workflow run from an unexpected path")
        if run.get("event") != "pull_request" or run.get("head_sha") != head_sha:
            raise CommentLabelError("GitHub API returned a workflow run from an unexpected event or SHA")
        if run.get("head_branch") != head_ref:
            raise CommentLabelError("GitHub API returned a workflow run from an unexpected head ref")
        run_head_repository = run.get("head_repository")
        if not isinstance(run_head_repository, dict) or run_head_repository.get("id") != head_repository_id:
            raise CommentLabelError("GitHub API returned a workflow run from an unexpected repository")
        pull_requests = run.get("pull_requests")
        if not isinstance(pull_requests, list):
            raise CommentLabelError("GitHub API returned invalid workflow-run pull requests")
        pull_numbers = set()
        for pull_request in pull_requests:
            if not isinstance(pull_request, dict):
                raise CommentLabelError("GitHub API returned an invalid workflow-run pull request")
            pull_numbers.add(_positive_int(pull_request.get("number"), "workflow-run pull number"))
        if pull_number in pull_numbers or (not pull_numbers and allow_empty_pull_requests):
            associated.append((run_number, run_id, run))

    if not associated:
        return None
    _, run_id, latest = max(associated, key=lambda item: (item[0], item[1]))
    if latest.get("status") == "completed" and latest.get("conclusion") == "failure":
        return run_id
    return None


def _require_unique_head_pull(
    api,
    pull_number,
    head_sha,
    head_repository_id,
    head_owner_login,
    head_ref,
):
    pulls = api.list_pulls_for_head(head_owner_login, head_ref)
    if len(pulls) != 1:
        raise CommentLabelError("fork head does not identify exactly one pull request")
    pull = pulls[0]
    if not isinstance(pull, dict) or pull.get("number") != pull_number or pull.get("state") != "open":
        raise CommentLabelError("fork head identifies an unexpected pull request")
    base = pull.get("base")
    base_repository = base.get("repo") if isinstance(base, dict) else None
    head = pull.get("head")
    head_repository = head.get("repo") if isinstance(head, dict) else None
    head_repository_owner = head_repository.get("owner") if isinstance(head_repository, dict) else None
    if (
        not isinstance(base_repository, dict)
        or base_repository.get("id") != REPOSITORY_ID
        or base_repository.get("full_name") != REPOSITORY
        or not isinstance(head_repository, dict)
        or head_repository.get("id") != head_repository_id
        or not isinstance(head_repository_owner, dict)
        or head_repository_owner.get("login") != head_owner_login
        or head.get("ref") != head_ref
        or head.get("sha") != head_sha
    ):
        raise CommentLabelError("fork head pull-request identity does not match the live pull request")


def _rerun_failed_ci(
    api,
    pull_number,
    actor_id,
    actor_login,
    allowed_permissions,
    head_sha,
    head_repository_id,
    head_owner_login,
    head_ref,
):
    is_fork = head_repository_id != REPOSITORY_ID
    if is_fork:
        _require_unique_head_pull(
            api,
            pull_number,
            head_sha,
            head_repository_id,
            head_owner_login,
            head_ref,
        )
    candidates = []
    for workflow_file, workflow_path in RERUN_WORKFLOWS:
        runs = api.list_workflow_runs(workflow_file, head_sha)
        run_id = _latest_failed_run(
            runs,
            workflow_path,
            pull_number,
            head_sha,
            head_repository_id,
            head_ref,
            is_fork,
        )
        if run_id is not None:
            candidates.append((run_id, workflow_file, workflow_path))

    candidates.sort()
    if not candidates:
        require_permission(api, actor_id, actor_login, allowed_permissions)
    for run_id, workflow_file, workflow_path in candidates:
        current_pull = api.get_pull(pull_number)
        (
            _,
            current_head_sha,
            current_head_repository_id,
            current_head_owner_login,
            current_head_ref,
        ) = _validate_live_pull(current_pull, pull_number)
        if (
            current_head_sha != head_sha
            or current_head_repository_id != head_repository_id
            or current_head_owner_login != head_owner_login
            or current_head_ref != head_ref
        ):
            raise CommentLabelError("pull request head changed before rerun")
        if is_fork:
            _require_unique_head_pull(
                api,
                pull_number,
                head_sha,
                head_repository_id,
                head_owner_login,
                head_ref,
            )
        current_run_id = _latest_failed_run(
            api.list_workflow_runs(workflow_file, head_sha),
            workflow_path,
            pull_number,
            head_sha,
            head_repository_id,
            head_ref,
            is_fork,
        )
        if current_run_id != run_id:
            raise CommentLabelError("latest workflow-run state changed before rerun")
        require_permission(api, actor_id, actor_login, allowed_permissions)
        try:
            api.rerun_failed_jobs(run_id)
        except CommentLabelError as error:
            raise CommentLabelError(f"could not rerun failed jobs for workflow run {run_id}: {error}") from error

    return {
        "actor_id": actor_id,
        "decision": "ALLOW_RERUN_REQUESTED" if candidates else "ALLOW_NO_FAILED_RUNS",
        "head_sha": head_sha,
        "pull_number": pull_number,
        "workflow_run_ids": [candidate[0] for candidate in candidates],
    }


def process_event(event, policy, api):
    pull_number, actor_id, actor_login, target, allowed_permissions = resolve_policy(event, policy)

    pull = api.get_pull(pull_number)
    (
        current_labels,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
    ) = _validate_live_pull(pull, pull_number)

    if target == RERUN_COMMAND:
        return _rerun_failed_ci(
            api,
            pull_number,
            actor_id,
            actor_login,
            allowed_permissions,
            head_sha,
            head_repository_id,
            head_owner_login,
            head_ref,
        )

    require_permission(api, actor_id, actor_login, allowed_permissions)

    if target == CLEAR_COMMAND:
        labels_to_remove = sorted(label for label in current_labels if _is_ci_control_label(label))
        remaining_labels = current_labels
        for label in labels_to_remove:
            try:
                result = api.remove_label(pull_number, label)
            except CommentLabelError as error:
                raise CommentLabelError(f"could not remove CI label {label}: {error}") from error
            if (
                not isinstance(result, list)
                or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in result)
                or label in {item["name"] for item in result}
            ):
                raise CommentLabelError(f"GitHub API did not confirm removal of {label}")
            remaining_labels = frozenset(item["name"] for item in result)
        if any(_is_ci_control_label(label) for label in remaining_labels):
            raise CommentLabelError("GitHub API did not confirm that all CI labels were removed")
        return {
            "actor_id": actor_id,
            "decision": "ALLOW_CLEARED" if labels_to_remove else "ALLOW_ALREADY_CLEAR",
            "labels": labels_to_remove,
            "pull_number": pull_number,
        }

    label = target
    if label in current_labels:
        decision = "ALLOW_ALREADY_PRESENT"
    else:
        result = api.add_label(pull_number, label)
        if not isinstance(result, list) or label not in {
            item.get("name") for item in result if isinstance(item, dict)
        }:
            raise CommentLabelError("GitHub API did not confirm the requested label")
        decision = "ALLOW_ADDED"

    return {
        "actor_id": actor_id,
        "decision": decision,
        "label": label,
        "pull_number": pull_number,
    }


def authorize_policy(event, policy, api):
    pull_number, actor_id, actor_login, target, allowed_permissions = resolve_policy(event, policy)
    require_permission(api, actor_id, actor_login, allowed_permissions)
    return pull_number, actor_id, target


def main():
    try:
        event = load_json(os.environ["GITHUB_EVENT_PATH"])
        policy = load_policy(os.environ["CI_LABEL_POLICY_PATH"])
        api = GitHubAPI(os.environ["CI_LABEL_API_TOKEN"])
        if os.environ.get("CI_LABEL_PREFLIGHT") == "true":
            pull_number, actor_id, target = authorize_policy(event, policy, api)
            authorization = {"actor_id": actor_id, "pull_number": pull_number}
            if target in {CLEAR_COMMAND, RERUN_COMMAND}:
                authorization["command"] = f"/{target}"
            else:
                authorization["label"] = target
            capability = "actions" if target == RERUN_COMMAND else "issues"
            output_path = os.environ.get("GITHUB_OUTPUT")
            if output_path:
                try:
                    with Path(output_path).open("a", encoding="utf-8") as output:
                        output.write(f"capability={capability}\n")
                except OSError as error:
                    raise CommentLabelError(f"cannot write GITHUB_OUTPUT: {error}") from error
            print(
                json.dumps(
                    authorization,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        result = process_event(event, policy, api)
    except CommentLabelError as error:
        print(f"::error::{error}")
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
