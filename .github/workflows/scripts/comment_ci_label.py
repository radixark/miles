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
COMMAND_PATTERN = re.compile(r"/(run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*)")
LABEL_PATTERN = re.compile(r"run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*")
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
    if not isinstance(raw, dict) or set(raw) != {"version", "labels"}:
        raise CommentLabelError("policy must contain only version and labels")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise CommentLabelError("policy version must be 1")
    if not isinstance(raw["labels"], dict) or not raw["labels"]:
        raise CommentLabelError("policy labels must be a non-empty object")

    labels = {}
    for label, permissions in raw["labels"].items():
        if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
            raise CommentLabelError(f"invalid exact CI label: {label!r}")
        labels[label] = _validate_permissions(f"labels.{label}", permissions)

    return {"labels": labels}


def parse_command(body):
    if not isinstance(body, str):
        raise CommentLabelError("comment body must be a string")
    match = COMMAND_PATTERN.fullmatch(body.strip())
    if match is None:
        raise CommentLabelError("comment must contain only /run-ci-<key>")
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

    def _request(self, path, *, method="GET", payload=None):
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
                return json.loads(response.read().decode("utf-8"))
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
    _positive_int(head_repository.get("id"), "pull request head repository ID")

    labels = pull.get("labels")
    if not isinstance(labels, list) or any(not isinstance(item, dict) for item in labels):
        raise CommentLabelError("pull request labels are invalid")
    names = []
    for item in labels:
        name = item.get("name")
        if not isinstance(name, str):
            raise CommentLabelError("pull request label name is invalid")
        names.append(name)

    return frozenset(names)


def resolve_policy(event, policy):
    pull_number, actor_id, actor_login, label = parse_event(event)
    allowed_permissions = policy["labels"].get(label)
    if allowed_permissions is None:
        raise CommentLabelError("requested label is not exposed by policy")
    return pull_number, actor_id, actor_login, label, allowed_permissions


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
        raise CommentLabelError("comment author is not authorized for the requested label")


def process_event(event, policy, api):
    pull_number, actor_id, actor_login, label, allowed_permissions = resolve_policy(event, policy)

    pull = api.get_pull(pull_number)
    current_labels = _validate_live_pull(pull, pull_number)
    require_permission(api, actor_id, actor_login, allowed_permissions)

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
    pull_number, actor_id, actor_login, label, allowed_permissions = resolve_policy(event, policy)
    require_permission(api, actor_id, actor_login, allowed_permissions)
    return pull_number, actor_id, label


def main():
    try:
        event = load_json(os.environ["GITHUB_EVENT_PATH"])
        policy = load_policy(os.environ["CI_LABEL_POLICY_PATH"])
        api = GitHubAPI(os.environ["CI_LABEL_API_TOKEN"])
        if os.environ.get("CI_LABEL_PREFLIGHT") == "true":
            pull_number, actor_id, label = authorize_policy(event, policy, api)
            print(
                json.dumps(
                    {"actor_id": actor_id, "label": label, "pull_number": pull_number},
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
