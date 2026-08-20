#!/usr/bin/env python3

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

API_ROOT = "https://api.github.com"
REPOSITORY = "radixark/miles"
REPOSITORY_ID = 1072725553
CLEAR_COMMAND = "clear-labels"
RERUN_COMMAND = "rerun-failed-ci"
RUN_FILE_COMMAND = "rerun-test"
RUN_FILE_WORKFLOW = "run-ci-file.yml"
RUN_FILE_WORKFLOW_REF = "main"
CLEAR_EXACT_LABELS = frozenset({"nightly", "bypass-fastfail"})
RERUN_WORKFLOWS = (
    ("pre-commit.yml", ".github/workflows/pre-commit.yml"),
    ("pr-test.yml", ".github/workflows/pr-test.yml"),
    ("pr-test-rocm.yml", ".github/workflows/pr-test-rocm.yml"),
)
COMMAND_PATTERN = re.compile(r"/(run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*|bypass-fastfail)")
# "/run-ci" remains the label-family marker, while "/rerun-test" owns targeted
# file runs; either family must parse as one exact command or fail loudly.
COMMAND_MARKERS = ("/run-ci", "/rerun-test", "/bypass-fastfail", "/clear-labels", "/rerun-failed-ci")
# Registered test files only: fixed roots, path segments that cannot form
# ".." or an absolute path, and a test_*.py basename. The dispatched workflow
# re-validates the same shape before the path reaches any shell command.
TEST_FILE_PATTERN = re.compile(
    r"tests/(?:e2e|fast|fast-gpu|ci)(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*/test_[A-Za-z0-9_][A-Za-z0-9_.-]*\.py"
)
RUN_FILE_COMMAND_PATTERN = re.compile(rf"/{RUN_FILE_COMMAND} +({TEST_FILE_PATTERN.pattern})")
# PR-body dependency pins, mirroring the exact patterns the pr-test.yml
# resolve steps grep for; each match is forwarded as one dispatch input.
PR_BODY_PINS = (
    (
        "ci_image_tag",
        re.compile(r"^ci-image-tag:\s+(\S+)", re.MULTILINE),
        re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"),
    ),
    (
        "ci_megatron_pr",
        re.compile(r"^ci-megatron-pr:\s+(\S+)", re.MULTILINE),
        re.compile(r"#[0-9]+|[A-Za-z0-9_][A-Za-z0-9_./-]*"),
    ),
    (
        "ci_sglang_pr",
        re.compile(r"^ci-sglang-pr:\s+(\S+)", re.MULTILINE),
        re.compile(r"#[0-9]+|[A-Za-z0-9_][A-Za-z0-9_./-]*"),
    ),
)
LABEL_PATTERN = re.compile(r"(?:run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*|bypass-fastfail)")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
POLICY_PERMISSIONS = frozenset({"write", "admin"})
POLICY_GROUPS = frozenset({"add_label_access", "repo_write_access"})


class CommentCommandError(Exception):
    pass


class AddLabel(NamedTuple):
    label: str


class ClearLabels(NamedTuple):
    pass


class RerunFailedCI(NamedTuple):
    pass


class RunTestFile(NamedTuple):
    test_file: str


class CommandSpec(NamedTuple):
    policy_key: str
    capability: str
    handler: object
    resource_key: object
    resource_loader: object
    resource_getter: object
    allows_user_ids: bool
    audit_key: str
    audit_value: object


class CommandContext(NamedTuple):
    api: object
    pull_number: int
    actor_id: int
    actor_login: str
    allowed_permissions: frozenset
    allowed_user_ids: frozenset
    current_labels: frozenset
    head_sha: str
    head_repository_id: int
    head_owner_login: str
    head_ref: str
    body: str


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CommentCommandError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise CommentCommandError(f"non-standard JSON number: {value}")


def load_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommentCommandError(f"cannot read JSON from {path}: {error}") from error


def _validate_permissions(name, values):
    if not isinstance(values, list) or not values:
        raise CommentCommandError(f"{name} must be a non-empty array")
    if any(type(value) is not str or value not in POLICY_PERMISSIONS for value in values):
        raise CommentCommandError(f"{name} must contain only write or admin")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate permissions")
    return frozenset(values)


def _validate_user_ids(name, values):
    if not isinstance(values, list):
        raise CommentCommandError(f"{name} must be an array")
    if any(type(value) is not int or value <= 0 for value in values):
        raise CommentCommandError(f"{name} must contain only positive integers")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate user IDs")
    return frozenset(values)


def _validate_labels(name, values):
    if not isinstance(values, list) or not values:
        raise CommentCommandError(f"{name} must be a non-empty array")
    for label in values:
        if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
            raise CommentCommandError(f"invalid exact CI label: {label!r}")
    if len(set(values)) != len(values):
        raise CommentCommandError(f"{name} contains duplicate labels")
    return frozenset(values)


def load_policy(path):
    raw = load_json(path)
    if not isinstance(raw, dict) or set(raw) != {"version", "groups", "commands"}:
        raise CommentCommandError("policy must contain only version, groups, and commands")
    if type(raw["version"]) is not int or raw["version"] != 2:
        raise CommentCommandError("policy version must be 2")

    raw_groups = raw["groups"]
    if not isinstance(raw_groups, dict) or set(raw_groups) != POLICY_GROUPS:
        raise CommentCommandError("policy groups must contain only add_label_access and repo_write_access")
    groups = {}
    for name, raw_group in raw_groups.items():
        expected_keys = {"repository_permissions"}
        if name == "add_label_access":
            expected_keys.add("user_ids")
        if not isinstance(raw_group, dict) or set(raw_group) != expected_keys:
            raise CommentCommandError(f"invalid fields for policy group {name}")
        groups[name] = {
            "repository_permissions": _validate_permissions(
                f"groups.{name}.repository_permissions",
                raw_group["repository_permissions"],
            ),
            "user_ids": _validate_user_ids(
                f"groups.{name}.user_ids",
                raw_group.get("user_ids", []),
            ),
        }

    specs_by_key = {spec.policy_key: spec for spec in COMMAND_REGISTRY.values()}
    if len(specs_by_key) != len(COMMAND_REGISTRY):
        raise CommentCommandError("command registry contains duplicate policy keys")
    raw_commands = raw["commands"]
    if not isinstance(raw_commands, dict) or set(raw_commands) != set(specs_by_key):
        raise CommentCommandError("policy commands do not match the static command registry")
    commands = {}
    for name, spec in specs_by_key.items():
        raw_command = raw_commands[name]
        expected_keys = {"group"}
        if spec.resource_key is not None:
            expected_keys.add(spec.resource_key)
        if not isinstance(raw_command, dict) or set(raw_command) != expected_keys:
            raise CommentCommandError(f"invalid fields for policy command {name}")
        group_name = raw_command["group"]
        if not isinstance(group_name, str) or group_name not in groups:
            raise CommentCommandError(f"policy command {name} references an unknown group")
        if not spec.allows_user_ids and groups[group_name]["user_ids"]:
            raise CommentCommandError(f"policy command {name} cannot grant access by user ID")
        command = {"group": group_name}
        if spec.resource_key is not None:
            command[spec.resource_key] = spec.resource_loader(
                f"commands.{name}.{spec.resource_key}",
                raw_command[spec.resource_key],
            )
        commands[name] = command

    return {"groups": groups, "commands": commands}


def parse_command(body):
    if not isinstance(body, str):
        raise CommentCommandError("comment body must be a string")
    command = body.strip()
    if command == f"/{CLEAR_COMMAND}":
        return ClearLabels()
    if command == f"/{RERUN_COMMAND}":
        return RerunFailedCI()
    match = COMMAND_PATTERN.fullmatch(command)
    if match is not None:
        return AddLabel(match.group(1))
    match = RUN_FILE_COMMAND_PATTERN.fullmatch(command)
    if match is not None:
        return RunTestFile(match.group(1))
    if any(marker in command for marker in COMMAND_MARKERS):
        raise CommentCommandError(
            "comment must contain one exact /<label>, /rerun-test <test-file>, /clear-labels, "
            "or /rerun-failed-ci command"
        )
    return None


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise CommentCommandError(f"{name} must be a positive integer")
    return value


def parse_event(event):
    if not isinstance(event, dict) or event.get("action") != "created":
        raise CommentCommandError("event must be issue_comment.created")

    repository = event.get("repository")
    if not isinstance(repository, dict):
        raise CommentCommandError("event repository is missing")
    if repository.get("id") != REPOSITORY_ID or repository.get("full_name") != REPOSITORY:
        raise CommentCommandError("event repository does not match radixark/miles")

    issue = event.get("issue")
    if not isinstance(issue, dict) or not isinstance(issue.get("pull_request"), dict):
        raise CommentCommandError("comment is not attached to a pull request")
    pull_number = _positive_int(issue.get("number"), "pull request number")

    comment = event.get("comment")
    if not isinstance(comment, dict) or not isinstance(comment.get("user"), dict):
        raise CommentCommandError("comment author is missing")
    if comment["user"].get("type") != "User":
        raise CommentCommandError("comment author must be a human GitHub user")
    actor_id = _positive_int(comment["user"].get("id"), "comment author ID")
    actor_login = comment["user"].get("login")
    if not isinstance(actor_login, str) or not actor_login:
        raise CommentCommandError("comment author login is missing")

    sender = event.get("sender")
    if not isinstance(sender, dict) or sender.get("id") != actor_id:
        raise CommentCommandError("event sender does not match the comment author")

    return pull_number, actor_id, actor_login, parse_command(comment.get("body"))


class GitHubAPI:
    def __init__(self, token):
        if not token:
            raise CommentCommandError("GitHub API token is missing")
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
                    raise CommentCommandError(
                        f"GitHub API returned HTTP {response.status}; expected {expected_status}"
                    )
                body = response.read()
                if not expect_json:
                    if body.strip():
                        raise CommentCommandError("GitHub API returned an unexpected response body")
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise CommentCommandError(f"GitHub API returned HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise CommentCommandError(f"GitHub API request failed: {error.reason}") from error
        except TimeoutError as error:
            raise CommentCommandError("GitHub API request timed out") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CommentCommandError("GitHub API returned invalid JSON") from error

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
                raise CommentCommandError("GitHub API returned an invalid workflow-run page")
            page_total = result.get("total_count")
            page_runs = result.get("workflow_runs")
            if type(page_total) is not int or page_total < 0 or page_total > 1000:
                raise CommentCommandError("GitHub API returned an invalid workflow-run count")
            if not isinstance(page_runs, list):
                raise CommentCommandError("GitHub API returned invalid workflow runs")
            if total_count is None:
                total_count = page_total
            elif page_total != total_count:
                raise CommentCommandError("workflow-run count changed during pagination")
            runs.extend(page_runs)
            if len(runs) >= total_count:
                if len(runs) != total_count:
                    raise CommentCommandError("GitHub API returned too many workflow runs")
                return runs
            if not page_runs:
                raise CommentCommandError("GitHub API returned an incomplete workflow-run listing")
        raise CommentCommandError("workflow-run listing exceeded the 1000-run API limit")

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
                raise CommentCommandError("GitHub API returned an invalid head pull-request page")
            pulls.extend(result)
            if len(result) < 100:
                return pulls
        raise CommentCommandError("head pull-request listing exceeded the 1000-result limit")

    def rerun_failed_jobs(self, run_id):
        self._request(
            f"/repos/{REPOSITORY}/actions/runs/{run_id}/rerun-failed-jobs",
            method="POST",
            expected_status=201,
            expect_json=False,
        )

    def create_workflow_dispatch(self, workflow_file, ref, inputs):
        encoded_workflow = urllib.parse.quote(workflow_file, safe="")
        self._request(
            f"/repos/{REPOSITORY}/actions/workflows/{encoded_workflow}/dispatches",
            method="POST",
            payload={"ref": ref, "inputs": inputs},
            expected_status=204,
            expect_json=False,
        )


def _validate_live_pull(pull, pull_number):
    if not isinstance(pull, dict) or pull.get("number") != pull_number:
        raise CommentCommandError("GitHub API returned the wrong pull request")
    if pull.get("state") != "open":
        raise CommentCommandError("pull request is not open")

    base = pull.get("base")
    base_repository = base.get("repo") if isinstance(base, dict) else None
    if not isinstance(base_repository, dict):
        raise CommentCommandError("pull request base repository is missing")
    if base_repository.get("id") != REPOSITORY_ID or base_repository.get("full_name") != REPOSITORY:
        raise CommentCommandError("pull request base repository does not match radixark/miles")

    head = pull.get("head")
    head_repository = head.get("repo") if isinstance(head, dict) else None
    if not isinstance(head_repository, dict):
        raise CommentCommandError("pull request head repository is missing")
    head_repository_id = _positive_int(head_repository.get("id"), "pull request head repository ID")
    head_repository_owner = head_repository.get("owner")
    head_owner_login = head_repository_owner.get("login") if isinstance(head_repository_owner, dict) else None
    if not isinstance(head_owner_login, str) or not head_owner_login:
        raise CommentCommandError("pull request head repository owner is missing")
    head_ref = head.get("ref") if isinstance(head, dict) else None
    if not isinstance(head_ref, str) or not head_ref:
        raise CommentCommandError("pull request head ref is missing")

    labels = pull.get("labels")
    if not isinstance(labels, list) or any(not isinstance(item, dict) for item in labels):
        raise CommentCommandError("pull request labels are invalid")
    names = []
    for item in labels:
        name = item.get("name")
        if not isinstance(name, str):
            raise CommentCommandError("pull request label name is invalid")
        names.append(name)

    head_sha = head.get("sha") if isinstance(head, dict) else None
    if not isinstance(head_sha, str) or SHA_PATTERN.fullmatch(head_sha) is None:
        raise CommentCommandError("pull request head SHA is invalid")

    return frozenset(names), head_sha, head_repository_id, head_owner_login, head_ref


def _pull_body(pull):
    body = pull.get("body")
    if body is None:
        return ""
    if not isinstance(body, str):
        raise CommentCommandError("pull request body is invalid")
    return body


def _command_spec(request):
    if request is None:
        raise CommentCommandError("comment does not contain a recognized command")
    spec = COMMAND_REGISTRY.get(type(request))
    if spec is None:
        raise CommentCommandError("request type is not registered")
    return spec


def resolve_policy(event, policy):
    pull_number, actor_id, actor_login, request = parse_event(event)
    spec = _command_spec(request)
    try:
        command_policy = policy["commands"][spec.policy_key]
        group = policy["groups"][command_policy["group"]]
        allowed_permissions = group["repository_permissions"]
        allowed_user_ids = group["user_ids"] if spec.allows_user_ids else frozenset()
    except (KeyError, TypeError) as error:
        raise CommentCommandError("resolved command policy is invalid") from error
    if spec.resource_key is not None:
        resource = spec.resource_getter(request)
        if resource not in command_policy[spec.resource_key]:
            raise CommentCommandError("requested label is not exposed by policy")
    return (
        pull_number,
        actor_id,
        actor_login,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
    )


def require_permission(api, actor_id, actor_login, allowed_permissions):
    permission_result = api.get_permission(actor_login)
    if not isinstance(permission_result, dict):
        raise CommentCommandError("GitHub API returned an invalid repository permission")
    permission_user = permission_result.get("user")
    permission_user_id = permission_user.get("id") if isinstance(permission_user, dict) else None
    if type(permission_user_id) is not int or permission_user_id <= 0:
        raise CommentCommandError("GitHub API returned an invalid repository permission identity")
    if permission_user_id != actor_id:
        raise CommentCommandError("repository permission identity does not match the comment author")
    permission = permission_result.get("permission")
    if not isinstance(permission, str):
        raise CommentCommandError("GitHub API returned an invalid repository permission")
    if permission not in allowed_permissions:
        raise CommentCommandError("comment author is not authorized for the requested operation")


def require_access(api, actor_id, actor_login, allowed_permissions, allowed_user_ids):
    if actor_id in allowed_user_ids:
        return
    require_permission(api, actor_id, actor_login, allowed_permissions)


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
            raise CommentCommandError("GitHub API returned an invalid workflow run")
        run_id = _positive_int(run.get("id"), "workflow run ID")
        if run_id in seen_ids:
            raise CommentCommandError("GitHub API returned a duplicate workflow run")
        seen_ids.add(run_id)
        run_number = _positive_int(run.get("run_number"), "workflow run number")
        if run.get("path") != workflow_path:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected path")
        if run.get("event") != "pull_request" or run.get("head_sha") != head_sha:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected event or SHA")
        if run.get("head_branch") != head_ref:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected head ref")
        run_head_repository = run.get("head_repository")
        if not isinstance(run_head_repository, dict) or run_head_repository.get("id") != head_repository_id:
            raise CommentCommandError("GitHub API returned a workflow run from an unexpected repository")
        pull_requests = run.get("pull_requests")
        if not isinstance(pull_requests, list):
            raise CommentCommandError("GitHub API returned invalid workflow-run pull requests")
        pull_numbers = set()
        for pull_request in pull_requests:
            if not isinstance(pull_request, dict):
                raise CommentCommandError("GitHub API returned an invalid workflow-run pull request")
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
        raise CommentCommandError("fork head does not identify exactly one pull request")
    pull = pulls[0]
    if not isinstance(pull, dict) or pull.get("number") != pull_number or pull.get("state") != "open":
        raise CommentCommandError("fork head identifies an unexpected pull request")
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
        raise CommentCommandError("fork head pull-request identity does not match the live pull request")


def _handle_rerun_failed_ci(context, request):
    if type(request) is not RerunFailedCI:
        raise CommentCommandError("rerun handler received the wrong request type")
    (
        api,
        pull_number,
        actor_id,
        actor_login,
        allowed_permissions,
        allowed_user_ids,
        _,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
        _,
    ) = context
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
        require_access(api, actor_id, actor_login, allowed_permissions, allowed_user_ids)
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
            raise CommentCommandError("pull request head changed before rerun")
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
            raise CommentCommandError("latest workflow-run state changed before rerun")
        require_access(api, actor_id, actor_login, allowed_permissions, allowed_user_ids)
        try:
            api.rerun_failed_jobs(run_id)
        except CommentCommandError as error:
            raise CommentCommandError(f"could not rerun failed jobs for workflow run {run_id}: {error}") from error

    return {
        "actor_id": actor_id,
        "decision": "ALLOW_RERUN_REQUESTED" if candidates else "ALLOW_NO_FAILED_RUNS",
        "head_sha": head_sha,
        "pull_number": pull_number,
        "workflow_run_ids": [candidate[0] for candidate in candidates],
    }


def _handle_add_label(context, request):
    if type(request) is not AddLabel:
        raise CommentCommandError("add-label handler received the wrong request type")
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.allowed_permissions,
        context.allowed_user_ids,
    )
    label = request.label
    if label in context.current_labels:
        decision = "ALLOW_ALREADY_PRESENT"
    else:
        result = context.api.add_label(context.pull_number, label)
        if not isinstance(result, list) or label not in {
            item.get("name") for item in result if isinstance(item, dict)
        }:
            raise CommentCommandError("GitHub API did not confirm the requested label")
        decision = "ALLOW_ADDED"
    return {
        "actor_id": context.actor_id,
        "decision": decision,
        "label": label,
        "pull_number": context.pull_number,
    }


def _handle_clear_labels(context, request):
    if type(request) is not ClearLabels:
        raise CommentCommandError("clear-labels handler received the wrong request type")
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.allowed_permissions,
        context.allowed_user_ids,
    )
    labels_to_remove = sorted(label for label in context.current_labels if _is_ci_control_label(label))
    remaining_labels = context.current_labels
    for label in labels_to_remove:
        try:
            result = context.api.remove_label(context.pull_number, label)
        except CommentCommandError as error:
            raise CommentCommandError(f"could not remove CI label {label}: {error}") from error
        if (
            not isinstance(result, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in result)
            or label in {item["name"] for item in result}
        ):
            raise CommentCommandError(f"GitHub API did not confirm removal of {label}")
        remaining_labels = frozenset(item["name"] for item in result)
    if any(_is_ci_control_label(label) for label in remaining_labels):
        raise CommentCommandError("GitHub API did not confirm that all CI labels were removed")
    return {
        "actor_id": context.actor_id,
        "decision": "ALLOW_CLEARED" if labels_to_remove else "ALLOW_ALREADY_CLEAR",
        "labels": labels_to_remove,
        "pull_number": context.pull_number,
    }


def _pr_body_pins(body):
    pins = {}
    for input_name, body_pattern, value_pattern in PR_BODY_PINS:
        match = body_pattern.search(body)
        if match is None:
            continue
        value = match.group(1)
        if value_pattern.fullmatch(value) is None:
            raise CommentCommandError(
                f"PR body pin {input_name.replace('_', '-')} has an unsupported value: {value!r}"
            )
        pins[input_name] = value
    return pins


def _handle_run_test_file(context, request):
    if type(request) is not RunTestFile:
        raise CommentCommandError("rerun-test handler received the wrong request type")
    # Only same-repository heads are accepted; the fixed default-branch
    # workflow receives the exact PR SHA as data and checks it out separately.
    if context.head_repository_id != REPOSITORY_ID:
        raise CommentCommandError("/rerun-test supports only same-repository pull requests")
    require_access(
        context.api,
        context.actor_id,
        context.actor_login,
        context.allowed_permissions,
        context.allowed_user_ids,
    )
    inputs = {
        "pull_number": str(context.pull_number),
        "head_sha": context.head_sha,
        "test_file": request.test_file,
    }
    inputs.update(_pr_body_pins(context.body))
    context.api.create_workflow_dispatch(RUN_FILE_WORKFLOW, RUN_FILE_WORKFLOW_REF, inputs)
    return {
        "actor_id": context.actor_id,
        "decision": "ALLOW_FILE_RUN_DISPATCHED",
        "head_sha": context.head_sha,
        "pull_number": context.pull_number,
        "test_file": request.test_file,
    }


def _add_label_resource(request):
    if type(request) is not AddLabel:
        raise CommentCommandError("add-label resource received the wrong request type")
    return request.label


def _clear_command_value(request):
    if type(request) is not ClearLabels:
        raise CommentCommandError("clear-labels audit received the wrong request type")
    return f"/{CLEAR_COMMAND}"


def _rerun_command_value(request):
    if type(request) is not RerunFailedCI:
        raise CommentCommandError("rerun audit received the wrong request type")
    return f"/{RERUN_COMMAND}"


def _run_file_value(request):
    if type(request) is not RunTestFile:
        raise CommentCommandError("rerun-test audit received the wrong request type")
    return request.test_file


COMMAND_REGISTRY = {
    AddLabel: CommandSpec(
        "add_label",
        "issues",
        _handle_add_label,
        "allowed_labels",
        _validate_labels,
        _add_label_resource,
        True,
        "label",
        _add_label_resource,
    ),
    ClearLabels: CommandSpec(
        "clear_labels",
        "issues",
        _handle_clear_labels,
        None,
        None,
        None,
        False,
        "command",
        _clear_command_value,
    ),
    RerunFailedCI: CommandSpec(
        "rerun_failed_ci",
        "actions",
        _handle_rerun_failed_ci,
        None,
        None,
        None,
        False,
        "command",
        _rerun_command_value,
    ),
    RunTestFile: CommandSpec(
        "run_test_file",
        "actions",
        _handle_run_test_file,
        None,
        None,
        None,
        False,
        "test_file",
        _run_file_value,
    ),
}


def process_event(event, policy, api):
    (
        pull_number,
        actor_id,
        actor_login,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
    ) = resolve_policy(event, policy)
    pull = api.get_pull(pull_number)
    (
        current_labels,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
    ) = _validate_live_pull(pull, pull_number)
    context = CommandContext(
        api,
        pull_number,
        actor_id,
        actor_login,
        allowed_permissions,
        allowed_user_ids,
        current_labels,
        head_sha,
        head_repository_id,
        head_owner_login,
        head_ref,
        _pull_body(pull),
    )
    return spec.handler(context, request)


def authorize_policy(event, policy, api):
    (
        pull_number,
        actor_id,
        actor_login,
        request,
        spec,
        allowed_permissions,
        allowed_user_ids,
    ) = resolve_policy(event, policy)
    require_access(api, actor_id, actor_login, allowed_permissions, allowed_user_ids)
    return pull_number, actor_id, request, spec


def _write_capability(capability):
    if capability not in {"none", "issues", "actions"}:
        raise CommentCommandError("command registry selected an unknown capability")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"capability={capability}\n")
    except OSError as error:
        raise CommentCommandError(f"cannot write GITHUB_OUTPUT: {error}") from error


def main():
    try:
        event = load_json(os.environ["GITHUB_EVENT_PATH"])
        if os.environ.get("CI_COMMAND_PREFLIGHT") == "true":
            pull_number, actor_id, _, request = parse_event(event)
            if request is None:
                _write_capability("none")
                print(
                    json.dumps(
                        {
                            "actor_id": actor_id,
                            "decision": "IGNORE_UNRECOGNIZED_COMMAND",
                            "pull_number": pull_number,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                return 0

        policy = load_policy(os.environ["CI_COMMAND_POLICY_PATH"])
        api = GitHubAPI(os.environ["CI_COMMAND_API_TOKEN"])
        if os.environ.get("CI_COMMAND_PREFLIGHT") == "true":
            pull_number, actor_id, request, spec = authorize_policy(event, policy, api)
            authorization = {"actor_id": actor_id, "pull_number": pull_number}
            authorization[spec.audit_key] = spec.audit_value(request)
            _write_capability(spec.capability)
            print(
                json.dumps(
                    authorization,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        result = process_event(event, policy, api)
    except CommentCommandError as error:
        print(f"::error::{error}")
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
