import datetime
import importlib.util
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.labels import KNOWN_LABELS

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
HANDLER_PATH = ROOT / ".github/workflows/scripts/comment_ci_command.py"
POLICY_PATH = ROOT / ".github/workflows/policies/comment-command-access.json"
WORKFLOW_PATH = ROOT / ".github/workflows/comment-ci-command.yml"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDLER = load_module("comment_ci_command", HANDLER_PATH)
ACTOR_ID = 1234
HEAD_SHA = "a" * 40
HEAD_REF = "feature/test"
WRITE_PERMISSIONS = frozenset({"write", "admin"})
RUN_FILE_BODY = "/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py"
RUN_FILE_PATH = "tests/e2e/precision/test_hf_attention_cp_relayout.py"
WORKFLOW_RUN_ID = 987654321
WORKFLOW_RUN_API_URL = f"https://api.github.com/repos/radixark/miles/actions/runs/{WORKFLOW_RUN_ID}"
WORKFLOW_RUN_URL = f"https://github.com/radixark/miles/actions/runs/{WORKFLOW_RUN_ID}"
RUN_MARKER = f"<!-- rerun-test-run:{WORKFLOW_RUN_ID} -->"
RUN_STARTED_AT = "2026-08-23T09:25:25Z"
# 3m11s after RUN_STARTED_AT, the shape of a real 4-GPU file run.
REPORT_NOW = datetime.datetime(2026, 8, 23, 9, 28, 36, tzinfo=datetime.timezone.utc)
NEW_COMMENT_ID = 4242
EXISTING_COMMENT_ID = 777
ANNOUNCE_BODY = (
    f"⏳ `{RUN_FILE_PATH}` is **running** — [workflow run]({WORKFLOW_RUN_URL})\n\n"
    "Started at 2026-08-23 09:28:36 UTC; elapsed time and the result will be recorded "
    f"here when it finishes.\n\n{RUN_MARKER}"
)


def file_run_env(mode, **overrides):
    environ = {
        "CI_COMMAND_FILE_RUN_STATUS": mode,
        "FILE_RUN_PULL_NUMBER": "123",
        "FILE_RUN_TEST_FILE": RUN_FILE_PATH,
        "FILE_RUN_RUN_ID": str(WORKFLOW_RUN_ID),
    }
    if mode == "report":
        environ.update(
            {
                "FILE_RUN_COMMENT_ID": str(EXISTING_COMMENT_ID),
                "FILE_RUN_SUITE": "stage-c-4-gpu-h200",
                "FILE_RUN_RESOLVE_RESULT": "success",
                "FILE_RUN_CUDA_RESULT": "success",
                "FILE_RUN_CPU_RESULT": "skipped",
            }
        )
    environ.update(overrides)
    return environ


def file_run_status(mode="report", **overrides):
    return HANDLER._file_run_status_inputs(file_run_env(mode, **overrides))[1]


class FakeAPI:
    def __init__(self, pull, *, permission="write", permission_actor_id=ACTOR_ID):
        self.pull = pull
        self.permission = {
            "permission": permission,
            "user": {"id": permission_actor_id},
        }
        self.calls = []
        self.get_calls = []
        self.permission_calls = []
        self.add_calls = []
        self.remove_calls = []
        self.workflow_runs = {workflow_file: [] for workflow_file, _ in HANDLER.RERUN_WORKFLOWS}
        self.list_run_calls = []
        self.rerun_calls = []
        self.list_pull_calls = []
        self.dispatch_calls = []
        self.reaction_calls = []
        self.comment_calls = []
        self.update_calls = []
        self.run_calls = []
        self.workflow_run = {"run_started_at": RUN_STARTED_AT}
        self.head_pulls = [pull]

    def get_pull(self, pull_number):
        self.calls.append(("get_pull", pull_number))
        self.get_calls.append(pull_number)
        return self.pull

    def get_permission(self, actor_login):
        self.calls.append(("get_permission", actor_login))
        self.permission_calls.append(actor_login)
        return self.permission

    def add_label(self, pull_number, label):
        self.calls.append(("add_label", pull_number, label))
        self.add_calls.append((pull_number, label))
        return [*self.pull["labels"], {"name": label}]

    def remove_label(self, pull_number, label):
        self.calls.append(("remove_label", pull_number, label))
        self.remove_calls.append((pull_number, label))
        self.pull["labels"] = [item for item in self.pull["labels"] if item["name"] != label]
        return self.pull["labels"]

    def list_workflow_runs(self, workflow_file, head_sha):
        self.calls.append(("list_workflow_runs", workflow_file, head_sha))
        self.list_run_calls.append((workflow_file, head_sha))
        return self.workflow_runs[workflow_file]

    def rerun_failed_jobs(self, run_id):
        self.calls.append(("rerun_failed_jobs", run_id))
        self.rerun_calls.append(run_id)

    def list_pulls_for_head(self, owner_login, head_ref):
        self.calls.append(("list_pulls_for_head", owner_login, head_ref))
        self.list_pull_calls.append((owner_login, head_ref))
        return self.head_pulls

    def create_workflow_dispatch(self, workflow_file, ref, inputs):
        self.calls.append(("create_workflow_dispatch", workflow_file, ref, inputs))
        self.dispatch_calls.append((workflow_file, ref, inputs))
        return WORKFLOW_RUN_URL

    def add_comment_reaction(self, comment_id, content):
        self.calls.append(("add_comment_reaction", comment_id, content))
        self.reaction_calls.append((comment_id, content))

    def create_issue_comment(self, pull_number, body):
        self.calls.append(("create_issue_comment", pull_number, body))
        self.comment_calls.append((pull_number, body))
        return NEW_COMMENT_ID

    def update_issue_comment(self, comment_id, body):
        self.calls.append(("update_issue_comment", comment_id, body))
        self.update_calls.append((comment_id, body))

    def get_workflow_run(self, run_id):
        self.calls.append(("get_workflow_run", run_id))
        self.run_calls.append(run_id)
        return self.workflow_run


def event(*, body="/run-ci-short", actor_id=ACTOR_ID, author_association="NONE"):
    return {
        "action": "created",
        "repository": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY},
        "issue": {"number": 123, "pull_request": {"url": "https://example.invalid/pulls/123"}},
        "comment": {
            "id": 5678,
            "body": body,
            "author_association": author_association,
            "user": {"id": actor_id, "login": "actor", "type": "User"},
        },
        "sender": {"id": actor_id, "login": "actor", "type": "User"},
    }


def pull(
    *,
    head_repository_id=HANDLER.REPOSITORY_ID,
    head_sha=HEAD_SHA,
    head_ref=HEAD_REF,
    state="open",
    labels=(),
):
    head_owner = "radixark" if head_repository_id == HANDLER.REPOSITORY_ID else "fork-owner"
    return {
        "number": 123,
        "state": state,
        "base": {"repo": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY}},
        "head": {
            "ref": head_ref,
            "sha": head_sha,
            "repo": {"id": head_repository_id, "owner": {"login": head_owner}},
        },
        "labels": [{"name": label} for label in labels],
    }


def policy(
    *,
    permissions=("write", "admin"),
    user_ids=(),
    prior_user_ids=(),
    labels=("run-ci-short", "bypass-fastfail"),
    repo_permissions=("write", "admin"),
    author_associations=("OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"),
):
    return {
        "groups": {
            "add_label_access": {
                "repository_permissions": frozenset(permissions),
                "user_ids": frozenset(user_ids),
                "author_associations": frozenset(),
            },
            "repo_write_access": {
                "repository_permissions": frozenset(repo_permissions),
                "user_ids": frozenset(),
                "author_associations": frozenset(),
            },
            "prior_contributor_access": {
                "repository_permissions": frozenset(repo_permissions),
                "user_ids": frozenset(prior_user_ids),
                "author_associations": frozenset(author_associations),
            },
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": frozenset(labels),
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "prior_contributor_access"},
            "run_test_file": {"group": "prior_contributor_access"},
        },
    }


def workflow_run(
    workflow_path,
    *,
    run_id=10,
    run_number=10,
    status="completed",
    conclusion="failure",
    head_sha=HEAD_SHA,
    head_repository_id=HANDLER.REPOSITORY_ID,
    head_ref=HEAD_REF,
    pull_number=123,
    pull_requests=None,
):
    return {
        "conclusion": conclusion,
        "event": "pull_request",
        "head_repository": {"id": head_repository_id},
        "head_branch": head_ref,
        "head_sha": head_sha,
        "id": run_id,
        "path": workflow_path,
        "pull_requests": ([{"number": pull_number}] if pull_requests is None else pull_requests),
        "run_number": run_number,
        "status": status,
    }


@pytest.mark.parametrize(
    "body",
    [
        "/run-ci-short extra",
        "/run-ci-short\n/run-ci-long",
        "please /run-ci-short",
        "/run-ci-",
        "/run-ci-/unsafe",
        "/run-ci",
        "/run-ci ",
        "/run-ci tests/e2e/test_a.py",
        "/rerun-test",
        "/rerun-test ",
        "/rerun-test tests/e2e/test_a.py extra",
        "/rerun-test tests/e2e/test_a.py\n/run-ci-short",
        "please /rerun-test tests/e2e/test_a.py",
        "/rerun-test tests/unit/test_a.py",
        "/rerun-test tests/e2e/helper.py",
        "/rerun-test /tests/e2e/test_a.py",
        "/rerun-test ../tests/e2e/test_a.py",
        "/rerun-test tests/e2e/../fast/test_a.py",
        "/rerun-test tests/e2e/test_a.py;rm",
        "/rerun-test tests/e2e/test_a.py tests/e2e/test_b.py",
        "/rerun-failed-ci extra",
        "/rerun-failed-ci\n/run-ci-short",
        "/clear-labels extra",
        "/clear-labels\n/run-ci-short",
        "please /clear-labels",
    ],
)
def test_command_parser_rejects_non_exact_commands(body):
    with pytest.raises(HANDLER.CommentCommandError, match="one exact /<label>"):
        HANDLER.parse_command(body)


def test_command_parser_accepts_one_exact_command_with_outer_whitespace():
    assert HANDLER.parse_command(" \n/run-ci-a_B.c-d\t") == HANDLER.AddLabel("run-ci-a_B.c-d")


def test_command_parser_accepts_exact_bypass_fastfail_command():
    assert HANDLER.parse_command("/bypass-fastfail") == HANDLER.AddLabel("bypass-fastfail")


def test_command_parser_accepts_exact_clear_command_with_outer_whitespace():
    assert HANDLER.parse_command(" \n/clear-labels\t") == HANDLER.ClearLabels()


def test_command_parser_accepts_exact_rerun_command_with_outer_whitespace():
    assert HANDLER.parse_command(" \n/rerun-failed-ci\t") == HANDLER.RerunFailedCI()


@pytest.mark.parametrize(
    ("body", "test_file"),
    [
        (
            "/rerun-test tests/e2e/precision/test_hf_attention_cp_relayout.py",
            "tests/e2e/precision/test_hf_attention_cp_relayout.py",
        ),
        (" \n/rerun-test tests/e2e/test_a.py\t", "tests/e2e/test_a.py"),
        ("/rerun-test   tests/fast/rollout/test_b.py", "tests/fast/rollout/test_b.py"),
        ("/rerun-test tests/fast-gpu/test_c.py", "tests/fast-gpu/test_c.py"),
        ("/rerun-test tests/ci/test/test_d.py", "tests/ci/test/test_d.py"),
    ],
)
def test_command_parser_accepts_run_file_command(body, test_file):
    assert HANDLER.parse_command(body) == HANDLER.RunTestFile(test_file)


@pytest.mark.parametrize("body", ["", "ordinary review comment", "/Clear-labels", "/unknown-command"])
def test_command_parser_ignores_unrelated_comments(body):
    assert HANDLER.parse_command(body) is None


def test_static_registry_owns_policy_capability_and_handler_routing():
    expected = {
        HANDLER.AddLabel: ("add_label", "issues", HANDLER._handle_add_label, "none"),
        HANDLER.ClearLabels: ("clear_labels", "issues", HANDLER._handle_clear_labels, "none"),
        HANDLER.RerunFailedCI: ("rerun_failed_ci", "actions", HANDLER._handle_rerun_failed_ci, "none"),
        HANDLER.RunTestFile: ("run_test_file", "actions", HANDLER._handle_run_test_file, "+1"),
    }
    assert set(HANDLER.COMMAND_REGISTRY) == set(expected)
    for request_type, (policy_key, capability, handler, success_reaction) in expected.items():
        spec = HANDLER.COMMAND_REGISTRY[request_type]
        assert (spec.policy_key, spec.capability, spec.handler, spec.success_reaction) == (
            policy_key,
            capability,
            handler,
            success_reaction,
        )


def test_unknown_request_type_fails_closed():
    class UnknownRequest:
        pass

    with pytest.raises(HANDLER.CommentCommandError, match="not registered"):
        HANDLER._command_spec(UnknownRequest())


def raw_policy():
    return {
        "version": 4,
        "groups": {
            "add_label_access": {
                "repository_permissions": ["write", "admin"],
                "users": [],
            },
            "repo_write_access": {"repository_permissions": ["write", "admin"]},
            "prior_contributor_access": {
                "repository_permissions": ["write", "admin"],
                "users": [],
                "author_associations": ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"],
            },
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": ["run-ci-short", "bypass-fastfail"],
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "prior_contributor_access"},
            "run_test_file": {"group": "prior_contributor_access"},
        },
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"version":1,"version":1,"labels":[]}', "duplicate JSON key"),
        ('{"version":NaN,"labels":[]}', "non-standard JSON number"),
        (
            '{"version":1,"add_label_access":{"repository_permissions":[true],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["read"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write","write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "repository_permissions contains duplicate permissions",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[true]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "user_ids must contain only positive integers",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[123,123]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "user_ids contains duplicate user IDs",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],"labels":["unsafe label"]}',
            "invalid exact CI label",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short","run-ci-short"]}',
            "labels contains duplicate labels",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":[true],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "clear_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["read"],"rerun_permissions":["write"],"labels":["run-ci-short"]}',
            "clear_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write","write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short"]}',
            "clear_permissions contains duplicate permissions",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["read"],"labels":["run-ci-short"]}',
            "rerun_permissions must contain only write or admin",
        ),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write","write"],'
            '"labels":["run-ci-short"]}',
            "rerun_permissions contains duplicate permissions",
        ),
        ('{"version":1,"labels":["run-ci-short"]}', "only version"),
        (
            '{"version":1,"add_label_access":{"repository_permissions":["write"],"user_ids":[]},'
            '"clear_permissions":["write"],"rerun_permissions":["write"],'
            '"labels":["run-ci-short"],'
            '"roles":{}}',
            "only version",
        ),
    ],
)
def test_policy_parser_rejects_legacy_or_nonstandard_schema(tmp_path, text, message):
    path = tmp_path / "policy.json"
    path.write_text(text)
    expected = message if "duplicate JSON" in message or "non-standard JSON" in message else "only version, groups"
    with pytest.raises(HANDLER.CommentCommandError, match=expected):
        HANDLER.load_policy(path)


@pytest.mark.parametrize(
    ("path_parts", "value", "message"),
    [
        (("version",), 3, "version must be 4"),
        (("groups", "add_label_access", "repository_permissions"), [True], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["read"], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["write", "write"], "duplicate permissions"),
        (("groups", "prior_contributor_access", "author_associations"), [], "non-empty array"),
        (("groups", "prior_contributor_access", "author_associations"), [True], "only GitHub author associations"),
        (
            ("groups", "prior_contributor_access", "author_associations"),
            ["read"],
            "only GitHub author associations",
        ),
        (
            ("groups", "prior_contributor_access", "author_associations"),
            ["MEMBER", "MEMBER"],
            "duplicate author associations",
        ),
        (("commands", "add_label", "allowed_labels"), ["unsafe label"], "invalid exact CI label"),
        (("commands", "add_label", "allowed_labels"), ["run-ci-short", "run-ci-short"], "duplicate labels"),
        (("commands", "add_label", "group"), "missing", "unknown group"),
        (("commands", "clear_labels", "unexpected"), True, "invalid fields"),
        (("groups", "repo_write_access", "users"), [{"id": 123, "login": "actor"}], "invalid fields"),
        (("groups", "repo_write_access", "author_associations"), ["MEMBER"], "invalid fields"),
        (("groups", "add_label_access", "author_associations"), ["MEMBER"], "invalid fields"),
    ],
)
def test_policy_parser_rejects_invalid_group_command_or_resource(tmp_path, path_parts, value, message):
    raw = raw_policy()
    target = raw
    for part in path_parts[:-1]:
        target = target[part]
    target[path_parts[-1]] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.load_policy(path)


def test_policy_parser_loads_independent_users_for_both_customizable_tiers(tmp_path):
    raw = raw_policy()
    raw["groups"]["add_label_access"]["users"] = [{"id": 111, "login": "alice"}]
    raw["groups"]["prior_contributor_access"]["users"] = [{"id": 222, "login": "bob"}]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))

    loaded = HANDLER.load_policy(path)

    assert loaded["groups"]["add_label_access"]["user_ids"] == {111}
    assert loaded["groups"]["prior_contributor_access"]["user_ids"] == {222}


@pytest.mark.parametrize(
    ("group", "body"),
    [
        ("add_label_access", "/run-ci-short"),
        ("prior_contributor_access", RUN_FILE_BODY),
    ],
)
def test_user_login_is_display_only_for_authorization(tmp_path, group, body):
    raw = raw_policy()
    raw["groups"][group]["users"] = [{"id": ACTOR_ID, "login": "stale-login"}]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.authorize_policy(
        event(body=body, author_association="FIRST_TIMER"),
        HANDLER.load_policy(path),
        api,
    )

    assert result[0:2] == (123, ACTOR_ID)
    assert api.calls == []


@pytest.mark.parametrize("section", ["groups", "commands"])
def test_policy_parser_rejects_unknown_group_or_command(tmp_path, section):
    raw = raw_policy()
    raw[section]["unexpected"] = {}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    expected = "groups must contain only" if section == "groups" else "commands do not match"
    with pytest.raises(HANDLER.CommentCommandError, match=expected):
        HANDLER.load_policy(path)


def test_repo_write_command_cannot_use_a_group_with_explicit_user_ids(tmp_path):
    raw = raw_policy()
    raw["groups"]["add_label_access"]["users"] = [{"id": ACTOR_ID, "login": "actor"}]
    raw["commands"]["clear_labels"]["group"] = "add_label_access"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HANDLER.CommentCommandError, match="cannot grant access by user ID"):
        HANDLER.load_policy(path)


def test_checked_in_policy_exposes_exact_labels_and_access_groups():
    loaded = HANDLER.load_policy(POLICY_PATH)
    labels = loaded["commands"]["add_label"]["allowed_labels"]
    assert labels == {f"run-ci-{key}" for key in KNOWN_LABELS} | {
        "bypass-fastfail",
        "run-ci-image",
    }
    assert loaded["groups"]["add_label_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset({82826991, 59716405, 101526713, 106564213}),
        "author_associations": frozenset(),
    }
    assert loaded["groups"]["repo_write_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
        "author_associations": frozenset(),
    }
    assert loaded["groups"]["prior_contributor_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
        "author_associations": frozenset({"OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"}),
    }
    assert loaded["commands"]["clear_labels"] == {"group": "repo_write_access"}
    assert loaded["commands"]["rerun_failed_ci"] == {"group": "prior_contributor_access"}
    assert loaded["commands"]["run_test_file"] == {"group": "prior_contributor_access"}
    assert all(HANDLER.LABEL_PATTERN.fullmatch(label) for label in labels)


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_adds_one_exact_label(permission):
    api = FakeAPI(pull(), permission=permission)

    result = HANDLER.process_event(event(), policy(), api)

    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_ADDED",
        "label": "run-ci-short",
        "pull_number": 123,
    }
    assert api.get_calls == [123]
    assert api.permission_calls == ["actor"]
    assert api.add_calls == [(123, "run-ci-short")]
    assert api.calls == [
        ("get_pull", 123),
        ("get_permission", "actor"),
        ("add_label", 123, "run-ci-short"),
    ]


def test_maintain_role_is_allowed_by_legacy_write_permission():
    api = FakeAPI(pull(), permission="write")
    api.permission["role_name"] = "maintain"

    result = HANDLER.process_event(event(), policy(), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_label_without_repository_write():
    api = FakeAPI(pull(), permission="read")

    result = HANDLER.process_event(event(), policy(user_ids=(ACTOR_ID,)), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_bypass_fastfail():
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body="/bypass-fastfail"),
        policy(user_ids=(ACTOR_ID,)),
        api,
    )

    assert result["label"] == "bypass-fastfail"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "bypass-fastfail")]


def test_add_label_access_user_id_preflight_does_not_require_repository_permission():
    api = FakeAPI(pull(), permission="none")

    assert HANDLER.authorize_policy(event(), policy(user_ids=(ACTOR_ID,)), api) == (
        123,
        ACTOR_ID,
        HANDLER.AddLabel("run-ci-short"),
        HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel],
    )
    assert api.calls == []


@pytest.mark.parametrize(
    ("body", "request_type"),
    [
        (RUN_FILE_BODY, HANDLER.RunTestFile),
        ("/rerun-failed-ci", HANDLER.RerunFailedCI),
    ],
)
def test_prior_contributor_access_user_id_preflight_does_not_require_repository_permission(body, request_type):
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.authorize_policy(
        event(body=body, author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert result[0:2] == (123, ACTOR_ID)
    assert type(result[2]) is request_type
    assert api.calls == []


@pytest.mark.parametrize("body", ["/clear-labels", "/rerun-failed-ci"])
def test_add_label_access_user_id_cannot_clear_or_rerun(body):
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=body), policy(user_ids=(ACTOR_ID,)), api)

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []


@pytest.mark.parametrize(("permission", "allowed"), [("write", False), ("admin", True)])
def test_add_label_access_group_can_require_admin(permission, allowed):
    api = FakeAPI(pull(), permission=permission)

    if allowed:
        assert HANDLER.process_event(event(), policy(permissions=("admin",)), api)["decision"] == "ALLOW_ADDED"
    else:
        with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
            HANDLER.process_event(event(), policy(permissions=("admin",)), api)


def test_existing_label_is_an_authorized_no_op():
    api = FakeAPI(pull(labels=("run-ci-short", "documentation")))

    result = HANDLER.process_event(event(), policy(), api)

    assert result["decision"] == "ALLOW_ALREADY_PRESENT"
    assert api.add_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_clears_only_ci_control_labels(permission):
    api = FakeAPI(
        pull(
            labels=(
                "run-ci-short",
                "run-ci",
                "run-ci-all",
                "run-ci-historical",
                "nightly",
                "bypass-fastfail",
                "documentation",
                "bug",
            )
        ),
        permission=permission,
    )

    result = HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    removed = [
        "bypass-fastfail",
        "nightly",
        "run-ci",
        "run-ci-all",
        "run-ci-historical",
        "run-ci-short",
    ]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_CLEARED",
        "labels": removed,
        "pull_number": 123,
    }
    assert api.remove_calls == [(123, label) for label in removed]
    assert api.pull["labels"] == [{"name": "documentation"}, {"name": "bug"}]


def test_clear_is_an_authorized_no_op_when_no_ci_control_label_exists():
    api = FakeAPI(pull(labels=("documentation", "bug")))

    result = HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    assert result["decision"] == "ALLOW_ALREADY_CLEAR"
    assert result["labels"] == []
    assert api.remove_calls == []


def test_unknown_request_does_not_call_github_api():
    api = FakeAPI(pull())
    with pytest.raises(HANDLER.CommentCommandError, match="not exposed"):
        HANDLER.process_event(event(body="/run-ci-unknown"), policy(), api)
    assert api.calls == []


@pytest.mark.parametrize("permission", ["read", "none", "unknown"])
def test_caller_without_write_permission_cannot_mutate(permission):
    api = FakeAPI(pull(), permission=permission)
    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(), policy(), api)
    assert api.calls == [("get_pull", 123), ("get_permission", "actor")]
    assert api.add_calls == []


def test_preflight_checks_permission_without_reading_pull_request():
    api = FakeAPI(pull())

    assert HANDLER.authorize_policy(event(), policy(), api) == (
        123,
        ACTOR_ID,
        HANDLER.AddLabel("run-ci-short"),
        HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel],
    )
    assert api.calls == [("get_permission", "actor")]


def test_clear_preflight_uses_its_own_permission_policy_without_reading_pull_request():
    api = FakeAPI(pull(), permission="admin")

    assert HANDLER.authorize_policy(
        event(body="/clear-labels"),
        policy(repo_permissions=("admin",)),
        api,
    ) == (
        123,
        ACTOR_ID,
        HANDLER.ClearLabels(),
        HANDLER.COMMAND_REGISTRY[HANDLER.ClearLabels],
    )
    assert api.calls == [("get_permission", "actor")]


def test_clear_requires_its_exact_live_permission_policy():
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="write")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body="/clear-labels"),
            policy(repo_permissions=("admin",)),
            api,
        )

    assert api.remove_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_can_add_a_label_to_a_fork_pr(permission):
    api = FakeAPI(pull(head_repository_id=999), permission=permission, permission_actor_id=2)

    result = HANDLER.process_event(event(actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.add_calls == [(123, "run-ci-short")]


def test_add_label_access_user_id_can_add_a_label_to_a_fork_pr():
    api = FakeAPI(
        pull(head_repository_id=999),
        permission="none",
        permission_actor_id=2,
    )

    result = HANDLER.process_event(
        event(actor_id=2),
        policy(user_ids=(2,)),
        api,
    )

    assert result["decision"] == "ALLOW_ADDED"
    assert api.permission_calls == []
    assert api.add_calls == [(123, "run-ci-short")]


def test_repository_writer_can_clear_ci_labels_from_a_fork_pr():
    api = FakeAPI(
        pull(head_repository_id=999, labels=("run-ci-short", "documentation")),
        permission_actor_id=2,
    )

    result = HANDLER.process_event(event(body="/clear-labels", actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_CLEARED"
    assert api.remove_calls == [(123, "run-ci-short")]
    assert api.pull["labels"] == [{"name": "documentation"}]


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_reruns_latest_failed_pr_ci(permission):
    api = FakeAPI(pull(), permission=permission)
    expected_ids = []
    for index, (workflow_file, workflow_path) in enumerate(HANDLER.RERUN_WORKFLOWS):
        old_id = 100 + index
        latest_id = 200 + index
        api.workflow_runs[workflow_file] = [
            workflow_run(workflow_path, run_id=old_id, run_number=1),
            workflow_run(workflow_path, run_id=latest_id, run_number=2),
        ]
        expected_ids.append(latest_id)

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_RERUN_REQUESTED",
        "head_sha": HEAD_SHA,
        "pull_number": 123,
        "workflow_run_ids": expected_ids,
    }
    expected_list_calls = [(workflow_file, HEAD_SHA) for workflow_file, _ in HANDLER.RERUN_WORKFLOWS]
    assert api.list_run_calls == expected_list_calls + expected_list_calls
    assert api.rerun_calls == expected_ids
    assert api.permission_calls == ["actor"] * len(expected_ids)


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("queued", None),
        ("in_progress", None),
        ("completed", "success"),
        ("completed", "skipped"),
        ("completed", "cancelled"),
        ("completed", "timed_out"),
    ],
)
def test_newer_non_failure_run_does_not_revive_an_older_failure(status, conclusion):
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [
        workflow_run(workflow_path, run_id=10, run_number=1),
        workflow_run(
            workflow_path,
            run_id=20,
            run_number=2,
            status=status,
            conclusion=conclusion,
        ),
    ]

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result["decision"] == "ALLOW_NO_FAILED_RUNS"
    assert result["workflow_run_ids"] == []
    assert api.rerun_calls == []


def test_rerun_ignores_same_sha_run_not_associated_with_this_pr():
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, pull_number=456)]

    result = HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert result["decision"] == "ALLOW_NO_FAILED_RUNS"
    assert api.rerun_calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"path": ".github/workflows/docker-pr-tag-cleanup.yml"}, "unexpected path"),
        ({"event": "pull_request_target"}, "unexpected event or SHA"),
        ({"head_sha": "b" * 40}, "unexpected event or SHA"),
        ({"head_branch": "other/ref"}, "unexpected head ref"),
        ({"head_repository": {"id": 999}}, "unexpected repository"),
        ({"pull_requests": None}, "invalid workflow-run pull requests"),
    ],
)
def test_rerun_fails_closed_on_mismatched_run_identity(change, message):
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [{**workflow_run(workflow_path), **change}]

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_requires_its_exact_live_permission_policy():
    api = FakeAPI(pull(), permission="write")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body="/rerun-failed-ci"),
            policy(repo_permissions=("admin",)),
            api,
        )

    assert api.rerun_calls == []


def test_rerun_preflight_uses_its_own_policy_without_reading_pull_request():
    api = FakeAPI(pull(), permission="admin")

    assert HANDLER.authorize_policy(
        event(body="/rerun-failed-ci"),
        policy(repo_permissions=("admin",)),
        api,
    ) == (
        123,
        ACTOR_ID,
        HANDLER.RerunFailedCI(),
        HANDLER.COMMAND_REGISTRY[HANDLER.RerunFailedCI],
    )
    assert api.calls == [("get_permission", "actor")]


def test_repository_writer_can_rerun_failed_ci_for_a_fork_pr():
    api = FakeAPI(pull(head_repository_id=999), permission_actor_id=2)
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, head_repository_id=999, pull_requests=[])]

    result = HANDLER.process_event(event(body="/rerun-failed-ci", actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_RERUN_REQUESTED"
    assert api.rerun_calls == [10]
    assert api.list_pull_calls == [("fork-owner", HEAD_REF), ("fork-owner", HEAD_REF)]


@pytest.mark.parametrize("head_pulls", [[], [pull(head_repository_id=999)] * 2])
def test_fork_rerun_requires_one_unique_head_pull(head_pulls):
    api = FakeAPI(pull(head_repository_id=999))
    api.head_pulls = head_pulls
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path, head_repository_id=999, pull_requests=[])]

    with pytest.raises(HANDLER.CommentCommandError, match="exactly one pull request"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_fork_rerun_rejects_mismatched_unique_head_pull():
    api = FakeAPI(pull(head_repository_id=999))
    api.head_pulls = [pull(head_repository_id=999, head_sha="b" * 40)]

    with pytest.raises(HANDLER.CommentCommandError, match="identity does not match"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_if_pr_head_changes_before_post():
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]
    pulls = iter([pull(), pull(head_sha="b" * 40)])

    def get_pull(pull_number):
        api.get_calls.append(pull_number)
        return next(pulls)

    api.get_pull = get_pull

    with pytest.raises(HANDLER.CommentCommandError, match="head changed"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_if_latest_run_state_changes_before_post():
    api = FakeAPI(pull())
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    calls = 0

    def list_workflow_runs(requested_workflow, head_sha):
        nonlocal calls
        assert head_sha == HEAD_SHA
        if requested_workflow != workflow_file:
            return []
        calls += 1
        if calls == 1:
            return [workflow_run(workflow_path)]
        return [workflow_run(workflow_path, status="queued", conclusion=None)]

    api.list_workflow_runs = list_workflow_runs

    with pytest.raises(HANDLER.CommentCommandError, match="state changed"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == []


def test_rerun_stops_after_partial_failure_without_retry_or_rollback():
    api = FakeAPI(pull())
    for index, (workflow_file, workflow_path) in enumerate(HANDLER.RERUN_WORKFLOWS[:2]):
        api.workflow_runs[workflow_file] = [workflow_run(workflow_path, run_id=(index + 1) * 10)]

    def rerun_failed_jobs(run_id):
        api.rerun_calls.append(run_id)
        if run_id == 20:
            raise HANDLER.CommentCommandError("GitHub API request timed out")

    api.rerun_failed_jobs = rerun_failed_jobs

    with pytest.raises(HANDLER.CommentCommandError, match="workflow run 20"):
        HANDLER.process_event(event(body="/rerun-failed-ci"), policy(), api)

    assert api.rerun_calls == [10, 20]


@pytest.mark.parametrize(
    ("bad_event", "message"),
    [
        ({**event(), "sender": {"id": 2}}, "sender does not match"),
        (
            {
                **event(),
                "comment": {**event()["comment"], "user": {"id": ACTOR_ID, "type": "Bot"}},
            },
            "human GitHub user",
        ),
        (
            {
                **event(),
                "comment": {**event()["comment"], "user": {"id": ACTOR_ID, "type": "User"}},
            },
            "login is missing",
        ),
        ({**event(), "repository": {"id": 1, "full_name": "attacker/repo"}}, "event repository"),
    ],
)
def test_untrusted_event_identity_fails_before_api_access(bad_event, message):
    api = FakeAPI(pull())
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(bad_event, policy(), api)
    assert api.calls == []


@pytest.mark.parametrize(
    ("permission_result", "message"),
    [
        ([], "invalid repository permission"),
        ({"permission": "write"}, "invalid repository permission identity"),
        ({"permission": "write", "user": {"id": True}}, "invalid repository permission identity"),
        ({"permission": "write", "user": {"id": 2}}, "identity does not match"),
        ({"permission": None, "user": {"id": ACTOR_ID}}, "invalid repository permission"),
    ],
)
def test_invalid_permission_response_fails_before_mutation(permission_result, message):
    api = FakeAPI(pull())
    api.permission = permission_result

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(), policy(), api)

    assert api.calls == [("get_pull", 123), ("get_permission", "actor")]
    assert api.add_calls == []


@pytest.mark.parametrize(
    ("bad_pull", "message"),
    [
        (pull(state="closed"), "not open"),
        ({**pull(), "head": {"repo": None}}, "head repository is missing"),
        ({**pull(), "base": {"repo": {"id": 1, "full_name": "attacker/repo"}}}, "base repository"),
        ({**pull(), "labels": None}, "labels are invalid"),
    ],
)
def test_unverifiable_live_pull_fails_before_mutation(bad_pull, message):
    api = FakeAPI(bad_pull)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.process_event(event(), policy(), api)
    assert api.calls == [("get_pull", 123)]


def test_github_api_uses_only_fixed_repository_and_additive_endpoint(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'[{"name":"run-ci-short"}]'

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    api = HANDLER.GitHubAPI("secret-token")
    api.add_label(123, "run-ci-short")

    request, timeout = requests[0]
    assert request.full_url == "https://api.github.com/repos/radixark/miles/issues/123/labels"
    assert request.method == "POST"
    assert json.loads(request.data) == {"labels": ["run-ci-short"]}
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert timeout == 15


def test_remove_label_uses_encoded_name_in_a_fixed_repository_path(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"[]"

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").remove_label(123, "run-ci-a/b ?#%")

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/issues/123/labels/" "run-ci-a%2Fb%20%3F%23%25"
    )
    assert request.method == "DELETE"
    assert request.data is None
    assert timeout == 15


def test_list_workflow_runs_uses_fixed_filters_and_complete_pagination(monkeypatch):
    requests = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        page = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["page"][0]
        if page == "1":
            return Response({"total_count": 101, "workflow_runs": list(range(100))})
        return Response({"total_count": 101, "workflow_runs": [100]})

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    runs = HANDLER.GitHubAPI("secret-token").list_workflow_runs("pr-test.yml", HEAD_SHA)

    assert runs == list(range(101))
    assert len(requests) == 2
    for index, (request, timeout) in enumerate(requests, start=1):
        parsed = urllib.parse.urlparse(request.full_url)
        assert parsed.path == "/repos/radixark/miles/actions/workflows/pr-test.yml/runs"
        assert urllib.parse.parse_qs(parsed.query) == {
            "event": ["pull_request"],
            "head_sha": [HEAD_SHA],
            "page": [str(index)],
            "per_page": ["100"],
        }
        assert request.method == "GET"
        assert timeout == 15


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ([{"total_count": True, "workflow_runs": []}], "invalid workflow-run count"),
        ([{"total_count": 1001, "workflow_runs": []}], "invalid workflow-run count"),
        ([{"total_count": 1, "workflow_runs": []}], "incomplete workflow-run listing"),
        (
            [
                {"total_count": 101, "workflow_runs": list(range(100))},
                {"total_count": 102, "workflow_runs": [100]},
            ],
            "count changed",
        ),
    ],
)
def test_list_workflow_runs_fails_closed_on_invalid_pagination(monkeypatch, pages, message):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    page_iterator = iter(pages)
    monkeypatch.setattr(
        HANDLER.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(next(page_iterator)),
    )

    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").list_workflow_runs("pr-test.yml", HEAD_SHA)


def test_list_pulls_for_head_uses_encoded_all_state_query(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"[]"

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    assert HANDLER.GitHubAPI("secret-token").list_pulls_for_head("fork-owner", "feature/a b") == []

    request, timeout = requests[0]
    parsed = urllib.parse.urlparse(request.full_url)
    assert parsed.path == "/repos/radixark/miles/pulls"
    assert urllib.parse.parse_qs(parsed.query) == {
        "head": ["fork-owner:feature/a b"],
        "page": ["1"],
        "per_page": ["100"],
        "state": ["all"],
    }
    assert request.method == "GET"
    assert timeout == 15


def test_rerun_failed_jobs_uses_exact_endpoint_and_accepts_empty_201(monkeypatch):
    requests = []

    class Response:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b""

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").rerun_failed_jobs(987)

    request, timeout = requests[0]
    assert request.full_url == ("https://api.github.com/repos/radixark/miles/actions/runs/987/rerun-failed-jobs")
    assert request.method == "POST"
    assert request.data is None
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [(202, b"", "expected 201"), (201, b"{}", "unexpected response body")],
)
def test_rerun_failed_jobs_rejects_unconfirmed_response_without_retry(monkeypatch, status, body, message):
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").rerun_failed_jobs(987)
    assert attempts == 1


def test_permission_lookup_encodes_the_login_in_a_fixed_repository_path(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"permission":"write","user":{"id":1234}}'

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").get_permission("actor/../../attacker")

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/collaborators/actor%2F..%2F..%2Fattacker/permission"
    )
    assert request.method == "GET"
    assert timeout == 15


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (urllib.error.HTTPError("url", 404, "missing", {}, None), "HTTP 404"),
        (TimeoutError(), "timed out"),
    ],
)
def test_permission_api_failure_is_not_retried(monkeypatch, exception, message):
    attempts = []

    def urlopen(_request, *, timeout):
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert attempts == [15]


def test_permission_api_invalid_json_is_not_retried(monkeypatch):
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"not-json"

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match="invalid JSON"):
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert attempts == 1


@pytest.mark.parametrize("operation", ["add", "remove"])
@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (TimeoutError(), "timed out"),
    ],
)
def test_label_mutation_api_failure_is_not_retried(monkeypatch, operation, exception, message):
    attempts = []

    def urlopen(_request, *, timeout):
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        api = HANDLER.GitHubAPI("secret-token")
        if operation == "add":
            api.add_label(123, "run-ci-short")
        else:
            api.remove_label(123, "run-ci-short")
    assert attempts == [15]


def test_http_error_names_the_request_and_accepted_permissions(monkeypatch):
    def urlopen(_request, *, timeout):
        raise urllib.error.HTTPError(
            "url", 403, "forbidden", {"X-Accepted-GitHub-Permissions": "pull_requests=write"}, None
        )

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError) as excinfo:
        HANDLER.GitHubAPI("secret-token").add_label(123, "run-ci-short")
    message = str(excinfo.value)
    assert message == (
        "GitHub API returned HTTP 403 for POST /repos/radixark/miles/issues/123/labels; "
        "accepted permissions: pull_requests=write"
    )
    assert "secret-token" not in message


def test_http_error_without_accepted_permissions_still_names_the_request(monkeypatch):
    def urlopen(_request, *, timeout):
        raise urllib.error.HTTPError("url", 404, "missing", {}, None)

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError) as excinfo:
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert str(excinfo.value) == (
        "GitHub API returned HTTP 404 for GET /repos/radixark/miles/collaborators/actor/permission"
    )


def test_unconfirmed_mutation_response_fails_without_rollback():
    api = FakeAPI(pull())
    api.add_label = lambda _pull_number, _label: []
    with pytest.raises(HANDLER.CommentCommandError, match="did not confirm"):
        HANDLER.process_event(event(), policy(), api)


def test_unconfirmed_label_removal_fails_without_rollback():
    api = FakeAPI(pull(labels=("run-ci-short", "documentation")))
    api.remove_label = lambda _pull_number, _label: [
        {"name": "run-ci-short"},
        {"name": "documentation"},
    ]

    with pytest.raises(HANDLER.CommentCommandError, match="did not confirm removal"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)


def test_clear_stops_after_partial_failure_without_retry_or_rollback():
    api = FakeAPI(pull(labels=("run-ci-a", "run-ci-b", "documentation")))

    def remove_label(pull_number, label):
        api.calls.append(("remove_label", pull_number, label))
        api.remove_calls.append((pull_number, label))
        if label == "run-ci-b":
            raise HANDLER.CommentCommandError("GitHub API request timed out")
        api.pull["labels"] = [item for item in api.pull["labels"] if item["name"] != label]
        return api.pull["labels"]

    api.remove_label = remove_label

    with pytest.raises(HANDLER.CommentCommandError, match="could not remove CI label run-ci-b"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)

    assert api.remove_calls == [(123, "run-ci-a"), (123, "run-ci-b")]
    assert api.pull["labels"] == [{"name": "run-ci-b"}, {"name": "documentation"}]


def test_clear_rejects_a_final_response_with_a_new_ci_control_label():
    api = FakeAPI(pull(labels=("run-ci-short",)))
    api.remove_label = lambda _pull_number, _label: [{"name": "nightly"}]

    with pytest.raises(HANDLER.CommentCommandError, match="all CI labels were removed"):
        HANDLER.process_event(event(body="/clear-labels"), policy(), api)


@pytest.mark.parametrize(
    ("body", "capability", "success_reaction"),
    [
        ("/run-ci-short", "issues", "none"),
        ("/bypass-fastfail", "issues", "none"),
        ("/clear-labels", "issues", "none"),
        ("/rerun-failed-ci", "actions", "none"),
        (RUN_FILE_BODY, "actions", "+1"),
    ],
)
def test_preflight_writes_only_fixed_routing(monkeypatch, tmp_path, body, capability, success_reaction):
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=body))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    assert output_path.read_text() == (f"capability={capability}\nsuccess_reaction={success_reaction}\n")
    assert api.calls == [("get_permission", "actor")]


def test_preflight_rejects_an_unknown_registry_capability(monkeypatch, tmp_path):
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    spec = HANDLER.COMMAND_REGISTRY[HANDLER.AddLabel]
    monkeypatch.setitem(HANDLER.COMMAND_REGISTRY, HANDLER.AddLabel, spec._replace(capability="unknown"))
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event())
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 1
    assert not output_path.exists()


def test_preflight_rejects_an_unknown_registry_success_reaction(monkeypatch, tmp_path):
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    spec = HANDLER.COMMAND_REGISTRY[HANDLER.RunTestFile]
    monkeypatch.setitem(
        HANDLER.COMMAND_REGISTRY,
        HANDLER.RunTestFile,
        spec._replace(success_reaction="heart"),
    )
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "token")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 1
    assert not output_path.exists()


def test_unrelated_comment_preflight_uses_no_policy_or_api(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body="ordinary review comment"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: pytest.fail("API must not be initialized"))
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    assert output_path.read_text() == "capability=none\nsuccess_reaction=none\n"


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_dispatches_a_file_run(permission):
    api = FakeAPI(pull(), permission=permission)

    result = HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_FILE_RUN_DISPATCHED",
        "head_sha": HEAD_SHA,
        "pull_number": 123,
        "test_file": RUN_FILE_PATH,
        "workflow_run_url": WORKFLOW_RUN_URL,
    }


def test_file_run_main_reports_the_dispatched_run_without_writing_an_output(monkeypatch, tmp_path, capsys):
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: policy())
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_POLICY_PATH", "policy.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "actions-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    # The dispatched run URL stays in the audit record; the dispatched run now
    # owns the pull-request status comment, so no job output carries it.
    assert json.loads(capsys.readouterr().out)["workflow_run_url"] == WORKFLOW_RUN_URL
    assert not output_path.exists()


def test_file_run_forwards_validated_pr_body_pins():
    target = pull()
    target["body"] = "Summary line\nci-image-tag: pr-42\nci-megatron-pr: #77\nci-sglang-pr: feature/pin-x\n"
    api = FakeAPI(target)

    HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    (_, _, inputs) = api.dispatch_calls[0]
    assert inputs == {
        "pull_number": "123",
        "head_sha": HEAD_SHA,
        "test_file": RUN_FILE_PATH,
        "ci_image_tag": "pr-42",
        "ci_megatron_pr": "#77",
        "ci_sglang_pr": "feature/pin-x",
    }


@pytest.mark.parametrize(
    "line",
    ["ci-image-tag: -bad", "ci-megatron-pr: $(reboot)", "ci-sglang-pr: bad;ref"],
)
def test_file_run_rejects_an_invalid_pr_body_pin(line):
    target = pull()
    target["body"] = f"{line}\n"
    api = FakeAPI(target)

    with pytest.raises(HANDLER.CommentCommandError, match="unsupported value"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_fork_file_run_without_contributor_tier_or_write_is_denied(permission):
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission=permission)

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_dispatches_a_fork_file_run(permission):
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission=permission)

    result = HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)

    assert api.list_pull_calls == [("fork-owner", HEAD_REF)]
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"])
def test_prior_contributor_dispatches_a_fork_file_run_without_a_permission_lookup(author_association):
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission="none")

    result = HANDLER.process_event(event(body=RUN_FILE_BODY, author_association=author_association), policy(), api)

    assert api.permission_calls == []
    assert api.list_pull_calls == [("fork-owner", HEAD_REF)]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE", "MANNEQUIN"])
def test_first_time_contributor_cannot_dispatch_a_fork_file_run(author_association):
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY, author_association=author_association), policy(), api)
    assert api.dispatch_calls == []


def test_fork_file_run_requires_one_unique_head_pull():
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1))
    api.head_pulls = []

    with pytest.raises(HANDLER.CommentCommandError, match="exactly one pull request"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_file_run_requires_live_write_permission(permission):
    api = FakeAPI(pull(), permission=permission)

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.dispatch_calls == []


def test_add_label_access_user_id_cannot_dispatch_a_file_run():
    api = FakeAPI(pull(), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(user_ids=(ACTOR_ID,)), api)
    assert api.dispatch_calls == []


@pytest.mark.parametrize("author_association", ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"])
def test_prior_contributor_dispatches_a_file_run_without_a_permission_lookup(author_association):
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body=RUN_FILE_BODY, author_association=author_association),
        policy(),
        api,
    )

    assert api.permission_calls == []
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("author_association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE", "MANNEQUIN"])
def test_first_time_contributor_cannot_dispatch_a_file_run(author_association):
    api = FakeAPI(pull(), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body=RUN_FILE_BODY, author_association=author_association),
            policy(),
            api,
        )
    assert api.dispatch_calls == []


def test_prior_contributor_access_user_id_dispatches_a_file_run():
    api = FakeAPI(pull(), permission="none")

    result = HANDLER.process_event(
        event(body=RUN_FILE_BODY, author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert api.permission_calls == []
    assert api.dispatch_calls == [
        (
            "run-ci-file.yml",
            "main",
            {"pull_number": "123", "head_sha": HEAD_SHA, "test_file": RUN_FILE_PATH},
        )
    ]
    assert result["decision"] == "ALLOW_FILE_RUN_DISPATCHED"


@pytest.mark.parametrize("body", ["/run-ci-short", "/clear-labels"])
def test_contributor_association_grants_no_label_command(body):
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="read")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(event(body=body, author_association="CONTRIBUTOR"), policy(), api)

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []
    assert api.dispatch_calls == []


@pytest.mark.parametrize("body", ["/run-ci-short", "/clear-labels"])
def test_prior_contributor_access_user_id_grants_no_label_command(body):
    api = FakeAPI(pull(labels=("run-ci-short",)), permission="none")

    with pytest.raises(HANDLER.CommentCommandError, match="not authorized"):
        HANDLER.process_event(
            event(body=body, author_association="FIRST_TIMER"),
            policy(prior_user_ids=(ACTOR_ID,)),
            api,
        )

    assert api.add_calls == []
    assert api.remove_calls == []
    assert api.rerun_calls == []
    assert api.dispatch_calls == []


def test_contributor_reruns_failed_ci_without_a_permission_lookup():
    api = FakeAPI(pull(), permission="none")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    result = HANDLER.process_event(event(body="/rerun-failed-ci", author_association="CONTRIBUTOR"), policy(), api)

    assert api.permission_calls == []
    assert api.rerun_calls == [10]
    assert result["decision"] == "ALLOW_RERUN_REQUESTED"


def test_prior_contributor_access_user_id_reruns_failed_ci():
    api = FakeAPI(pull(), permission="none")
    workflow_file, workflow_path = HANDLER.RERUN_WORKFLOWS[0]
    api.workflow_runs[workflow_file] = [workflow_run(workflow_path)]

    result = HANDLER.process_event(
        event(body="/rerun-failed-ci", author_association="FIRST_TIMER"),
        policy(prior_user_ids=(ACTOR_ID,)),
        api,
    )

    assert api.permission_calls == []
    assert api.rerun_calls == [10]
    assert result["decision"] == "ALLOW_RERUN_REQUESTED"


@pytest.mark.parametrize("author_association", [None, True, "", "collaborator", "EVERYONE"])
def test_event_parser_rejects_an_invalid_author_association(author_association):
    invalid = event(body=RUN_FILE_BODY)
    if author_association is None:
        del invalid["comment"]["author_association"]
    else:
        invalid["comment"]["author_association"] = author_association

    with pytest.raises(HANDLER.CommentCommandError, match="author association is invalid"):
        HANDLER.parse_event(invalid)


def test_create_workflow_dispatch_requests_and_returns_exact_run_details(monkeypatch):
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "workflow_run_id": WORKFLOW_RUN_ID,
                    "run_url": WORKFLOW_RUN_API_URL,
                    "html_url": WORKFLOW_RUN_URL,
                }
            ).encode("utf-8")

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    result = HANDLER.GitHubAPI("secret-token").create_workflow_dispatch(
        "run-ci-file.yml", "feature/test", {"test_file": "tests/e2e/test_a.py"}
    )

    request, timeout = requests[0]
    assert request.full_url == (
        "https://api.github.com/repos/radixark/miles/actions/workflows/run-ci-file.yml/dispatches"
    )
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {
        "ref": "feature/test",
        "inputs": {"test_file": "tests/e2e/test_a.py"},
        "return_run_details": True,
    }
    assert timeout == 15
    assert result == WORKFLOW_RUN_URL


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [(204, b"", "expected 200"), (200, b"", "invalid JSON")],
)
def test_create_workflow_dispatch_rejects_unconfirmed_response(monkeypatch, status, body, message):
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_workflow_dispatch("run-ci-file.yml", "feature/test", {})
    assert attempts == 1


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({}, "workflow run ID must be a positive integer"),
        (
            {
                "workflow_run_id": WORKFLOW_RUN_ID,
                "run_url": "https://api.github.com/repos/radixark/miles/actions/runs/1",
                "html_url": WORKFLOW_RUN_URL,
            },
            "mismatched workflow dispatch details",
        ),
        (
            {
                "workflow_run_id": WORKFLOW_RUN_ID,
                "run_url": WORKFLOW_RUN_API_URL,
                "html_url": "https://example.invalid/actions/runs/987654321",
            },
            "workflow run URL is invalid",
        ),
    ],
)
def test_create_workflow_dispatch_rejects_invalid_run_details(monkeypatch, details, message):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(details).encode("utf-8")

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", lambda _request, timeout: Response())
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_workflow_dispatch("run-ci-file.yml", "main", {})


def test_create_issue_comment_uses_exact_endpoint_and_confirms_body(monkeypatch):
    requests = []

    class Response:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"body": ANNOUNCE_BODY, "id": NEW_COMMENT_ID}).encode("utf-8")

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    assert HANDLER.GitHubAPI("secret-token").create_issue_comment(123, ANNOUNCE_BODY) == NEW_COMMENT_ID

    request, timeout = requests[0]
    assert request.full_url == "https://api.github.com/repos/radixark/miles/issues/123/comments"
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"body": ANNOUNCE_BODY}
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (200, {"body": ANNOUNCE_BODY, "id": NEW_COMMENT_ID}, "expected 201"),
        (201, {"body": "wrong", "id": NEW_COMMENT_ID}, "did not confirm"),
        (201, {"body": ANNOUNCE_BODY}, "comment ID"),
    ],
)
def test_create_issue_comment_rejects_unconfirmed_response_without_retry(
    monkeypatch,
    status,
    body,
    message,
):
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(body).encode("utf-8")

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").create_issue_comment(123, ANNOUNCE_BODY)
    assert attempts == 1


@pytest.mark.parametrize("status", [200, 201])
def test_add_comment_reaction_accepts_created_or_existing_reaction(monkeypatch, status):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"content":"+1"}'

    response = Response()
    response.status = status

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")

    request, timeout = requests[0]
    assert request.full_url == ("https://api.github.com/repos/radixark/miles/issues/comments/5678/reactions")
    assert request.method == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"content": "+1"}
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (202, b'{"content":"+1"}', "expected 200 or 201"),
        (204, b"", "expected 200 or 201"),
        (201, b'{"content":"heart"}', "did not confirm"),
        (201, b"not-json", "invalid JSON"),
    ],
)
def test_add_comment_reaction_rejects_unconfirmed_response_without_retry(monkeypatch, status, body, message):
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return body

    response = Response()
    response.status = status

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 15
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")
    assert attempts == 1


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (TimeoutError(), "timed out"),
    ],
)
def test_comment_reaction_api_failure_is_not_retried(monkeypatch, exception, message):
    attempts = []

    def urlopen(_request, *, timeout):
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER.GitHubAPI("secret-token").add_comment_reaction(5678, "+1")
    assert attempts == [15]


def test_acknowledge_event_reacts_only_for_run_test_file():
    api = FakeAPI(pull())

    result = HANDLER.acknowledge_event(event(body=RUN_FILE_BODY), api)

    assert api.reaction_calls == [(5678, "+1")]
    assert result == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_SUCCESS_REACTION_CONFIRMED",
        "pull_number": 123,
        "reaction": "+1",
    }


def test_acknowledge_event_rejects_commands_without_a_success_reaction():
    api = FakeAPI(pull())

    with pytest.raises(HANDLER.CommentCommandError, match="does not define"):
        HANDLER.acknowledge_event(event(body="/rerun-failed-ci"), api)
    assert api.reaction_calls == []


@pytest.mark.parametrize("comment_id", [None, 0, "5678"])
def test_acknowledge_event_rejects_an_invalid_comment_id(comment_id):
    api = FakeAPI(pull())
    command_event = event(body=RUN_FILE_BODY)
    command_event["comment"]["id"] = comment_id

    with pytest.raises(HANDLER.CommentCommandError, match="comment ID must be a positive integer"):
        HANDLER.acknowledge_event(command_event, api)
    assert api.reaction_calls == []


def test_acknowledge_mode_reacts_without_loading_policy(monkeypatch, capsys):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body=RUN_FILE_BODY))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "reaction-token")
    monkeypatch.setenv("CI_COMMAND_ACKNOWLEDGE", "true")

    assert HANDLER.main() == 0
    assert api.calls == [("add_comment_reaction", 5678, "+1")]
    assert json.loads(capsys.readouterr().out) == {
        "actor_id": ACTOR_ID,
        "decision": "ALLOW_SUCCESS_REACTION_CONFIRMED",
        "pull_number": 123,
        "reaction": "+1",
    }


def test_acknowledge_mode_rejects_a_command_without_a_success_reaction(monkeypatch):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body="/rerun-failed-ci"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "reaction-token")
    monkeypatch.setenv("CI_COMMAND_ACKNOWLEDGE", "true")

    assert HANDLER.main() == 1
    assert api.calls == []


def test_announce_posts_a_running_status_comment_keyed_to_the_run(monkeypatch):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.announce_file_run(api, file_run_status("announce"))

    pull_number, body = api.comment_calls[0]
    assert pull_number == 123
    assert body == ANNOUNCE_BODY
    assert result == {
        "comment_id": NEW_COMMENT_ID,
        "decision": "FILE_RUN_ANNOUNCED",
        "pull_number": 123,
        "run_id": WORKFLOW_RUN_ID,
        "test_file": RUN_FILE_PATH,
    }


def test_report_updates_the_announced_comment_with_the_result_and_duration(monkeypatch):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)

    assert api.comment_calls == []
    comment_id, body = api.update_calls[0]
    assert comment_id == EXISTING_COMMENT_ID
    assert body == (
        f"\u2705 `{RUN_FILE_PATH}` **passed** on `stage-c-4-gpu-h200` in 3m11s "
        f"\u2014 [workflow run]({WORKFLOW_RUN_URL})\n\n{RUN_MARKER}"
    )
    assert result == {
        "comment_id": EXISTING_COMMENT_ID,
        "decision": "FILE_RUN_PASSED",
        "duration": "3m11s",
        "pull_number": 123,
        "run_id": WORKFLOW_RUN_ID,
        "test_file": RUN_FILE_PATH,
    }


@pytest.mark.parametrize(
    ("overrides", "decision", "icon", "explanation"),
    [
        (
            {"FILE_RUN_CUDA_RESULT": "failure"},
            "FILE_RUN_FAILED",
            "\u274c",
            "The execution job failed; inspect the workflow run for the failing step.",
        ),
        (
            {"FILE_RUN_RESOLVE_RESULT": "failure", "FILE_RUN_CUDA_RESULT": "skipped"},
            "FILE_RUN_FAILED",
            "\u274c",
            "The run never started: resolving the test file's execution plan failed.",
        ),
        (
            {"FILE_RUN_CUDA_RESULT": "cancelled"},
            "FILE_RUN_CANCELLED",
            "\u26aa",
            "The run was cancelled.",
        ),
        (
            {"FILE_RUN_RESOLVE_RESULT": "cancelled", "FILE_RUN_CUDA_RESULT": "skipped"},
            "FILE_RUN_CANCELLED",
            "\u26aa",
            "The run was cancelled.",
        ),
        (
            {"FILE_RUN_CUDA_RESULT": "skipped", "FILE_RUN_CPU_RESULT": "skipped"},
            "FILE_RUN_FAILED",
            "\u274c",
            "No execution job ran: the resolved plan selected neither the CUDA nor the CPU job.",
        ),
    ],
)
def test_report_never_calls_a_non_passing_run_a_pass(monkeypatch, overrides, decision, icon, explanation):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(**overrides), EXISTING_COMMENT_ID)

    body = api.update_calls[0][1]
    assert body.startswith(icon)
    assert "**passed**" not in body
    assert explanation in body
    assert result["decision"] == decision


def test_report_reads_the_run_start_from_the_api_not_from_an_input(monkeypatch):
    api = FakeAPI(pull())
    api.workflow_run = {"run_started_at": "2026-08-23T08:25:25Z"}
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    result = HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)

    assert api.run_calls == [WORKFLOW_RUN_ID]
    assert result["duration"] == "1h03m11s"


@pytest.mark.parametrize(
    "workflow_run",
    [
        {},
        {"created_at": RUN_STARTED_AT},
        {"run_started_at": ""},
        {"run_started_at": "not-a-timestamp"},
        {"run_started_at": "2026-08-23T09:25:25"},
        {"run_started_at": 17},
        "nope",
    ],
)
def test_report_fails_closed_on_an_unusable_run_start(monkeypatch, workflow_run):
    api = FakeAPI(pull())
    api.workflow_run = workflow_run
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)

    with pytest.raises(HANDLER.CommentCommandError):
        HANDLER.report_file_run(api, file_run_status(), EXISTING_COMMENT_ID)
    assert api.update_calls == []
    assert api.comment_calls == []


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(0, "0s"), (9, "9s"), (59, "59s"), (60, "1m00s"), (191, "3m11s"), (3600, "1h00m00s"), (3791, "1h03m11s")],
)
def test_duration_formatting_is_stable(seconds, formatted):
    assert HANDLER._format_duration(seconds) == formatted


def test_negative_duration_is_a_hard_error():
    with pytest.raises(HANDLER.CommentCommandError, match="negative"):
        HANDLER._format_duration(-1)


@pytest.mark.parametrize(
    ("mode", "overrides", "message"),
    [
        ("announce", {"CI_COMMAND_FILE_RUN_STATUS": "publish"}, "announce or report"),
        ("announce", {"FILE_RUN_PULL_NUMBER": "0"}, "pull request number"),
        ("announce", {"FILE_RUN_PULL_NUMBER": "12a"}, "pull request number"),
        ("announce", {"FILE_RUN_PULL_NUMBER": ""}, "pull request number"),
        ("announce", {"FILE_RUN_RUN_ID": "-1"}, "workflow run ID"),
        ("announce", {"FILE_RUN_TEST_FILE": "tests/../etc/passwd"}, "test file is invalid"),
        ("announce", {"FILE_RUN_TEST_FILE": ""}, "test file is invalid"),
        ("report", {"FILE_RUN_SUITE": "stage a cpu; rm -rf /"}, "suite is invalid"),
        ("report", {"FILE_RUN_RESOLVE_RESULT": "exploded"}, "resolve result is invalid"),
        ("report", {"FILE_RUN_CPU_RESULT": "success"}, "two executing jobs"),
        ("report", {"FILE_RUN_COMMENT_ID": ""}, "comment ID"),
        ("report", {"FILE_RUN_COMMENT_ID": "0"}, "comment ID"),
    ],
)
def test_file_run_status_inputs_fail_closed(mode, overrides, message):
    with pytest.raises(HANDLER.CommentCommandError, match=message):
        HANDLER._file_run_status_inputs(file_run_env(mode, **overrides))


def test_status_mode_needs_no_event_and_loads_no_policy(monkeypatch, capsys):
    api = FakeAPI(pull())
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: pytest.fail("no event payload may be read"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    for name, value in file_run_env("report").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "FILE_RUN_PASSED"


def test_announce_mode_writes_the_created_comment_id(monkeypatch, tmp_path, capsys):
    api = FakeAPI(pull())
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: api)
    monkeypatch.setattr(HANDLER, "_utc_now", lambda: REPORT_NOW)
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    for name, value in file_run_env("announce").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 0
    assert output_path.read_text() == f"comment_id={NEW_COMMENT_ID}\n"
    assert json.loads(capsys.readouterr().out)["decision"] == "FILE_RUN_ANNOUNCED"


def test_status_mode_reports_a_handler_error_as_a_failed_step(monkeypatch, capsys):
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: pytest.fail("token is validated first"))
    monkeypatch.setenv("CI_COMMAND_API_TOKEN", "status-token")
    for name, value in file_run_env("report", FILE_RUN_TEST_FILE="not-a-test").items():
        monkeypatch.setenv(name, value)

    assert HANDLER.main() == 1
    assert "::error::file run test file is invalid" in capsys.readouterr().out


def test_the_gateway_no_longer_owns_a_reply_path():
    # The run itself reports progress and the result; a second static comment
    # from the gateway would duplicate it.
    assert not hasattr(HANDLER, "reply_event")
    assert not hasattr(HANDLER, "_write_workflow_run_url")


def test_workflow_runs_only_trusted_code_with_minimal_permissions():
    workflow = WORKFLOW_PATH.read_text()
    assert "issue_comment:\n    types: [created]" in workflow
    assert "github.event.comment.body" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    # The command App exists only for the issues capability: label mutations
    # made with GITHUB_TOKEN would never trigger the labeled CI workflows.
    assert workflow.count("actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1") == 1
    assert workflow.count("client-id: ${{ vars.CI_COMMAND_APP_CLIENT_ID }}") == 1
    assert workflow.count("private-key: ${{ secrets.CI_COMMAND_APP_PRIVATE_KEY }}") == 1
    assert "permission-issues: write" in workflow
    assert "permission-actions" not in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ github.token }}" in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ steps.issues-token.outputs.token }}" in workflow
    assert "CI_COMMAND_APP_TOKEN" not in workflow
    assert workflow.index("CI_COMMAND_PREFLIGHT") < workflow.index("actions/create-github-app-token@")
    # The actions capability never waits on the App gate; label commands fail
    # loudly when the App is not enabled instead of skipping silently.
    assert "vars.CI_COMMAND_APP_ENABLED != 'true'" in workflow
    assert "vars.CI_COMMAND_APP_ENABLED == 'true'" not in workflow
    assert "steps.authorize.outputs.capability != 'none'" in workflow
    assert "steps.authorize.outputs.capability != 'issues'" in workflow
    assert "steps.authorize.outputs.capability != 'actions'" in workflow
    assert "capability: ${{ steps.authorize.outputs.capability }}" in workflow
    assert "success_reaction: ${{ steps.authorize.outputs.success_reaction }}" in workflow
    assert "needs: handle-command" in workflow
    assert "if: needs.handle-command.outputs.capability == 'actions'" in workflow
    assert "group: comment-ci-actions-${{ github.event.issue.number }}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "queue: max" in workflow
    handle_job = workflow.split("  handle-command:", 1)[1].split("  actions-command:", 1)[0]
    actions_job = workflow.split("  actions-command:", 1)[1].split("  acknowledge-command:", 1)[0]
    acknowledge_job = workflow.split("  acknowledge-command:", 1)[1]
    issues_token = workflow.split("- name: Mint the issues-scoped App token", 1)[1].split(
        "- name: Authorize and run the issues command", 1
    )[0]
    assert "permission-issues: write" in issues_token
    # The label lands on a pull request, and GitHub gates issues-API calls whose
    # target issue is a PR on pull-requests scope, so the App token needs write.
    assert "permission-pull-requests: write" in issues_token
    assert "permission-pull-requests: read" not in issues_token
    assert "Require the command App for label commands" in handle_job
    assert handle_job.index("Require the command App for label commands") < handle_job.index(
        "Mint the issues-scoped App token"
    )
    # Each GITHUB_TOKEN job scopes its own permissions; only the actions job
    # may dispatch or rerun, and only the acknowledgement job may write issues.
    assert ("permissions:\n      contents: read\n      actions: write\n      pull-requests: read") in actions_job
    assert "issues: write" not in actions_job
    assert "create-github-app-token" not in actions_job
    assert "CI_COMMAND_API_TOKEN: ${{ github.token }}" in actions_job
    # The acknowledgement job targets a comment on a pull request, and issues-API calls
    # whose target issue is a PR are gated on pull-requests scope.
    assert "permissions:\n      contents: read\n      issues: write\n      pull-requests: write" in acknowledge_job
    assert "actions: write" not in acknowledge_job
    assert "create-github-app-token" not in acknowledge_job
    assert workflow.index("Authorize and run the actions command") < workflow.index(
        "Acknowledge the successful command"
    )
    assert "CI_COMMAND_ACKNOWLEDGE" not in actions_job
    assert "needs: [handle-command, actions-command]" in acknowledge_job
    assert (
        "if: >-\n"
        "      needs.actions-command.result == 'success' &&\n"
        "      needs.handle-command.outputs.success_reaction == '+1'"
    ) in acknowledge_job
    assert "always()" not in acknowledge_job
    assert 'CI_COMMAND_ACKNOWLEDGE: "true"' in acknowledge_job
    assert "CI_COMMAND_REPLY" not in actions_job
    assert "workflow_run_url:" not in actions_job
    assert "reply-command:" not in workflow
    assert "CI_COMMAND_REPLY" not in workflow
    assert "CI_COMMAND_WORKFLOW_RUN_URL" not in workflow
    assert "pull_request_target" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "pip install" not in workflow
    assert "contents: write" not in workflow
    assert "\n  actions: write" not in workflow
