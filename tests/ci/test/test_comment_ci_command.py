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


def event(*, body="/run-ci-short", actor_id=ACTOR_ID):
    return {
        "action": "created",
        "repository": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY},
        "issue": {"number": 123, "pull_request": {"url": "https://example.invalid/pulls/123"}},
        "comment": {
            "body": body,
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
    labels=("run-ci-short", "bypass-fastfail"),
    repo_permissions=("write", "admin"),
):
    return {
        "groups": {
            "add_label_access": {
                "repository_permissions": frozenset(permissions),
                "user_ids": frozenset(user_ids),
            },
            "repo_write_access": {
                "repository_permissions": frozenset(repo_permissions),
                "user_ids": frozenset(),
            },
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": frozenset(labels),
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "repo_write_access"},
            "run_test_file": {"group": "repo_write_access"},
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
        HANDLER.AddLabel: ("add_label", "issues", HANDLER._handle_add_label),
        HANDLER.ClearLabels: ("clear_labels", "issues", HANDLER._handle_clear_labels),
        HANDLER.RerunFailedCI: ("rerun_failed_ci", "actions", HANDLER._handle_rerun_failed_ci),
        HANDLER.RunTestFile: ("run_test_file", "actions", HANDLER._handle_run_test_file),
    }
    assert set(HANDLER.COMMAND_REGISTRY) == set(expected)
    for request_type, (policy_key, capability, handler) in expected.items():
        spec = HANDLER.COMMAND_REGISTRY[request_type]
        assert (spec.policy_key, spec.capability, spec.handler) == (policy_key, capability, handler)


def test_unknown_request_type_fails_closed():
    class UnknownRequest:
        pass

    with pytest.raises(HANDLER.CommentCommandError, match="not registered"):
        HANDLER._command_spec(UnknownRequest())


def raw_policy():
    return {
        "version": 2,
        "groups": {
            "add_label_access": {
                "repository_permissions": ["write", "admin"],
                "user_ids": [],
            },
            "repo_write_access": {"repository_permissions": ["write", "admin"]},
        },
        "commands": {
            "add_label": {
                "group": "add_label_access",
                "allowed_labels": ["run-ci-short", "bypass-fastfail"],
            },
            "clear_labels": {"group": "repo_write_access"},
            "rerun_failed_ci": {"group": "repo_write_access"},
            "run_test_file": {"group": "repo_write_access"},
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
        (("version",), 1, "version must be 2"),
        (("groups", "add_label_access", "repository_permissions"), [True], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["read"], "only write or admin"),
        (("groups", "add_label_access", "repository_permissions"), ["write", "write"], "duplicate permissions"),
        (("groups", "add_label_access", "user_ids"), [True], "only positive integers"),
        (("groups", "add_label_access", "user_ids"), [123, 123], "duplicate user IDs"),
        (("commands", "add_label", "allowed_labels"), ["unsafe label"], "invalid exact CI label"),
        (("commands", "add_label", "allowed_labels"), ["run-ci-short", "run-ci-short"], "duplicate labels"),
        (("commands", "add_label", "group"), "missing", "unknown group"),
        (("commands", "clear_labels", "unexpected"), True, "invalid fields"),
        (("groups", "repo_write_access", "user_ids"), [123], "invalid fields"),
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


@pytest.mark.parametrize("section", ["groups", "commands"])
def test_policy_parser_rejects_unknown_group_or_command(tmp_path, section):
    raw = raw_policy()
    raw[section]["unexpected"] = {}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    expected = "groups must contain only" if section == "groups" else "commands do not match"
    with pytest.raises(HANDLER.CommentCommandError, match=expected):
        HANDLER.load_policy(path)


def test_non_label_commands_cannot_use_a_group_with_explicit_user_ids(tmp_path):
    raw = raw_policy()
    raw["groups"]["add_label_access"]["user_ids"] = [ACTOR_ID]
    raw["commands"]["clear_labels"]["group"] = "add_label_access"
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(HANDLER.CommentCommandError, match="cannot grant access by user ID"):
        HANDLER.load_policy(path)


def test_checked_in_policy_exposes_exact_labels_and_add_label_access_group():
    loaded = HANDLER.load_policy(POLICY_PATH)
    labels = loaded["commands"]["add_label"]["allowed_labels"]
    assert labels == {f"run-ci-{key}" for key in KNOWN_LABELS} | {
        "bypass-fastfail",
        "run-ci-image",
    }
    assert loaded["groups"]["add_label_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
    }
    assert loaded["groups"]["repo_write_access"] == {
        "repository_permissions": WRITE_PERMISSIONS,
        "user_ids": frozenset(),
    }
    assert loaded["commands"]["clear_labels"] == {"group": "repo_write_access"}
    assert loaded["commands"]["rerun_failed_ci"] == {"group": "repo_write_access"}
    assert loaded["commands"]["run_test_file"] == {"group": "repo_write_access"}
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
    ("body", "capability"),
    [
        ("/run-ci-short", "issues"),
        ("/bypass-fastfail", "issues"),
        ("/clear-labels", "issues"),
        ("/rerun-failed-ci", "actions"),
        (RUN_FILE_BODY, "actions"),
    ],
)
def test_preflight_writes_only_a_fixed_capability(monkeypatch, tmp_path, body, capability):
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
    assert output_path.read_text() == f"capability={capability}\n"
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


def test_unrelated_comment_preflight_uses_no_policy_or_api(monkeypatch, tmp_path):
    output_path = tmp_path / "github-output"
    monkeypatch.setattr(HANDLER, "load_json", lambda _path: event(body="ordinary review comment"))
    monkeypatch.setattr(HANDLER, "load_policy", lambda _path: pytest.fail("policy must not be loaded"))
    monkeypatch.setattr(HANDLER, "GitHubAPI", lambda _token: pytest.fail("API must not be initialized"))
    monkeypatch.setenv("GITHUB_EVENT_PATH", "event.json")
    monkeypatch.setenv("CI_COMMAND_PREFLIGHT", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    assert HANDLER.main() == 0
    assert output_path.read_text() == "capability=none\n"


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
    }


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


def test_file_run_rejects_fork_prs_before_authorization():
    api = FakeAPI(pull(head_repository_id=HANDLER.REPOSITORY_ID + 1))

    with pytest.raises(HANDLER.CommentCommandError, match="same-repository"):
        HANDLER.process_event(event(body=RUN_FILE_BODY), policy(), api)
    assert api.permission_calls == []
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


def test_create_workflow_dispatch_uses_exact_endpoint_and_accepts_empty_204(monkeypatch):
    requests = []

    class Response:
        status = 204

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
    HANDLER.GitHubAPI("secret-token").create_workflow_dispatch(
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
    }
    assert timeout == 15


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [(200, b"", "expected 204"), (204, b"{}", "unexpected response body")],
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


def test_workflow_runs_only_trusted_code_with_minimal_permissions():
    workflow = WORKFLOW_PATH.read_text()
    assert "issue_comment:\n    types: [created]" in workflow
    assert "vars.CI_COMMAND_APP_ENABLED == 'true'" in workflow
    assert "github.event.comment.body" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert workflow.count("actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1") == 2
    assert "client-id: ${{ vars.CI_COMMAND_APP_CLIENT_ID }}" in workflow
    assert "private-key: ${{ secrets.CI_COMMAND_APP_PRIVATE_KEY }}" in workflow
    assert "permission-issues: write" in workflow
    assert workflow.count("permission-actions: write") == 1
    assert "permission-pull-requests: read" in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ github.token }}" in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ steps.issues-token.outputs.token }}" in workflow
    assert "CI_COMMAND_API_TOKEN: ${{ steps.actions-token.outputs.token }}" in workflow
    assert "CI_COMMAND_APP_TOKEN" not in workflow
    assert workflow.index("CI_COMMAND_PREFLIGHT") < workflow.index("actions/create-github-app-token@")
    assert "steps.authorize.outputs.capability != 'none'" in workflow
    assert "steps.authorize.outputs.capability != 'issues'" in workflow
    assert "steps.authorize.outputs.capability != 'actions'" in workflow
    assert "capability: ${{ steps.authorize.outputs.capability }}" in workflow
    assert "needs: handle-command" in workflow
    assert "if: needs.handle-command.outputs.capability == 'actions'" in workflow
    assert "group: comment-ci-actions-${{ github.event.issue.number }}" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "queue: max" in workflow
    issues_token = workflow.split("- name: Mint the issues-scoped App token", 1)[1].split(
        "- name: Authorize and run the issues command", 1
    )[0]
    actions_token = workflow.split("- name: Mint the actions-scoped App token", 1)[1].split(
        "- name: Authorize and run the actions command", 1
    )[0]
    assert "permission-issues: write" in issues_token
    assert "permission-actions: write" not in issues_token
    assert "permission-actions: write" in actions_token
    assert "permission-issues: write" not in actions_token
    assert "pull_request_target" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "pip install" not in workflow
    assert "contents: write" not in workflow
    assert "\n  actions: write" not in workflow
