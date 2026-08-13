import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.labels import KNOWN_LABELS

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
HANDLER_PATH = ROOT / ".github/workflows/scripts/comment_ci_label.py"
POLICY_PATH = ROOT / ".github/workflows/policies/ci-label-access.json"
WORKFLOW_PATH = ROOT / ".github/workflows/comment-ci-label.yml"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDLER = load_module("comment_ci_label", HANDLER_PATH)
ACTOR_ID = 1234
WRITE_PERMISSIONS = frozenset({"write", "admin"})


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


def pull(*, head_repository_id=HANDLER.REPOSITORY_ID, state="open", labels=()):
    return {
        "number": 123,
        "state": state,
        "base": {"repo": {"id": HANDLER.REPOSITORY_ID, "full_name": HANDLER.REPOSITORY}},
        "head": {"repo": {"id": head_repository_id}},
        "labels": [{"name": label} for label in labels],
    }


def policy(*, permissions=("write", "admin")):
    return {"labels": {"run-ci-short": frozenset(permissions)}}


@pytest.mark.parametrize(
    "body",
    [
        "/run-ci-short extra",
        "/run-ci-short\n/run-ci-long",
        "please /run-ci-short",
        "/run-ci-",
        "/run-ci-/unsafe",
        "/rerun-failed-ci",
    ],
)
def test_command_parser_rejects_non_exact_commands(body):
    with pytest.raises(HANDLER.CommentLabelError, match="only /run-ci-<key>"):
        HANDLER.parse_command(body)


def test_command_parser_accepts_one_exact_command_with_outer_whitespace():
    assert HANDLER.parse_command(" \n/run-ci-a_B.c-d\t") == "run-ci-a_B.c-d"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"version":1,"version":1,"labels":{}}', "duplicate JSON key"),
        ('{"version":NaN,"labels":{}}', "non-standard JSON number"),
        ('{"version":1,"labels":{"run-ci-short":[true]}}', "only write or admin"),
        ('{"version":1,"labels":{"run-ci-short":["read"]}}', "only write or admin"),
        ('{"version":1,"labels":{"run-ci-short":["write","write"]}}', "duplicate permissions"),
        ('{"version":1,"labels":{"nightly":["write"]}}', "invalid exact CI label"),
        ('{"version":1,"labels":{"run-ci-short":["write"]},"roles":{}}', "only version"),
    ],
)
def test_policy_parser_rejects_ambiguous_or_expanded_schema(tmp_path, text, message):
    path = tmp_path / "policy.json"
    path.write_text(text)
    with pytest.raises(HANDLER.CommentLabelError, match=message):
        HANDLER.load_policy(path)


def test_checked_in_policy_is_exact_and_requires_write_permission():
    loaded = HANDLER.load_policy(POLICY_PATH)
    assert set(loaded["labels"]) == {f"run-ci-{key}" for key in KNOWN_LABELS} | {"run-ci-image"}
    assert set(loaded["labels"].values()) == {WRITE_PERMISSIONS}
    assert all(HANDLER.LABEL_PATTERN.fullmatch(label) for label in loaded["labels"])


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


@pytest.mark.parametrize(("permission", "allowed"), [("write", False), ("admin", True)])
def test_exact_label_policy_can_require_admin(permission, allowed):
    api = FakeAPI(pull(), permission=permission)

    if allowed:
        assert HANDLER.process_event(event(), policy(permissions=("admin",)), api)["decision"] == "ALLOW_ADDED"
    else:
        with pytest.raises(HANDLER.CommentLabelError, match="not authorized"):
            HANDLER.process_event(event(), policy(permissions=("admin",)), api)


def test_existing_label_is_an_authorized_no_op():
    api = FakeAPI(pull(labels=("run-ci-short", "documentation")))

    result = HANDLER.process_event(event(), policy(), api)

    assert result["decision"] == "ALLOW_ALREADY_PRESENT"
    assert api.add_calls == []


def test_unknown_request_does_not_call_github_api():
    api = FakeAPI(pull())
    with pytest.raises(HANDLER.CommentLabelError, match="not exposed"):
        HANDLER.process_event(event(body="/run-ci-unknown"), policy(), api)
    assert api.calls == []


@pytest.mark.parametrize("permission", ["read", "none", "unknown"])
def test_caller_without_write_permission_cannot_mutate(permission):
    api = FakeAPI(pull(), permission=permission)
    with pytest.raises(HANDLER.CommentLabelError, match="not authorized"):
        HANDLER.process_event(event(), policy(), api)
    assert api.calls == [("get_pull", 123), ("get_permission", "actor")]
    assert api.add_calls == []


def test_preflight_checks_permission_without_reading_pull_request():
    api = FakeAPI(pull())

    assert HANDLER.authorize_policy(event(), policy(), api) == (123, ACTOR_ID, "run-ci-short")
    assert api.calls == [("get_permission", "actor")]


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_repository_writer_can_add_a_label_to_a_fork_pr(permission):
    api = FakeAPI(pull(head_repository_id=999), permission=permission, permission_actor_id=2)

    result = HANDLER.process_event(event(actor_id=2), policy(), api)

    assert result["decision"] == "ALLOW_ADDED"
    assert api.add_calls == [(123, "run-ci-short")]


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
    with pytest.raises(HANDLER.CommentLabelError, match=message):
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

    with pytest.raises(HANDLER.CommentLabelError, match=message):
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
    with pytest.raises(HANDLER.CommentLabelError, match=message):
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
    with pytest.raises(HANDLER.CommentLabelError, match=message):
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
    with pytest.raises(HANDLER.CommentLabelError, match="invalid JSON"):
        HANDLER.GitHubAPI("secret-token").get_permission("actor")
    assert attempts == 1


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (urllib.error.HTTPError("url", 403, "forbidden", {}, None), "HTTP 403"),
        (TimeoutError(), "timed out"),
    ],
)
def test_label_api_failure_is_not_retried(monkeypatch, exception, message):
    attempts = []

    def urlopen(_request, *, timeout):
        attempts.append(timeout)
        raise exception

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    with pytest.raises(HANDLER.CommentLabelError, match=message):
        HANDLER.GitHubAPI("secret-token").add_label(123, "run-ci-short")
    assert attempts == [15]


def test_unconfirmed_mutation_response_fails_without_rollback():
    api = FakeAPI(pull())
    api.add_label = lambda _pull_number, _label: []
    with pytest.raises(HANDLER.CommentLabelError, match="did not confirm"):
        HANDLER.process_event(event(), policy(), api)


def test_workflow_runs_only_trusted_code_with_minimal_permissions():
    workflow = WORKFLOW_PATH.read_text()
    assert "issue_comment:\n    types: [created]" in workflow
    assert "vars.CI_LABEL_APP_ENABLED == 'true'" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1" in workflow
    assert "client-id: ${{ vars.CI_LABEL_APP_CLIENT_ID }}" in workflow
    assert "private-key: ${{ secrets.CI_LABEL_APP_PRIVATE_KEY }}" in workflow
    assert "permission-issues: write" in workflow
    assert "permission-pull-requests: read" in workflow
    assert "CI_LABEL_API_TOKEN: ${{ github.token }}" in workflow
    assert "CI_LABEL_API_TOKEN: ${{ steps.app-token.outputs.token }}" in workflow
    assert "CI_LABEL_APP_TOKEN" not in workflow
    assert workflow.index("CI_LABEL_PREFLIGHT") < workflow.index("actions/create-github-app-token@")
    assert "pull_request_target" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "pip install" not in workflow
    assert "contents: write" not in workflow
    assert "actions: write" not in workflow
