import hashlib
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT_DIR = ROOT / ".github/workflows/scripts"
ANALYZER_PATH = SCRIPT_DIR / "ci_failure_analysis.py"
POLICY_PATH = ROOT / ".github/workflows/policies/ci-failure-analysis.json"
PROMPT_PATH = ROOT / ".github/workflows/prompts/ci-failure-analysis.md"
SCHEMA_PATH = ROOT / ".github/workflows/policies/ci-failure-response-schema.json"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = load_module("ci_failure_analysis_test", ANALYZER_PATH)
SECRET = "ghp_012345678901234567890123456789012345"


def run(**overrides):
    value = {
        "id": 123,
        "name": "PR Test",
        "run_attempt": 1,
        "head_sha": "a" * 40,
        "repository": {"id": 1072725553, "full_name": "radixark/miles"},
    }
    value.update(overrides)
    return value


def job(job_id=10, name="unit", conclusion="failure"):
    return {"id": job_id, "name": name, "conclusion": conclusion, "html_url": f"https://example/jobs/{job_id}"}


def write_policy(tmp_path, **overrides):
    value = json.loads(POLICY_PATH.read_text())
    value.update({"enabled": True, **overrides})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value))
    return path


class FakeGitHub:
    def __init__(self, logs=None, pulls=None, files=None, contents=None):
        self.logs = logs or {}
        self.pulls = [] if pulls is None else pulls
        self.files = [] if files is None else files
        self.contents = {} if contents is None else contents
        self.calls = []

    def job_log(self, job_id, max_bytes):
        self.calls.append(("job_log", job_id, max_bytes))
        value = self.logs.get(job_id)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise RuntimeError("missing")
        return value

    def pulls_for_commit(self, sha):
        self.calls.append(("pulls_for_commit", sha))
        return self.pulls

    def pull_files(self, number):
        self.calls.append(("pull_files", number))
        return self.files

    def file_content(self, path, sha, max_bytes):
        self.calls.append(("file_content", path, sha, max_bytes))
        value = self.contents[path]
        return value[:max_bytes]


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if callable(self.response):
            return self.response(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = FakeResponses(response, error)


class APIConnectionError(Exception):
    def __init__(self, message, request_url):
        super().__init__(message)
        self.request = SimpleNamespace(url=request_url)


class ConnectError(Exception):
    def __init__(self, message, request_url):
        super().__init__(message)
        self.request = SimpleNamespace(url=request_url)


def with_cause(error, cause):
    error.__cause__ = cause
    return error


def valid_response(kwargs):
    packet = json.loads(kwargs["input"])
    analyses = []
    for item in packet["jobs"]:
        analyses.append(
            {
                "job_id": item["job_id"],
                "reason": "The assertion failed because the returned value was 3 instead of 4.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": [item["evidence_refs"][0]],
            }
        )
    return SimpleNamespace(
        output_text=json.dumps({"schema_version": "1", "analyses": analyses}),
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 20, "output_tokens": 10}),
    )


def analyze(tmp_path, jobs, gh, client, **policy_overrides):
    emitted = []
    outcome = ANALYZER.analyze_failures(
        run=run(),
        jobs=jobs,
        repo="radixark/miles",
        gh=gh,
        policy_path=write_policy(tmp_path, **policy_overrides),
        prompt_path=PROMPT_PATH,
        schema_path=SCHEMA_PATH,
        client_factory=lambda timeout: client,
        emit=emitted.append,
    )
    return outcome, emitted


def test_checked_in_policy_is_valid_and_disabled():
    policy = ANALYZER.load_policy(POLICY_PATH)
    assert policy.enabled is False
    assert policy.max_jobs == 15
    assert policy.max_model_calls == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"schema_version":"1","schema_version":"1"}', "duplicate JSON field"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "extra": 1}), "unknown policy field"),
        (
            json.dumps({key: value for key, value in json.loads(POLICY_PATH.read_text()).items() if key != "mode"}),
            "missing policy field",
        ),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "max_jobs": 16}), "max_jobs"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "model": "arbitrary-model"}), "unapproved policy model"),
        (json.dumps({**json.loads(POLICY_PATH.read_text()), "enabled": 1}), "enabled"),
    ],
)
def test_policy_rejects_invalid_or_expansive_configuration(tmp_path, contents, message):
    path = tmp_path / "policy.json"
    path.write_text(contents)
    with pytest.raises(ANALYZER.AnalysisConfigError, match=message):
        ANALYZER.load_policy(path)


def test_prompt_sha_is_the_git_blob_sha():
    prompt, blob_sha = ANALYZER.load_prompt(PROMPT_PATH)
    data = PROMPT_PATH.read_bytes()
    assert prompt
    assert blob_sha == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_redaction_precedes_bounded_evidence_extraction():
    raw = (
        f"\x1b[31mAuthorization: Bearer top-secret\x1b[0m\nTOKEN={SECRET}\n"
        "AWS_ACCESS_KEY_ID=AKIA0123456789ABCDEF\n"
        "https://example.invalid/file?X-Amz-Signature=signed-secret\n"
        "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----\n"
        "tests/unit/test_math.py:12: AssertionError: expected 4\x00\n"
    )
    evidence = ANALYZER.extract_log_evidence(raw, 10, 2000)
    assert evidence is not None
    assert "AssertionError" in evidence["text"]
    assert "\x1b" not in evidence["text"] and "\x00" not in evidence["text"]
    assert SECRET not in evidence["text"] and "top-secret" not in evidence["text"]
    assert "AKIA0123456789ABCDEF" not in evidence["text"] and "signed-secret" not in evidence["text"]
    assert "private-material" not in evidence["text"]


def test_log_windows_are_deterministic_and_unicode_safe():
    text = "\n".join(
        [*(f"line {index}" for index in range(80)), "AssertionError: 🐦 failure", *("tail" for _ in range(80))]
    )
    first = ANALYZER.extract_log_evidence(text, 10, 180)
    second = ANALYZER.extract_log_evidence(text, 10, 180)
    assert first == second
    assert len(first["text"]) <= 180
    first["text"].encode("utf-8")


def test_source_locations_keep_only_safe_repository_paths():
    text = (
        'File "/home/runner/work/miles/miles/tests/ci/test/test_gate.py", line 41\n'
        "miles/core/train.py:22: failure\n"
        "../secrets/key.py:1\nhttps://evil.invalid/a.py:2\n/tmp/random.py:3"
    )
    locations = ANALYZER.extract_source_locations(text)
    assert any(path.endswith("tests/ci/test/test_gate.py") and line == 41 for path, line in locations)
    assert ("miles/core/train.py", 22) in locations
    assert all(
        ".." not in path and "evil.invalid" not in path and not path.startswith("tmp/") for path, _ in locations
    )


def test_validate_response_accepts_exact_job_and_evidence_contract():
    jobs = [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}]
    raw = {
        "schema_version": "1",
        "analyses": [
            {
                "job_id": 10,
                "reason": "The assertion failed because the result was empty.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": ["job:10:log:1-2"],
            }
        ],
    }
    assert ANALYZER.validate_response(json.dumps(raw), jobs, 280)[10].startswith("The assertion")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(extra="bad"),
        lambda item: item.update(job_id=11),
        lambda item: item.update(reason="One sentence. Another sentence."),
        lambda item: item.update(reason="First cause. second cause."),
        lambda item: item.update(reason="See https://evil.invalid/a."),
        lambda item: item.update(reason="See www.evil.invalid for details."),
        lambda item: item.update(reason="See //evil.invalid for details."),
        lambda item: item.update(reason="The **assertion** failed."),
        lambda item: item.update(reason="x" * 280 + "."),
        lambda item: item.update(reason="missing terminal punctuation"),
        lambda item: item.update(category="remediation"),
        lambda item: item.update(evidence_refs=["job:other:log:1"]),
    ],
)
def test_validate_response_rejects_untrusted_or_unverifiable_output(mutation):
    item = {
        "job_id": 10,
        "reason": "The assertion failed because the result was empty.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
    }
    mutation(item)
    with pytest.raises(ValueError):
        ANALYZER.validate_response(
            json.dumps({"schema_version": "1", "analyses": [item]}),
            [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
            280,
        )


def test_disabled_policy_makes_no_github_or_model_call(tmp_path):
    client = FakeClient(response=valid_response)
    gh = FakeGitHub(logs={10: "AssertionError: no"})
    outcome = ANALYZER.analyze_failures(
        run=run(),
        jobs=[job()],
        repo="radixark/miles",
        gh=gh,
        policy_path=POLICY_PATH,
        client_factory=lambda timeout: client,
    )
    assert outcome == ANALYZER.AnalysisOutcome(enabled=False, reasons={})
    assert gh.calls == [] and client.responses.calls == []


def test_missing_analysis_app_token_preserves_base_card_contract(tmp_path):
    outcome, emitted = analyze(tmp_path, [job()], None, FakeClient(response=valid_response))
    assert outcome.unavailable and outcome.reasons == {}
    assert '"stage":"github_auth"' in emitted[0]


def test_all_missing_logs_get_per_row_fallback_without_model_call(tmp_path):
    client = FakeClient(response=valid_response)
    outcome, emitted = analyze(tmp_path, [job(10), job(11)], FakeGitHub(), client)
    assert outcome.reasons == {10: ANALYZER.UNAVAILABLE_REASON, 11: ANALYZER.UNAVAILABLE_REASON}
    assert not outcome.unavailable and client.responses.calls == []
    assert '"request_count":0' in emitted[0]


def test_four_jobs_are_batched_into_one_bounded_tool_free_request(tmp_path):
    logs = {
        index: f"TOKEN={SECRET}\ntests/ci/test/test_gate.py:12: AssertionError: value {index}"
        for index in range(10, 14)
    }
    client = FakeClient(response=valid_response)
    outcome, emitted = analyze(tmp_path, [job(index) for index in range(10, 14)], FakeGitHub(logs=logs), client)
    assert len(outcome.reasons) == 4 and not outcome.unavailable
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["store"] is False and request["tools"] == []
    assert request["model"] == "gpt-5.6-luna"
    assert request["text"]["format"]["type"] == "json_schema"
    assert SECRET not in request["input"] and SECRET not in "".join(emitted)
    packet = json.loads(request["input"])
    assert [item["job_id"] for item in packet["jobs"]] == [10, 11, 12, 13]
    assert sum(len(item["text"]) for item in packet["evidence"]) <= 60000
    assert '"request_count":1' in emitted[0] and '"validation":"valid"' in emitted[0]


def test_unique_pr_and_trace_path_add_bounded_context_at_exact_sha(tmp_path):
    log = 'Traceback (most recent call last):\n  File "/work/miles/tests/ci/test/test_gate.py", line 2\nAssertionError'
    gh = FakeGitHub(
        logs={10: log},
        pulls=[{"number": 55, "title": "Change gate", "body": "Updates assertion"}],
        files=[{"filename": "tests/ci/test/test_gate.py", "patch": "@@ -1 +1 @@\n-old\n+new"}],
        contents={"tests/ci/test/test_gate.py": "one\ntwo\nthree\n"},
    )
    client = FakeClient(response=valid_response)
    outcome, _ = analyze(tmp_path, [job()], gh, client)
    assert not outcome.unavailable
    assert ("pulls_for_commit", "a" * 40) in gh.calls
    assert ("pull_files", 55) in gh.calls
    assert any(call[:3] == ("file_content", "tests/ci/test/test_gate.py", "a" * 40) for call in gh.calls)
    evidence = json.loads(client.responses.calls[0]["input"])["evidence"]
    assert {item["kind"] for item in evidence} == {"job_log", "pull_request", "source"}


def test_invalid_model_response_discards_partial_reasons_and_marks_unavailable(tmp_path):
    response = SimpleNamespace(
        output_text=json.dumps(
            {
                "schema_version": "1",
                "analyses": [
                    {
                        "job_id": 10,
                        "reason": "The first assertion failed.",
                        "category": "test_failure",
                        "confidence": "high",
                        "evidence_refs": ["unknown"],
                    }
                ],
            }
        )
    )
    outcome, emitted = analyze(
        tmp_path,
        [job(10), job(11)],
        FakeGitHub(logs={10: "AssertionError: one", 11: "AssertionError: two"}),
        FakeClient(response=response),
    )
    assert outcome.unavailable and outcome.reasons == {}
    assert '"validation":"error"' in emitted[0]


def test_policy_cap_omits_hidden_jobs_and_records_count(tmp_path):
    client = FakeClient(response=valid_response)
    jobs = [job(index) for index in range(10, 14)]
    outcome, _ = analyze(
        tmp_path,
        jobs,
        FakeGitHub(logs={item["id"]: "AssertionError: fail" for item in jobs}),
        client,
        max_jobs=2,
    )
    assert set(outcome.reasons) == {10, 11, 12, 13}
    assert outcome.reasons[12] == ANALYZER.UNAVAILABLE_REASON
    assert outcome.reasons[13] == ANALYZER.UNAVAILABLE_REASON
    assert outcome.omitted_count == 2
    assert [item["job_id"] for item in json.loads(client.responses.calls[0]["input"])["jobs"]] == [10, 11]


def test_unauthorized_run_never_touches_github_or_openai(tmp_path):
    client = FakeClient(response=valid_response)
    gh = FakeGitHub(logs={10: "AssertionError"})
    emitted = []
    outcome = ANALYZER.analyze_failures(
        run=run(repository={"id": 999, "full_name": "attacker/fork"}),
        jobs=[job()],
        repo="radixark/miles",
        gh=gh,
        policy_path=write_policy(tmp_path),
        prompt_path=PROMPT_PATH,
        schema_path=SCHEMA_PATH,
        client_factory=lambda timeout: client,
        emit=emitted.append,
    )
    assert outcome.unavailable
    assert gh.calls == [] and client.responses.calls == []


def test_oidc_provider_rejects_unapproved_host_before_sending_token(monkeypatch):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://evil.invalid/oidc")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    assert type(caught.value.__cause__).__name__ == "RuntimeError"
    assert "evil.invalid" not in str(caught.value)
    assert "sensitive-request-token" not in str(caught.value)


def test_oidc_callback_transport_failure_has_stable_safe_classification(monkeypatch):
    secret = "oidc-provider-secret-message"
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")

    class FailingOpener:
        def open(self, request, timeout):
            raise urllib.error.URLError(secret)

    monkeypatch.setattr(ANALYZER.urllib.request, "build_opener", lambda *args: FailingOpener())
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    assert type(caught.value.__cause__).__name__ == "URLError"
    assert ANALYZER._analysis_error_audit(caught.value) == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "cause_type": "URLError",
    }
    assert secret not in json.dumps(ANALYZER._analysis_error_audit(caught.value))


def test_wrapped_oidc_callback_failure_is_identified_without_emitting_messages(tmp_path):
    provider_error = with_cause(
        ANALYZER.GitHubOIDCProviderError("provider-secret-message"),
        urllib.error.URLError("oidc-cause-secret-message"),
    )
    error = with_cause(
        APIConnectionError("outer-secret-message", "https://api.openai.com/v1/responses?secret=query-secret"),
        provider_error,
    )
    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=error),
    )

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable and outcome.reasons == {}
    assert audit["stage"] == "github_oidc"
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "GitHubOIDCProviderError"
    assert not any(
        secret in emitted[0]
        for secret in (
            "provider-secret-message",
            "oidc-cause-secret-message",
            "outer-secret-message",
            "query-secret",
        )
    )


@pytest.mark.parametrize(
    ("cause_url", "expected_stage"),
    [
        ("https://auth.openai.com/oauth/token?secret=wif-query-secret", "openai_wif_exchange"),
        ("https://api.openai.com/v1/responses?secret=api-query-secret", "openai_api_transport"),
    ],
)
def test_downstream_api_connection_error_classifies_cause_without_emitting_details(
    tmp_path, cause_url, expected_stage
):
    error = with_cause(
        APIConnectionError("outer-transport-secret", "https://api.openai.com/v1/responses"),
        ConnectError("cause-transport-secret", cause_url),
    )
    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=error),
    )

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable and outcome.reasons == {}
    assert audit["stage"] == expected_stage
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "ConnectError"
    assert "outer-transport-secret" not in emitted[0]
    assert "cause-transport-secret" not in emitted[0]
    assert "wif-query-secret" not in emitted[0]
    assert "api-query-secret" not in emitted[0]


def test_unknown_exception_class_and_message_are_not_emitted(tmp_path):
    class ArbitrarySecretFailure(Exception):
        pass

    outcome, emitted = analyze(
        tmp_path,
        [job()],
        FakeGitHub(logs={10: "AssertionError: fail"}),
        FakeClient(error=ArbitrarySecretFailure("arbitrary-secret-message")),
    )
    audit = json.loads(emitted[0].split("=", 1)[1])
    assert outcome.unavailable
    assert audit["stage"] == "openai_api"
    assert audit["error_type"] == "OtherError"
    assert "ArbitrarySecretFailure" not in emitted[0]
    assert "arbitrary-secret-message" not in emitted[0]
