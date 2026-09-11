import hashlib
import importlib.util
import io
import json
import re
import ssl
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

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
# Split so the repository's detect-private-key hook does not flag this redaction fixture.
PEM_HEADER = "-----BEGIN " + "PRIVATE KEY-----"
PEM_FOOTER = "-----END " + "PRIVATE KEY-----"


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


class SchemaError(Exception):
    pass


class OIDCResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class CapturingOIDCOpener:
    def __init__(self, response=b'{"value":"header.payload.signature"}', error=None):
        self.response = response
        self.error = error
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return OIDCResponse(self.response)


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
                "tags": [],
                "test_name": None,
                "reason": "The assertion failed because the returned value was 3 instead of 4.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": [item["evidence_refs"][0]],
                "related_pull_request": None,
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


def test_checked_in_policy_is_valid_and_enabled():
    policy = ANALYZER.load_policy(POLICY_PATH)
    assert policy.enabled is True
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
        f"{PEM_HEADER}\nprivate-material\n{PEM_FOOTER}\n"
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


def pytest_collection_log(noise_lines=22):
    """A pytest import failure: the decisive line trails its marker by a long importlib traceback."""
    return "\n".join(
        [
            *(f"setup line {index}" for index in range(60)),
            "_ ERROR collecting tests/fast/ray/test_layout.py _",
            "ImportError while importing test module '/w/tests/fast/ray/test_layout.py'.",
            "Traceback:",
            *("<frozen importlib._bootstrap>:1204: in _gcd_import" for _ in range(noise_lines)),
            "E   ModuleNotFoundError: No module named 'miles.backends.update_weight'",
            *(f"warning {index}" for index in range(40)),
            "=================== 1 error in 4.35s ===================",
            "##[error]Process completed with exit code 1.",
        ]
    )


def test_evidence_reaches_the_decisive_line_past_a_long_traceback():
    evidence = ANALYZER.extract_log_evidence(pytest_collection_log(), 10, 8400)
    assert "E   ModuleNotFoundError: No module named 'miles.backends.update_weight'" in evidence["text"]


def test_evidence_spends_the_budget_it_is_given():
    text = pytest_collection_log()
    small = ANALYZER.extract_log_evidence(text, 10, 900)
    large = ANALYZER.extract_log_evidence(text, 10, 8400)
    assert len(small["text"]) <= 900 < len(large["text"]) <= 8400
    assert large["text"].count("\n") > small["text"].count("\n")


def test_evidence_never_exceeds_a_budget_smaller_than_the_seed_window():
    evidence = ANALYZER.extract_log_evidence(pytest_collection_log(), 10, 300)
    assert len(evidence["text"]) <= 300


def test_evidence_stops_growing_at_the_edges_of_a_short_log():
    text = "\n".join(["AssertionError: boom", "second line"])
    evidence = ANALYZER.extract_log_evidence(text, 10, 8400)
    assert evidence["text"].splitlines()[1:] == ["AssertionError: boom", "second line"]


GROUNDING = (
    "E   ModuleNotFoundError: No module named 'miles.backends.megatron_utils.update_weight'\n"
    "tests/fast/ray/test_layout.py:10: in <module>\n"
    "Commits touching miles/backends: refactor(update-weight): move the protocols (#2754)\n"
)


def grounded_item(**overrides):
    item = {
        "job_id": 10,
        "tags": ["weight-update", "megatron"],
        "test_name": "tests/fast/ray/test_layout.py",
        "reason": "Collection failed because the update_weight module no longer exists.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": 2754,
    }
    item.update(overrides)
    return item


def validate_grounded(**overrides):
    return ANALYZER.validate_response(
        json.dumps({"schema_version": "1", "analyses": [grounded_item(**overrides)]}),
        [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
        280,
        ANALYZER.load_tags(),
        {10: GROUNDING},
    )


def test_tag_vocabulary_is_the_only_source_of_the_schema_enum():
    tags = ANALYZER.load_tags()
    schema = ANALYZER.load_schema(SCHEMA_PATH, tags)
    assert schema["properties"]["analyses"]["items"]["properties"]["tags"]["items"]["enum"] == tags
    assert tags == sorted(set(tags)) and "weight-update" in tags


def test_grounded_analysis_keeps_tags_test_name_and_cause_pull_request():
    analysis = validate_grounded()[10][0]
    assert analysis.tags == ("weight-update", "megatron")
    assert analysis.test_name == "tests/fast/ray/test_layout.py"
    assert analysis.related_pull_request == 2754


@pytest.mark.parametrize(
    "overrides",
    [
        {"reason": "Collection failed because the module is gone [job:10:log:1-2]."},
        {"reason": "no terminal punctuation"},
        {"category": "remediation"},
        {"evidence_refs": ["job:other:log:1"]},
    ],
)
def test_a_broken_core_contract_still_rejects_the_whole_response(overrides):
    with pytest.raises(ValueError):
        validate_grounded(**overrides)


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        ({"tags": ["deepseek-v9"]}, "tags", ()),
        ({"tags": ["inkling"]}, "tags", ()),
        ({"tags": ["multi-lora"]}, "tags", ()),
        ({"tags": ["megatron", "megatron"]}, "tags", ()),
        ({"tags": ["megatron", "lora", "fsdp"]}, "tags", ()),
        ({"test_name": "tests/fabricated/test_nope.py"}, "test_name", None),
        ({"test_name": "tests/fast/ray/test_layout.py; rm -rf /"}, "test_name", None),
        ({"related_pull_request": 9999}, "related_pull_request", None),
        ({"related_pull_request": -1}, "related_pull_request", None),
        ({"related_pull_request": True}, "related_pull_request", None),
    ],
)
def test_an_ungrounded_decoration_drops_itself_and_keeps_the_row(overrides, field, expected):
    analysis = validate_grounded(**overrides)[10][0]
    assert getattr(analysis, field) == expected
    assert analysis.reason.startswith("Collection failed")


def test_a_tag_needs_its_whole_phrase_not_scattered_words():
    vocabulary = ANALYZER.load_tags()
    scattered = "lora adapters across multi node runs in megatron_utils"
    with pytest.raises(ValueError):
        ANALYZER._validate_tags(["multi-lora"], vocabulary, scattered)
    assert ANALYZER._validate_tags(["megatron"], vocabulary, scattered) == ("megatron",)
    assert ANALYZER._validate_tags(["weight-update"], vocabulary, "update_weight_from_distributed") == (
        "weight-update",
    )


def test_prompt_states_the_reason_limit_the_policy_enforces():
    prompt, _ = ANALYZER.load_prompt(PROMPT_PATH)
    limit = ANALYZER.load_policy(POLICY_PATH).max_reason_chars
    assert f"at most {limit} characters" in prompt


def test_audit_names_our_own_validation_rule_but_never_foreign_text():
    ours = ANALYZER._analysis_error_audit(ValueError("tag is not grounded in the evidence"))
    assert ours["validation_reason"] == "tag is not grounded in the evidence"
    assert "validation_reason" not in ANALYZER._analysis_error_audit(ValueError("leaked secret value"))


def test_every_validation_message_is_declared_safe_for_the_audit():
    source = ANALYZER_PATH.read_text()
    raised = set(re.findall(r'raise ValueError\("([^"]+)"\)', source))
    assert raised and raised <= ANALYZER.SAFE_VALIDATION_REASONS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("__w/miles/miles/miles/utils/test_utils/runner.py", "miles/utils/test_utils/runner.py"),
        ("__w/miles/miles/tests/e2e/sglang/test_qwen36.py", "tests/e2e/sglang/test_qwen36.py"),
        ("home/runner/work/miles/miles/tests/fast/ray/test_layout.py", "tests/fast/ray/test_layout.py"),
        ("miles/utils/misc.py", "miles/utils/misc.py"),
        ("opt/hostedtoolcache/Python/3.11/site-packages/_pytest/python.py", None),
    ],
)
def test_a_runner_path_resolves_past_the_repeated_repository_name(raw, expected):
    assert ANALYZER._safe_path(raw) == expected


SUITE_SUMMARY = """2026-09-09T15:00:00.0Z FAILED:
2026-09-09T15:00:00.0Z   tests/e2e/a/test_one.py (exit code 1)
2026-09-09T15:00:00.0Z   tests/e2e/b/test_two.py (exit code 1)
2026-09-09T15:00:00.0Z ============================================================
"""


def two_failure_jobs():
    return [
        {
            "job_id": 10,
            "evidence_refs": ["job:10:log:1-2"],
            "failing_tests": ["tests/e2e/a/test_one.py", "tests/e2e/b/test_two.py"],
        }
    ]


def analysis_item(test_name, **overrides):
    item = {
        "job_id": 10,
        "tags": [],
        "test_name": test_name,
        "reason": "Collection failed because the module is gone.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": None,
    }
    item.update(overrides)
    return item


def validate_two(items):
    return ANALYZER.validate_response(
        json.dumps({"schema_version": "1", "analyses": items}),
        two_failure_jobs(),
        280,
        ANALYZER.load_tags(),
        {10: GROUNDING},
    )


def failure_block(test_name, body_lines, stamp="2026-09-11T16:56:05.4909230Z "):
    lines = [f"{stamp}Last output of {test_name}:"]
    lines += [f"{stamp}  | {line}" for line in body_lines]
    return "\n".join(lines)


def two_block_log():
    return "\n".join(
        [
            failure_block("tests/e2e/a/test_one.py", [f"noise {i}" for i in range(40)] + ["AssertionError: one"]),
            failure_block("tests/e2e/b/test_two.py", [f"noise {i}" for i in range(40)] + ["AssertionError: two"]),
            "2026-09-11T16:56:05.4909230Z FAILED:",
            "2026-09-11T16:56:05.4909230Z   tests/e2e/a/test_one.py (exit code 1)",
            "2026-09-11T16:56:05.4909230Z   tests/e2e/b/test_two.py (exit code 1)",
            "2026-09-11T16:56:05.4909230Z ============================================================",
        ]
    )


def test_log_evidence_drops_the_timestamp_every_line_carries():
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 40_000)
    assert "2026-09-11T16:56:05" not in evidence["text"]
    assert "AssertionError: one" in evidence["text"]


def test_every_failure_block_survives_whole_when_the_budget_allows():
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 40_000)
    assert evidence["text"].count("Last output of") == 2
    assert "AssertionError: one" in evidence["text"] and "AssertionError: two" in evidence["text"]


def test_a_budget_too_small_for_the_blocks_falls_back_to_marker_windows():
    evidence = ANALYZER.extract_log_evidence(two_block_log(), 10, 600)
    assert len(evidence["text"]) <= 600
    assert evidence["text"].strip()


def test_the_suite_summary_names_every_failing_test_in_a_job():
    assert ANALYZER.extract_failed_tests(SUITE_SUMMARY, 5) == [
        "tests/e2e/a/test_one.py",
        "tests/e2e/b/test_two.py",
    ]


def test_a_job_with_two_failures_keeps_both_analyses():
    analyses = validate_two([analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/b/test_two.py")])
    assert [a.test_name for a in analyses[10]] == ["tests/e2e/a/test_one.py", "tests/e2e/b/test_two.py"]


@pytest.mark.parametrize(
    "items",
    [
        [analysis_item("tests/e2e/a/test_one.py")],
        [analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/a/test_one.py")],
        [analysis_item("tests/e2e/a/test_one.py"), analysis_item("tests/e2e/c/test_invented.py")],
    ],
)
def test_a_response_must_cover_exactly_the_tests_the_suite_named(items):
    with pytest.raises(ValueError):
        validate_two(items)


def test_missing_module_paths_cover_deleted_packages_and_module_files():
    text = (
        "ModuleNotFoundError: No module named 'miles.backends.megatron_utils.update_weight'\n"
        "ImportError: cannot import name 'exec_command' from 'miles.utils.misc'\n"
        "No module named 'torch_memory_saver'\n"
    )
    paths = ANALYZER.extract_missing_module_paths(text)
    assert paths[0] == "miles/backends/megatron_utils/update_weight"
    assert "miles/utils/misc.py" in paths and paths.index("miles/utils/misc.py") < paths.index("miles/utils/misc")
    assert not any("torch_memory_saver" in path for path in paths)


def test_absent_test_name_and_cause_pull_request_are_allowed():
    analysis = validate_grounded(tags=[], test_name=None, related_pull_request=None)[10][0]
    assert analysis.tags == () and analysis.test_name is None and analysis.related_pull_request is None


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
                "tags": [],
                "test_name": None,
                "reason": "The assertion failed because the result was empty.",
                "category": "test_failure",
                "confidence": "high",
                "evidence_refs": ["job:10:log:1-2"],
                "related_pull_request": None,
            }
        ],
    }
    analyses = ANALYZER.validate_response(json.dumps(raw), jobs, 280, ANALYZER.load_tags(), {10: ""})
    assert analyses[10][0].reason.startswith("The assertion")
    assert analyses[10][0].tags == () and analyses[10][0].related_pull_request is None


def test_strict_response_schema_uses_only_supported_structured_output_keywords():
    schema = ANALYZER.load_schema(SCHEMA_PATH)
    supported = {
        "$id",
        "additionalProperties",
        "const",
        "enum",
        "items",
        "maxItems",
        "maxLength",
        "minItems",
        "minLength",
        "properties",
        "required",
        "type",
    }

    def schema_keywords(node):
        if not isinstance(node, dict):
            return set()
        result = set(node)
        for key, value in node.items():
            if key == "properties":
                for child in value.values():
                    result.update(schema_keywords(child))
            elif key == "items":
                result.update(schema_keywords(value))
        return result

    keywords = schema_keywords(schema)
    assert keywords <= supported
    assert "uniqueItems" not in keywords


@pytest.mark.parametrize(
    "refs",
    [
        [],
        ["job:10:log:1-2", "job:10:log:1-2"],
        ["job:other:log:1-2"],
        ["job:10:log:1-2", 123],
    ],
)
def test_local_validation_rejects_duplicate_empty_or_invalid_evidence_refs(refs):
    item = {
        "job_id": 10,
        "tags": [],
        "test_name": None,
        "reason": "The assertion failed because the result was empty.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": refs,
        "related_pull_request": None,
    }
    with pytest.raises(ValueError):
        ANALYZER.validate_response(
            json.dumps({"schema_version": "1", "analyses": [item]}),
            [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
            280,
            ANALYZER.load_tags(),
            {10: ""},
        )


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
        "tags": [],
        "test_name": None,
        "reason": "The assertion failed because the result was empty.",
        "category": "test_failure",
        "confidence": "high",
        "evidence_refs": ["job:10:log:1-2"],
        "related_pull_request": None,
    }
    mutation(item)
    with pytest.raises(ValueError):
        ANALYZER.validate_response(
            json.dumps({"schema_version": "1", "analyses": [item]}),
            [{"job_id": 10, "evidence_refs": ["job:10:log:1-2"]}],
            280,
            ANALYZER.load_tags(),
            {10: ""},
        )


def test_disabled_policy_makes_no_github_or_model_call(tmp_path):
    client = FakeClient(response=valid_response)
    gh = FakeGitHub(logs={10: "AssertionError: no"})
    outcome, _ = analyze(tmp_path, [job()], gh, client, enabled=False)
    assert outcome == ANALYZER.AnalysisOutcome(enabled=False, reasons={})
    assert gh.calls == [] and client.responses.calls == []


def test_missing_analysis_app_token_preserves_base_card_contract(tmp_path):
    outcome, emitted = analyze(tmp_path, [job()], None, FakeClient(response=valid_response))
    assert outcome.unavailable and outcome.reasons == {}
    assert '"stage":"github_auth"' in emitted[0]


def test_all_missing_logs_get_per_row_fallback_without_model_call(tmp_path):
    client = FakeClient(response=valid_response)
    outcome, emitted = analyze(tmp_path, [job(10), job(11)], FakeGitHub(), client)
    assert {key: value[0].reason for key, value in outcome.reasons.items()} == {
        10: ANALYZER.UNAVAILABLE_REASON,
        11: ANALYZER.UNAVAILABLE_REASON,
    }
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
    assert outcome.reasons[12][0].reason == ANALYZER.UNAVAILABLE_REASON
    assert outcome.reasons[13][0].reason == ANALYZER.UNAVAILABLE_REASON
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
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(AssertionError("request must not be sent")),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    assert type(caught.value.__cause__).__name__ == "RuntimeError"
    assert ANALYZER._analysis_error_audit(caught.value)["oidc_failure_reason"] == "endpoint_validation"
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
        "root_cause_type": "URLError",
        "oidc_failure_reason": "request_error",
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
    assert audit["root_cause_type"] == "URLError"
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
    assert audit["root_cause_type"] == "ConnectError"
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
    assert audit["root_cause_type"] == "OtherError"
    assert "ArbitrarySecretFailure" not in emitted[0]
    assert "arbitrary-secret-message" not in emitted[0]


def test_nested_oidc_http_error_reports_only_allowlisted_root_and_integer_status():
    root = urllib.error.HTTPError(
        "https://pipelines.actions.githubusercontent.com/oidc?jwt=secret-jwt",
        403,
        "sensitive-http-reason",
        {"Authorization": "Bearer sensitive-header-token"},
        io.BytesIO(b"sensitive-response-body"),
    )
    error = with_cause(
        APIConnectionError("outer-sensitive-message", "https://api.openai.com/v1/responses"),
        with_cause(ANALYZER.GitHubOIDCProviderError("provider-sensitive-message"), root),
    )
    emitted = []
    ANALYZER._audit(ANALYZER._analysis_error_audit(error), emitted.append)

    audit = json.loads(emitted[0].split("=", 1)[1])
    assert audit == {
        "stage": "github_oidc",
        "error_type": "APIConnectionError",
        "cause_type": "GitHubOIDCProviderError",
        "root_cause_type": "HTTPError",
        "root_http_status": 403,
    }
    assert isinstance(audit["root_http_status"], int)
    assert not any(
        secret in emitted[0]
        for secret in (
            "secret-jwt",
            "sensitive-http-reason",
            "sensitive-header-token",
            "sensitive-response-body",
            "outer-sensitive-message",
            "provider-sensitive-message",
        )
    )


@pytest.mark.parametrize(
    ("root", "expected_type"),
    [
        (urllib.error.URLError("sensitive-url-error"), "URLError"),
        (ssl.SSLError("sensitive-ssl-error"), "SSLError"),
        (json.JSONDecodeError("sensitive-json-error", "sensitive-json-document", 0), "JSONDecodeError"),
        (SchemaError("sensitive-schema-error"), "SchemaError"),
        (RuntimeError("sensitive-runtime-error"), "RuntimeError"),
    ],
)
def test_nested_root_types_are_allowlisted_without_messages(root, expected_type):
    error = with_cause(
        APIConnectionError("sensitive-outer-error", "https://api.openai.com/v1/responses?token=sensitive"),
        with_cause(ANALYZER.GitHubOIDCProviderError("sensitive-provider-error"), root),
    )
    audit = ANALYZER._analysis_error_audit(error)
    serialized = json.dumps(audit)

    assert audit["stage"] == "github_oidc"
    assert audit["error_type"] == "APIConnectionError"
    assert audit["cause_type"] == "GitHubOIDCProviderError"
    assert audit["root_cause_type"] == expected_type
    assert "root_http_status" not in audit
    assert "sensitive" not in serialized


def test_url_error_with_nested_ssl_reason_reports_ssl_as_root():
    ssl_error = ssl.SSLCertVerificationError("sensitive-certificate-message")
    url_error = urllib.error.URLError(ssl_error)
    error = with_cause(ANALYZER.GitHubOIDCProviderError("sensitive-provider-message"), url_error)

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "cause_type": "URLError",
        "root_cause_type": "SSLCertVerificationError",
    }
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize("payload", [b"{}", b'{"value":""}', b'{"other":"sensitive-token"}'])
def test_oidc_missing_token_value_has_fixed_reason_without_payload_leak(monkeypatch, payload):
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda request, timeout: OIDCResponse(payload)),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "missing_token_value"
    assert audit["root_cause_type"] == "RuntimeError"
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize("payload", [b"not-sensitive-json", b'["sensitive-token"]', b"\xff"])
def test_oidc_response_decode_has_fixed_reason_without_payload_leak(monkeypatch, payload):
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://pipelines.actions.githubusercontent.com/oidc?request_data=sensitive-query",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda request, timeout: OIDCResponse(payload)),
    )
    provider = ANALYZER._github_actions_oidc_provider("openai-audience")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()

    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "response_decode"
    assert audit["root_cause_type"] in {"JSONDecodeError", "ValueError", "OtherError"}
    assert "sensitive" not in json.dumps(audit)


def test_arbitrary_oidc_reason_attribute_cannot_spoof_audit():
    class AttributeSpoofError(RuntimeError):
        pass

    error = AttributeSpoofError("sensitive-message")
    error.oidc_failure_reason = "endpoint_validation"
    error.root_http_status = 418
    error.url = "https://evil.invalid/?token=sensitive-token"

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "openai_api",
        "error_type": "OtherError",
        "root_cause_type": "OtherError",
    }
    assert "sensitive" not in json.dumps(audit)


def test_oidc_marker_subclass_attribute_cannot_spoof_reason():
    class MarkerSpoofError(ANALYZER.GitHubOIDCProviderError):
        pass

    error = MarkerSpoofError("sensitive-message")
    error.oidc_failure_reason = "missing_token_value"

    audit = ANALYZER._analysis_error_audit(error)
    assert audit == {
        "stage": "github_oidc",
        "error_type": "GitHubOIDCProviderError",
        "root_cause_type": "GitHubOIDCProviderError",
    }
    assert "sensitive" not in json.dumps(audit)


@pytest.mark.parametrize(
    "url",
    [
        "https://pipelines.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://pipelines-ghubeus1.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://regional.runner.actions.githubusercontent.com/oidc/token?request_data=opaque",
        "https://PIPELINES.ACTIONS.GITHUBUSERCONTENT.COM:443/oidc/token?request_data=opaque",
    ],
)
def test_github_com_oidc_endpoint_accepts_strict_actions_subdomains(monkeypatch, url):
    opener = CapturingOIDCOpener()
    handlers = []
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", url)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: handlers.extend(args) or opener,
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    assert provider["get_token"]() == "header.payload.signature"
    assert len(opener.requests) == 1
    assert opener.requests[0][1] == 15
    assert any(isinstance(handler, ANALYZER._NoRedirect) for handler in handlers)


def test_oidc_endpoint_preserves_opaque_path_and_query_while_replacing_audience(monkeypatch):
    opener = CapturingOIDCOpener()
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://regional.actions.githubusercontent.com/opaque/%2Fpath?first=a%2Fb&repeat=1&repeat=2&audience=old&%61udience=older&flag",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(ANALYZER.urllib.request, "build_opener", lambda *args: opener)

    provider = ANALYZER._github_actions_oidc_provider("new audience/value")
    assert provider["get_token"]() == "header.payload.signature"
    requested_url = opener.requests[0][0].full_url
    assert requested_url == (
        "https://regional.actions.githubusercontent.com/opaque/%2Fpath?"
        "first=a%2Fb&repeat=1&repeat=2&flag&audience=new+audience%2Fvalue"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://actions.githubusercontent.com/oidc/token",
        "https://evilactions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com.evil.example/oidc/token",
        "https://actions.githubusercontent.com.evil.example/oidc/token",
        "https://pipelines.actions.githubusercontent.com./oidc/token",
        "https://127.0.0.1/oidc/token",
        "https://[::1]/oidc/token",
        "https://user@pipelines.actions.githubusercontent.com/oidc/token",
        "https://user:password@pipelines.actions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com:444/oidc/token",
        "https://pipelines.actions.githubusercontent.com:/oidc/token",
        "https://pipelines.actions.githubusercontent.com:notaport/oidc/token",
        "https://pipelines.actions.githubusercontent.com:65536/oidc/token",
        "https://pipelines.actions.githubusercontent.com/oidc/token#fragment",
        "https://pipelines.actions.githubusercontent.com/oidc/token#",
        "http://pipelines.actions.githubusercontent.com/oidc/token",
        "https://pipelines.actions.githubusercontent.com",
        "https://pipelines.actions.githubusercontent.com/",
        "https://pipelines.actions.githubusercontent.com//other/path",
        "https://pipelines.actions.githubusercontent.com/\\other/path",
        "https://pipelines.actions.githubusercontent.com/oidc path",
        "https://pipelines.actions.githubusercontent.com/oidc\tpath",
        "https://-regional.actions.githubusercontent.com/oidc/token",
        "https://regional-.actions.githubusercontent.com/oidc/token",
        "https://regional..actions.githubusercontent.com/oidc/token",
    ],
)
def test_github_com_oidc_endpoint_rejects_unsafe_or_malformed_urls_before_request(monkeypatch, url):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", url)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: (_ for _ in ()).throw(AssertionError("request must not be sent")),
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    audit = ANALYZER._analysis_error_audit(caught.value)
    assert audit["oidc_failure_reason"] == "endpoint_validation"
    assert url not in json.dumps(audit)


def test_oidc_provider_does_not_follow_redirects(monkeypatch):
    redirect = urllib.error.HTTPError(
        "https://regional.actions.githubusercontent.com/oidc/token",
        302,
        "sensitive-redirect-reason",
        {"Location": "https://evil.invalid/?token=sensitive-token"},
        None,
    )
    opener = CapturingOIDCOpener(error=redirect)
    handlers = []
    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://regional.actions.githubusercontent.com/oidc/token?request_data=opaque",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "sensitive-request-token")
    monkeypatch.setattr(
        ANALYZER.urllib.request,
        "build_opener",
        lambda *args: handlers.extend(args) or opener,
    )

    provider = ANALYZER._github_actions_oidc_provider("https://api.openai.com/v1")
    with pytest.raises(ANALYZER.GitHubOIDCProviderError) as caught:
        provider["get_token"]()
    audit = ANALYZER._analysis_error_audit(caught.value)
    assert len(opener.requests) == 1
    assert any(isinstance(handler, ANALYZER._NoRedirect) for handler in handlers)
    assert audit["oidc_failure_reason"] == "request_error"
    assert audit["root_http_status"] == 302
    assert "evil.invalid" not in json.dumps(audit)
    assert "sensitive" not in json.dumps(audit)
