import base64
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT_DIR = ROOT / ".github/workflows/scripts"
HANDLER_PATH = SCRIPT_DIR / "lark_notify.py"
WORKFLOW_PATH = ROOT / ".github/workflows/ci-lark-notify.yml"
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HANDLER = load_module("lark_notify_test", HANDLER_PATH)
ANALYSIS = HANDLER.analyze_failures.__globals__["JobAnalysis"]


def run(**overrides):
    value = {
        "id": 123,
        "name": "PR Test",
        "event": "schedule",
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
        "created_at": "2026-09-04T15:00:00Z",
        "run_started_at": "2026-09-04T15:01:00Z",
        "updated_at": "2026-09-04T15:03:05Z",
        "head_sha": "a" * 40,
        "head_commit": {"message": "Test commit\nbody"},
        "html_url": "https://github.com/radixark/miles/actions/runs/123",
        "repository": {"id": 1072725553, "full_name": "radixark/miles"},
    }
    value.update(overrides)
    return value


def job(job_id=10, name="unit", conclusion="failure"):
    return {"id": job_id, "name": name, "conclusion": conclusion, "html_url": f"https://example/jobs/{job_id}"}


def markdown(card):
    return "\n".join(
        element.get("content", "") for element in card["card"]["body"]["elements"] if element.get("tag") == "markdown"
    )


def test_analysis_disabled_preserves_exact_original_card():
    jobs = [job(), job(20, "pass", "success")]
    original = HANDLER.render_ci_status(run(), jobs, None)
    disabled = HANDLER.AnalysisOutcome(enabled=False, reasons={})
    assert HANDLER.render_ci_status(run(), jobs, None, disabled) == original


def test_validated_reason_is_directly_beneath_its_existing_job_link():
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={
            10: [
                ANALYSIS(
                    reason="The assertion expected 4 but received 3.",
                    tags=("megatron", "lora"),
                    test_name="tests/fast/test_thing.py",
                    related_pull_request=2754,
                )
            ]
        },
    )
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert (
        "- [unit](https://example/jobs/10)\n"
        "  ↳ [megatron][lora] `tests/fast/test_thing.py`\n"
        "  ↳ The assertion expected 4 but received 3.\n"
        "  ↳ related to [PR #2754](https://github.com/radixark/miles/pull/2754)" in content
    )
    assert content.count("↳") == 3


def shard(index):
    return {
        "id": index,
        "name": f"stage-a-cpu ({index}) / run-cpu",
        "html_url": f"https://example/jobs/{index}",
        "conclusion": "failure",
    }


def test_one_defect_across_every_shard_collapses_to_a_single_row():
    same = ANALYSIS(reason="Collection failed because the hardware list was rejected.", tags=("rollout",))
    content = HANDLER.list_jobs_md([shard(i) for i in range(5)], reasons={i: [same] for i in range(5)})
    assert content.count("↳") == 2, content
    assert content.count("stage-a-cpu") == 5
    assert content.count("\n- ") == 0


def test_a_different_failure_keeps_its_own_row():
    same = ANALYSIS(reason="Collection failed because the hardware list was rejected.")
    other = ANALYSIS(reason="The deterministic test exited with code 1.")
    content = HANDLER.list_jobs_md([shard(0), shard(1), shard(2)], reasons={0: [same], 1: [same], 2: [other]})
    assert content.count("\n- ") == 1, content
    assert "The deterministic test exited with code 1." in content


def test_rows_stay_separate_when_no_analysis_is_available():
    content = HANDLER.list_jobs_md([shard(0), shard(1), shard(2)], reasons=None)
    assert content.count("\n- ") == 2, content


def test_rerun_reasons_apply_only_to_current_failures():
    current = [job(20, "still"), job(30, "new")]
    previous = {"fixed": job(10, "fixed"), "still": job(19, "still")}
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={
            10: [ANALYSIS(reason="Wrong old reason.")],
            20: [ANALYSIS(reason="The same assertion still fails.")],
            30: [ANALYSIS(reason="A new timeout occurred.")],
        },
    )
    content = markdown(HANDLER.render_ci_status(run(run_attempt=2), current, previous, outcome))
    assert "Fixed by rerun" in content and "Wrong old reason." not in content
    assert content.count("↳") == 2


def test_model_failure_adds_one_note_without_removing_original_rows():
    outcome = HANDLER.AnalysisOutcome(enabled=True, reasons={}, unavailable=True)
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert "[unit](https://example/jobs/10)" in content
    assert content.count("AI analysis unavailable") == 1
    assert "↳" not in content


def test_per_job_missing_log_reason_and_omitted_footer_render_compactly():
    outcome = HANDLER.AnalysisOutcome(
        enabled=True,
        reasons={10: [ANALYSIS(reason=HANDLER.analyze_failures.__globals__["UNAVAILABLE_REASON"])]},
        omitted_count=2,
    )
    content = markdown(HANDLER.render_ci_status(run(), [job()], None, outcome))
    assert "↳ unavailable" in content
    assert "AI analysis omitted for 2 additional failed jobs" in content


class FakeGitHub:
    def __init__(self, current_run, jobs, previous=None):
        self.current_run = current_run
        self.jobs = jobs
        self.previous = previous or []
        self.calls = []

    def run(self, run_id):
        self.calls.append(("run", run_id))
        return self.current_run

    def run_jobs(self, run_id):
        self.calls.append(("run_jobs", run_id))
        return self.jobs

    def run_attempt_jobs(self, run_id, attempt):
        self.calls.append(("run_attempt_jobs", run_id, attempt))
        return self.previous


def args(**overrides):
    value = {
        "run_id": 123,
        "any_event": False,
        "webhook": "https://lark.invalid",
        "dry_run": False,
        "repo": "radixark/miles",
    }
    value.update(overrides)
    return type("Args", (), value)()


def test_green_card_makes_no_analysis_call_and_posts_once(monkeypatch):
    calls = []
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: calls.append(kwargs))
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(conclusion="success"), [job(conclusion="success")]))
    assert calls == [] and len(posted) == 1
    assert posted[0]["card"]["header"]["title"]["content"].endswith("PASSED (1 job)")


def test_failed_rerun_analyzes_only_still_and_new_displayed_jobs(monkeypatch):
    captured = []
    monkeypatch.setenv("CI_FAILURE_ANALYSIS_GITHUB_TOKEN", "app-token")
    monkeypatch.setattr(
        HANDLER,
        "analyze_failures",
        lambda **kwargs: captured.append(kwargs) or HANDLER.AnalysisOutcome(enabled=False, reasons={}),
    )
    monkeypatch.setattr(HANDLER, "post_card", lambda *unused: None)
    gh = FakeGitHub(
        run(run_attempt=2),
        [job(20, "still"), job(30, "new")],
        [job(10, "fixed"), job(19, "still")],
    )
    HANDLER.cmd_ci_status(args(), gh)
    assert [item["id"] for item in captured[0]["jobs"]] == [20, 30]
    assert captured[0]["gh"].token == "app-token"


def test_unexpected_analyzer_exception_still_posts_original_card_once(monkeypatch):
    monkeypatch.setattr(HANDLER, "analyze_failures", lambda **kwargs: (_ for _ in ()).throw(TypeError("bad")))
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda card, webhook, dry_run: posted.append(card))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(), [job()]))
    assert len(posted) == 1
    content = markdown(posted[0])
    assert "[unit](https://example/jobs/10)" in content
    assert content.count("AI analysis unavailable") == 1


def test_rerun_caps_current_failed_rows_across_still_and_new_sections():
    previous = {f"still-{index}": job(index, f"still-{index}") for index in range(1, 11)}
    current = [job(index + 100, f"still-{index}") for index in range(1, 11)] + [
        job(index + 200, f"new-{index}") for index in range(1, 11)
    ]
    content = markdown(HANDLER.render_ci_status(run(run_attempt=2), current, previous))
    assert content.count("https://example/jobs/") == 15
    assert "... and 5 more" in content


def test_non_schedule_and_incomplete_runs_still_skip_before_delivery(monkeypatch, capsys):
    posted = []
    monkeypatch.setattr(HANDLER, "post_card", lambda *values: posted.append(values))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(event="pull_request"), []))
    HANDLER.cmd_ci_status(args(), FakeGitHub(run(status="in_progress"), []))
    assert posted == []
    assert "not a scheduled run" in capsys.readouterr().out


def test_dry_run_prints_the_final_card_without_webhook_io(capsys):
    card = HANDLER.render_ci_status(run(), [job()], None)
    HANDLER.post_card(card, "https://lark.invalid", True)
    assert json.loads(capsys.readouterr().out) == card


def test_job_log_rejects_non_numeric_or_non_positive_ids():
    gh = HANDLER.GitHub("token", "radixark/miles")
    for value in (True, "10", 0, -1):
        try:
            gh.job_log(value, 100)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe job id {value!r}")


def test_file_content_accepts_github_line_wrapped_base64(monkeypatch):
    gh = HANDLER.GitHub("token", "radixark/miles")
    encoded = base64.b64encode(b"line one\nline two\n").decode()
    wrapped = encoded[:8] + "\n" + encoded[8:]
    monkeypatch.setattr(gh, "get", lambda path, params: {"encoding": "base64", "content": wrapped})
    assert gh.file_content("tests/example.py", "a" * 40, 100) == "line one\nline two\n"


class FakeBlob:
    """A redirect target that serves body, honouring an explicit byte range like Azure blob storage."""

    def __init__(self, body, *, honour_range=True, announce_length=True):
        self.body = body
        self.honour_range = honour_range
        self.announce_length = announce_length
        self.requests = []

    def open(self, target, timeout):
        url = target if isinstance(target, str) else target.full_url
        header = None if isinstance(target, str) else target.get_header("Range")
        self.requests.append((url, header))
        served = self.body
        if header and self.honour_range:
            start, end = header.removeprefix("bytes=").split("-")
            served = self.body[int(start) : int(end) + 1]
        headers = {"Content-Length": str(len(self.body))} if self.announce_length else {}
        return FakeBlobResponse(served, headers)


class FakeBlobResponse:
    def __init__(self, body, headers):
        self.body = body
        self.headers = headers
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def read(self, size):
        chunk = self.body[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def redirecting_github(monkeypatch, blob, token="dedicated-app-token"):
    first_requests = []

    class Opener:
        def open(self, request, timeout):
            first_requests.append(request)
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Found",
                {"Location": "https://ci-results.blob.core.windows.net/job/10"},
                None,
            )

    monkeypatch.setattr(HANDLER.urllib.request, "build_opener", lambda *unused: Opener())
    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", blob.open)
    return HANDLER.GitHub(token, "radixark/miles"), first_requests


def test_job_log_redirect_drops_app_token_and_bounds_the_download(monkeypatch):
    blob = FakeBlob(b"0123456789extra")
    gh, first_requests = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 10) == "56789extra"
    assert first_requests[0].get_header("Authorization") == "Bearer dedicated-app-token"
    assert [url for url, _ in blob.requests] == ["https://ci-results.blob.core.windows.net/job/10"] * 2
    assert blob.requests[1][1] == "bytes=5-14"
    assert all("Authorization" not in str(url) for url, _ in blob.requests)


def test_job_log_keeps_the_failure_at_the_end_of_an_oversized_log(monkeypatch):
    body = b"setup noise\n" * 4000 + b"##[error]Process completed with exit code 1.\n"
    gh, _ = redirecting_github(monkeypatch, FakeBlob(body))
    assert "##[error]Process completed with exit code 1." in gh.job_log(10, 200)


def test_job_log_keeps_the_tail_when_the_backend_ignores_the_range(monkeypatch):
    body = b"noise\n" * 100 + b"final line\n"
    blob = FakeBlob(body, honour_range=False)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 11).endswith("final line\n")


def test_job_log_streams_the_tail_when_the_backend_reports_no_length(monkeypatch):
    body = b"noise\n" * 100 + b"final line\n"
    blob = FakeBlob(body, announce_length=False)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 11) == "final line\n"
    assert len(blob.requests) == 1


def test_job_log_download_is_bounded_when_the_body_never_ends(monkeypatch):
    class EndlessResponse(FakeBlobResponse):
        def read(self, size):
            return b"x" * size

    class EndlessBlob(FakeBlob):
        def open(self, target, timeout):
            self.requests.append(target)
            return EndlessResponse(b"", {})

    blob = EndlessBlob(b"")
    monkeypatch.setattr(HANDLER, "MAX_LOG_STREAM_BYTES", 1 << 20)
    gh, _ = redirecting_github(monkeypatch, blob)
    assert gh.job_log(10, 32) == "x" * 32


def test_every_configuration_file_the_analyzer_reads_is_checked_out():
    analyzer = HANDLER.analyze_failures.__globals__
    workflow = (SCRIPT_DIR.parents[0] / "ci-lark-notify.yml").read_text()
    repo_root = SCRIPT_DIR.parents[2]
    for name in ("DEFAULT_POLICY_PATH", "DEFAULT_SCHEMA_PATH", "DEFAULT_PROMPT_PATH", "DEFAULT_TAGS_PATH"):
        relative = analyzer[name].relative_to(repo_root).as_posix()
        assert f"\n            {relative}\n" in workflow, f"{relative} is missing from sparse-checkout"


def test_notifier_workflow_has_pinned_read_only_identity_boundaries():
    workflow = WORKFLOW_PATH.read_text()
    assert "workflow_run:" in workflow and 'workflows: ["PR Test"]' in workflow
    assert "github.repository == 'radixark/miles'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "id-token: write" in workflow
    assert "permission-actions: read" in workflow
    assert "permission-contents: read" in workflow
    assert "permission-pull-requests: read" in workflow
    assert "permission-issues" not in workflow and "permission-actions: write" not in workflow
    assert "CI_FAILURE_ANALYSIS_APP_CLIENT_ID" in workflow
    assert "CI_COMMAND_APP" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow
    assert "persist-credentials: false" in workflow
    assert "comment_ci_command.py" not in workflow
    dispatch_inputs = workflow.split("workflow_dispatch:", 1)[1].split("env:", 1)[0]
    assert "prompt" not in dispatch_inputs and "model" not in dispatch_inputs and "repository" not in dispatch_inputs


def test_policy_prompt_and_schema_are_git_versioned_and_strict():
    policy = json.loads((ROOT / ".github/workflows/policies/ci-failure-analysis.json").read_text())
    schema = json.loads((ROOT / ".github/workflows/policies/ci-failure-response-schema.json").read_text())
    prompt = (ROOT / ".github/workflows/prompts/ci-failure-analysis.md").read_text()
    assert policy["enabled"] is True and policy["max_model_calls"] == 1
    assert schema["additionalProperties"] is False
    assert schema["properties"]["analyses"]["items"]["additionalProperties"] is False
    assert "untrusted" in prompt and "exactly one factual sentence" in prompt
