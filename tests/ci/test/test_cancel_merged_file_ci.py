import importlib.util
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]
SCRIPT = ".github/workflows/scripts/cancel_merged_file_ci.py"
spec = importlib.util.spec_from_file_location("cancel_merged_file_ci", ROOT / SCRIPT)
HANDLER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HANDLER)
EVENT = {
    "action": "closed",
    "number": 42,
    "pull_request": {"merged": True},
    "repository": {"full_name": "radixark/miles"},
}
PREFIX = "repos/radixark/miles/actions"


def file_run(run_id, *, pull=42, status="queued", **overrides):
    return {
        "id": run_id,
        "path": ".github/workflows/run-ci-file.yml",
        "event": "workflow_dispatch",
        "status": status,
        "display_title": f"/rerun-test tests/e2e/test_{run_id}.py (PR #{pull})",
        **overrides,
    }


class FakeAPI:
    def __init__(self, runs_by_status=None, *, conflict_status=None, error_code=None):
        self.runs_by_status = runs_by_status or {}
        self.conflict_status = conflict_status
        self.error_code = error_code
        self.calls = []
        self.cancelled = []

    def __call__(self, path, *, method="GET"):
        self.calls.append((method, path))
        if method == "POST":
            assert path.startswith(f"{PREFIX}/runs/") and path.endswith("/cancel")
            if self.error_code:
                raise urllib.error.HTTPError(path, self.error_code, "cancel failed", {}, None)
            self.cancelled.append(int(path.split("/")[-2]))
            return None
        if "/runs?" not in path:
            assert path.startswith(f"{PREFIX}/runs/")
            return {"status": self.conflict_status}
        endpoint, query = path.split("?", 1)
        assert endpoint == f"{PREFIX}/workflows/run-ci-file.yml/runs"
        query = urllib.parse.parse_qs(query)
        assert query["event"] == ["workflow_dispatch"]
        assert query["per_page"] == ["100"]
        assert "head_sha" not in query
        runs = self.runs_by_status.get(query["status"][0], [])
        start = (int(query["page"][0]) - 1) * 100
        return {"total_count": len(runs), "workflow_runs": runs[start : start + 100]}


def test_cancels_all_active_states_and_files_only_for_the_merged_pr():
    statuses = ("requested", "waiting", "pending", "queued", "in_progress")
    api = FakeAPI({status: [file_run(i, status=status)] for i, status in enumerate(statuses, 1)})
    api.runs_by_status["queued"] += [
        file_run(90, pull=420),
        file_run(91, pull=4),
        file_run(92, status="completed"),
        file_run(93, path=".github/workflows/pr-test.yml"),
        file_run(94, event="pull_request"),
        file_run(95, display_title="unrelated (PR #42)"),
        file_run(96, display_title="/rerun-test tests/e2e/test_a.py (PR #42) extra"),
    ]
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == [1, 2, 3, 4, 5]
    assert api.cancelled == [1, 2, 3, 4, 5]


def test_collects_all_pages_before_cancelling_and_deduplicates_state_transitions():
    runs = [file_run(i) for i in range(1, 102)]
    api = FakeAPI({"pending": runs, "in_progress": [file_run(1, status="in_progress")]})
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == list(range(1, 102))
    assert any("page=2" in path for _, path in api.calls)
    first_post = next(i for i, (method, _) in enumerate(api.calls) if method == "POST")
    assert all(method == "GET" for method, _ in api.calls[:first_post])
    assert all(method == "POST" for method, _ in api.calls[first_post:])


@pytest.mark.parametrize("action,merged", [("closed", False), ("opened", False), ("synchronize", False)])
def test_non_merge_events_do_not_call_github(action, merged):
    api = FakeAPI()
    event = {**EVENT, "action": action, "pull_request": {"merged": merged}}
    assert HANDLER.cancel_merged_file_runs(event, api) == []
    assert api.calls == []


def test_a_run_finishing_during_cancellation_is_not_an_error():
    api = FakeAPI({"in_progress": [file_run(1)]}, error_code=409, conflict_status="completed")
    assert HANDLER.cancel_merged_file_runs(EVENT, api) == []
    assert api.calls[-1] == ("GET", f"{PREFIX}/runs/1")


@pytest.mark.parametrize("code,status", [(409, "in_progress"), (403, "completed"), (500, "completed")])
def test_cancellation_errors_are_not_silently_ignored(code, status):
    api = FakeAPI({"queued": [file_run(1)]}, error_code=code, conflict_status=status)
    with pytest.raises(urllib.error.HTTPError) as error:
        HANDLER.cancel_merged_file_runs(EVENT, api)
    assert error.value.code == code


def test_api_truncation_fails_before_any_cancellation():
    api = FakeAPI({"queued": [file_run(i) for i in range(1001)]})
    with pytest.raises(RuntimeError, match="1000-run limit"):
        HANDLER.cancel_merged_file_runs(EVENT, api)
    assert api.cancelled == []


@pytest.mark.parametrize("method,status", [("GET", 200), ("POST", 202)])
def test_http_requests_use_the_scoped_token_and_correct_method(monkeypatch, method, status):
    monkeypatch.setenv("GH_TOKEN", "test-token")

    def urlopen(request, *, timeout):
        assert request.full_url == f"https://api.github.com/{PREFIX}/runs/1"
        assert request.get_method() == method
        assert request.get_header("Authorization") == "Bearer test-token"
        assert timeout == 30
        response = io.BytesIO(json.dumps({"status": "completed"}).encode() if method == "GET" else b"")
        response.status = status
        return response

    monkeypatch.setattr(HANDLER.urllib.request, "urlopen", urlopen)
    result = HANDLER.github_request(f"{PREFIX}/runs/1", method=method)
    assert result == ({"status": "completed"} if method == "GET" else None)


def test_workflow_uses_trusted_code_and_keeps_file_queue_and_identity_contracts():
    workflow = (ROOT / ".github/workflows/cancel-merged-file-ci.yml").read_text()
    assert "pull_request_target:\n    types: [closed]" in workflow
    assert "if: github.event.pull_request.merged == true" in workflow
    assert "permissions:\n  contents: read\n  actions: write" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
    assert f"sparse-checkout: {SCRIPT}" in workflow
    assert f"run: python3 {SCRIPT}" in workflow
    assert "pull_request.head" not in workflow
    file_workflow = (ROOT / ".github/workflows/run-ci-file.yml").read_text()
    assert 'run-name: "/rerun-test ${{ inputs.test_file }} (PR #${{ inputs.pull_number }})"' in file_workflow
    assert "group: run-ci-file-${{ inputs.pull_number }}-${{ inputs.test_file }}" in file_workflow
    assert "cancel-in-progress: false\n  queue: max" in file_workflow
