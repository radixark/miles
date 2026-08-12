"""Offline unit tests for the Daytona-sandbox agent function (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers only what is Daytona's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (client, per-task dir, delete-on-close) correctly.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openenv_daytona_agent_function as odaf  # noqa: E402


class _Throttled(Exception):
    def __str__(self):
        return "ThrottlerException: Too Many Requests"


# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_deleting(monkeypatch):
    """The sandbox is per-task (tasks_dir/task_id) and close_fn must DELETE it:
    a stopped-but-kept Daytona sandbox still occupies the org's quota."""
    deleted = []
    created = {}

    class _FakeDaytona:
        def delete(self, sandbox):
            deleted.append(sandbox)

    client = _FakeDaytona()

    def fake_create(daytona, task_dir, **kwargs):
        created.update(daytona=daytona, task_dir=task_dir, kwargs=kwargs)
        return "sandbox-handle", "https://preview.daytona/8000"

    monkeypatch.setattr(odaf.tb2_sandbox_daytona, "make_daytona", lambda: client)
    monkeypatch.setattr(odaf.tb2_sandbox_daytona, "create_task_sandbox", fake_create)

    close_fn, url = odaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://preview.daytona/8000"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    assert created["daytona"] is client
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert deleted == ["sandbox-handle"]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification():
    assert odaf._is_throttle_error(_Throttled())
    assert odaf._is_throttle_error(Exception("HTTP 429"))
    assert odaf._is_throttle_error(Exception("throttler tripped"))  # this backend's own vocabulary
    assert not odaf._is_throttle_error(RuntimeError("image build failed"))


def test_is_throttle_error_typed_daytona_class():
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords."""
    errors = pytest.importorskip("daytona.common.errors")
    assert odaf._is_throttle_error(errors.DaytonaRateLimitError("slow down"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "daytona.common.errors", types.ModuleType("daytona.common.errors"))
    assert odaf._is_throttle_error(Exception("HTTP 429"))
    assert not odaf._is_throttle_error(RuntimeError("image build failed"))
