"""Offline unit tests for the Runta-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Runta's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (per-task dir, delete-on-close).
"""

import asyncio
import sys
import types
from pathlib import Path

import openenv_runta_agent_function as oraf
import pytest


# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_deleting(monkeypatch):
    """The runtime is per-task (tasks_dir/task_id) and close_fn deletes it — the
    checkpoint stays, the runtime must not."""
    deleted = []
    created = {}
    runtime = types.SimpleNamespace(delete=lambda: deleted.append("runtime-handle"))

    def fake_create(task_dir, **kwargs):
        created.update(task_dir=task_dir, kwargs=kwargs)
        return runtime, "https://8000-abc.runta.dev"

    monkeypatch.setattr(oraf.tb2_sandbox_runta, "create_task_sandbox", fake_create)

    close_fn, url = oraf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://8000-abc.runta.dev"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    assert deleted == []
    close_fn()
    assert deleted == ["runtime-handle"]


# --- throttle classification ------------------------------------------------


class _FakeApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _fake_runta(monkeypatch):
    mod = types.ModuleType("runta")
    mod.ApiError = _FakeApiError
    monkeypatch.setitem(sys.modules, "runta", mod)


# What _checked_exec raises for a failed image build: the runtime's UUID and
# the tail of the build log, either of which can spell "429".
_BUILD_FAILURE = RuntimeError(
    "docker build failed in runtime 0b4290c1-7d1e-4f6a-9c42-9a1e5d3f4290 (exit 1): "
    "#7 ERROR: process did not complete successfully: exit code: 1\n"
    "E: Failed to fetch http://deb.debian.org/pool/main/c/curl_7.88.1.deb  429  Too Many Requests"
)


def test_is_throttle_error_is_not_read_from_message_text(monkeypatch):
    """Only the typed status counts. A create's own errors quote a runtime UUID
    and a build log, so matching text — as the other backends may — would turn
    a deterministic build failure into a throttle."""
    _fake_runta(monkeypatch)
    assert not oraf._is_throttle_error(_BUILD_FAILURE)
    assert not oraf._is_throttle_error(Exception("HTTP 429"))
    assert not oraf._is_throttle_error(Exception("rate limit exceeded, please retry"))
    assert not oraf._is_throttle_error(_FakeApiError("quota 429 MiB exceeded", status_code=400))


def test_a_failed_build_that_mentions_429_is_not_retried(monkeypatch):
    """End to end through the shared retry loop: the build failure surfaces on
    the first attempt instead of holding the build lock and a create slot for
    max_retries more."""
    _fake_runta(monkeypatch)
    attempts = []

    def failing_start(task_id, tasks_dir):
        attempts.append(task_id)
        raise _BUILD_FAILURE

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    monkeypatch.setattr(oraf.BACKEND, "start_sandbox", failing_start)
    monkeypatch.setattr(oraf.BACKEND, "backoff_base_s", 0.001)
    monkeypatch.setattr(oraf.BACKEND, "backoff_cap_s", 0.001)
    with pytest.raises(RuntimeError, match="docker build failed"):
        asyncio.run(oraf.BACKEND.start_task_sandbox("t1"))
    assert attempts == ["t1"]


def test_a_typed_429_is_retried(monkeypatch):
    _fake_runta(monkeypatch)
    attempts = []

    def throttled_once(task_id, tasks_dir):
        attempts.append(task_id)
        if len(attempts) == 1:
            raise _FakeApiError("slow down", status_code=429)
        return (lambda: None), "https://8000-abc.runta.dev"

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    monkeypatch.setattr(oraf.BACKEND, "start_sandbox", throttled_once)
    monkeypatch.setattr(oraf.BACKEND, "backoff_base_s", 0.001)
    monkeypatch.setattr(oraf.BACKEND, "backoff_cap_s", 0.001)
    _close_fn, url = asyncio.run(oraf.BACKEND.start_task_sandbox("t1"))
    assert url == "https://8000-abc.runta.dev"
    assert attempts == ["t1", "t1"]


def test_is_throttle_error_reads_the_status_off_the_sdk_error(monkeypatch):
    """The SDK raises one ApiError class for every API failure, so the status it
    carries is what tells a throttled create from a broken one — even when the
    message has no throttle keywords."""
    _fake_runta(monkeypatch)
    assert oraf._is_throttle_error(_FakeApiError("slow down", status_code=429))
    assert not oraf._is_throttle_error(_FakeApiError("bad token", status_code=401))
    assert not oraf._is_throttle_error(_FakeApiError("transport error"))


def test_is_throttle_error_typed_runta_class():
    """The real SDK's error carries the status the classifier reads."""
    runta = pytest.importorskip("runta")
    assert oraf._is_throttle_error(runta.ApiError("slow down", status_code=429))
    assert not oraf._is_throttle_error(runta.ApiError("bad token", status_code=401))
    # the status, not the text the SDK renders it into ("HTTP 500: ... 429 ...")
    assert not oraf._is_throttle_error(runta.ApiError("upstream said 429", status_code=500))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """Without the SDK nothing raised can be a Runta API error, so nothing is a
    throttle — and the classification itself must not fail on the import."""
    monkeypatch.setitem(sys.modules, "runta", None)
    assert not oraf._is_throttle_error(Exception("HTTP 429"))
    assert not oraf._is_throttle_error(RuntimeError("docker build failed"))
