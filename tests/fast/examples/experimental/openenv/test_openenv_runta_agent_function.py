"""Offline unit tests for the Runta-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Runta's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (per-task dir, delete-on-close).
"""

import sys
import types
from pathlib import Path

import openenv_runta_agent_function as oraf
import pytest


class _Throttled(Exception):
    def __str__(self):
        return "rate limit exceeded, please retry"


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


def test_is_throttle_error_classification(monkeypatch):
    _fake_runta(monkeypatch)
    assert oraf._is_throttle_error(_Throttled())
    assert oraf._is_throttle_error(Exception("HTTP 429"))
    assert oraf._is_throttle_error(Exception("Too Many Requests"))
    assert not oraf._is_throttle_error(RuntimeError("docker build failed"))


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


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "runta", None)
    assert oraf._is_throttle_error(Exception("HTTP 429"))
    assert not oraf._is_throttle_error(RuntimeError("docker build failed"))
