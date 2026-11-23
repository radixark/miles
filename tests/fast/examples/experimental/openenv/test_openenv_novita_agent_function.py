"""Offline unit tests for the Novita-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Novita's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (per-task dir, kill-on-close).
"""

import sys
import types
from pathlib import Path

import openenv_novita_agent_function as onaf
import pytest


class _Throttled(Exception):
    def __str__(self):
        return "rate limit exceeded, please retry"


# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_killing(monkeypatch):
    """The sandbox is per-task (tasks_dir/task_id) and close_fn kills it — the
    template stays, the sandbox must not."""
    killed = []
    created = {}

    def fake_create(task_dir, **kwargs):
        created.update(task_dir=task_dir, kwargs=kwargs)
        return "sandbox-handle", "https://8000-abc.us-phx-1.sandbox.novita.ai"

    monkeypatch.setattr(onaf.tb2_sandbox_novita, "create_task_sandbox", fake_create)
    monkeypatch.setattr(onaf.tb2_sandbox_novita, "kill_sandbox", killed.append)

    close_fn, url = onaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://8000-abc.us-phx-1.sandbox.novita.ai"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert killed == ["sandbox-handle"]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification():
    assert onaf._is_throttle_error(_Throttled())
    assert onaf._is_throttle_error(Exception("HTTP 429"))
    assert onaf._is_throttle_error(Exception("Too Many Requests"))
    # This backend's own vocabulary: the account's concurrent-sandbox ceiling.
    assert onaf._is_throttle_error(Exception("sandbox quota exceeded"))
    assert not onaf._is_throttle_error(RuntimeError("template build failed"))


def test_is_throttle_error_typed_novita_class():
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    sdk = pytest.importorskip("novita_sandbox")
    assert onaf._is_throttle_error(sdk.RateLimitException("slow down"))
    assert not onaf._is_throttle_error(sdk.AuthenticationException("bad key"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "novita_sandbox", types.ModuleType("novita_sandbox"))
    assert onaf._is_throttle_error(Exception("HTTP 429"))
    assert not onaf._is_throttle_error(RuntimeError("template build failed"))
