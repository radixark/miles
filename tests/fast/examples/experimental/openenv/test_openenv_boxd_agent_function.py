"""Offline unit tests for the boxd-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is boxd's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling, and that the start hook wires the
materialization (per-task dir, delete-on-close).
"""

import sys
import types
from pathlib import Path

import openenv_boxd_agent_function as obaf
import pytest


def test_start_hook_creates_per_task_and_closes_by_deleting(monkeypatch):
    """The machine is per-task (tasks_dir/task_id) and close_fn deletes it by id —
    the snapshot stays, the machine must not."""
    deleted = []
    created = {}

    def fake_create(task_dir, **kwargs):
        created.update(task_dir=task_dir, kwargs=kwargs)
        return "machine-id", "https://tb2-regex-chess-1a2b3c4d.fra.boxd.sh"

    monkeypatch.setattr(obaf.tb2_sandbox_boxd, "create_task_sandbox", fake_create)
    monkeypatch.setattr(obaf.tb2_sandbox_boxd, "delete_sandbox", deleted.append)

    close_fn, url = obaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://tb2-regex-chess-1a2b3c4d.fra.boxd.sh"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # every deadline is the materialization's own default, so the two cannot drift apart
    assert created["kwargs"] == {}
    close_fn()
    assert deleted == ["machine-id"]


def test_is_throttle_error_classification():
    assert obaf._is_throttle_error(Exception("HTTP 429"))
    assert obaf._is_throttle_error(Exception("Too Many Requests"))
    assert obaf._is_throttle_error(Exception("rate limit exceeded"))
    assert not obaf._is_throttle_error(RuntimeError("docker build exited 1"))


def test_is_throttle_error_typed_boxd_class():
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    sdk = pytest.importorskip("boxd")
    assert obaf._is_throttle_error(sdk.RateLimitError("machine limit reached"))
    assert not obaf._is_throttle_error(sdk.AuthenticationError("bad key"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "boxd", types.ModuleType("boxd"))
    assert obaf._is_throttle_error(Exception("HTTP 429"))
    assert not obaf._is_throttle_error(RuntimeError("docker build exited 1"))
