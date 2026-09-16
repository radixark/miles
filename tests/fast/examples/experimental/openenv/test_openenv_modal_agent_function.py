"""Offline unit tests for the Modal-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is Modal's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as capacity limits, and that the start hook wires the
materialization (per-task dir, the build+create deadline, terminate-on-close).
"""

import sys
import types
from pathlib import Path

import openenv_modal_agent_function as omaf
import pytest

# --- start hook -------------------------------------------------------------


def test_start_hook_creates_per_task_and_closes_by_terminating(monkeypatch):
    """The sandbox is per-task (tasks_dir/task_id), the create carries the
    build+create deadline this backend owns, and close_fn terminates the sandbox."""
    created = {}
    terminated = []

    class _FakeSandbox:
        def terminate(self):
            terminated.append(True)

    sandbox = _FakeSandbox()

    def fake_create(task_dir, **kwargs):
        created.update(task_dir=task_dir, kwargs=kwargs)
        return sandbox, "https://abc.r5.modal.host"

    monkeypatch.setattr(omaf.tb2_sandbox_modal, "create_task_sandbox", fake_create)

    close_fn, url = omaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://abc.r5.modal.host"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert terminated == [True]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification():
    assert omaf._is_throttle_error(Exception("HTTP 429"))
    assert omaf._is_throttle_error(Exception("Too Many Requests"))
    # This backend's own vocabulary: a hit container-concurrency ceiling.
    assert omaf._is_throttle_error(Exception("RESOURCE EXHAUSTED: too many containers"))
    assert not omaf._is_throttle_error(RuntimeError("image build failed"))


def test_is_throttle_error_typed_modal_class():
    """The SDK's typed capacity error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    ex = pytest.importorskip("modal.exception")
    assert omaf._is_throttle_error(ex.ResourceExhaustedError("no capacity"))
    assert not omaf._is_throttle_error(ex.AuthError("bad token"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.setitem(sys.modules, "modal.exception", types.ModuleType("modal.exception"))
    assert omaf._is_throttle_error(Exception("HTTP 429"))
    assert not omaf._is_throttle_error(RuntimeError("image build failed"))
