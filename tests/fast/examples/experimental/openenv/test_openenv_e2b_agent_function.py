"""Offline unit tests for the E2B-sandbox agent function (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers only what is E2B's own. Episode dispatch, create throttling, backoff
budgets and the cancel-mid-create reaper belong to the shared SandboxBackend and are
proven once in test_openenv_sandbox_common.py; what remains here is which of
this SDK's errors count as throttling — including the env-knob extension that
lets a self-hosted AgentENV's capacity errors be retried without a code change
— and that the start hook wires the materialization correctly.
"""

import sys
import types
from pathlib import Path

import openenv_e2b_agent_function as oeaf
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
        return "sandbox-handle", "https://8000-abc.e2b.app"

    monkeypatch.setattr(oeaf.tb2_sandbox_e2b, "create_task_sandbox", fake_create)
    monkeypatch.setattr(oeaf.tb2_sandbox_e2b, "kill_sandbox", killed.append)

    close_fn, url = oeaf._start_sandbox("regex-chess", "/tasks")

    assert url == "https://8000-abc.e2b.app"
    assert created["task_dir"] == Path("/tasks/regex-chess")
    # The backend passes only what identifies the sandbox; every deadline is
    # the materialization's own default, so the two cannot drift apart.
    assert created["kwargs"] == {}
    close_fn()
    assert killed == ["sandbox-handle"]


# --- throttle classification ------------------------------------------------


def test_is_throttle_error_classification(monkeypatch):
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    assert oeaf._is_throttle_error(_Throttled())
    assert oeaf._is_throttle_error(Exception("HTTP 429"))
    assert oeaf._is_throttle_error(Exception("Too Many Requests"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))


def test_is_throttle_error_extra_patterns_env(monkeypatch):
    """OPENENV_E2B_THROTTLE_PATTERNS extends the retryable set for
    provider-specific capacity errors (e.g. a full self-hosted AgentENV pool)
    without a code change."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    assert not oeaf._is_throttle_error(Exception("no node with sufficient capacity"))
    monkeypatch.setenv("OPENENV_E2B_THROTTLE_PATTERNS", "sufficient capacity, at capacity")
    assert oeaf._is_throttle_error(Exception("no node with sufficient capacity"))
    assert oeaf._is_throttle_error(Exception("cluster AT CAPACITY"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))


def test_is_throttle_error_typed_e2b_class(monkeypatch):
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    ex = pytest.importorskip("e2b.exceptions")
    assert oeaf._is_throttle_error(ex.RateLimitException("slow down"))
    assert not oeaf._is_throttle_error(ex.AuthenticationException("bad key"))


def test_is_throttle_error_without_the_sdk_installed(monkeypatch):
    """The typed check is best-effort: the backend still classifies by text when the
    SDK (or that class) is absent, rather than failing the classification."""
    monkeypatch.delenv("OPENENV_E2B_THROTTLE_PATTERNS", raising=False)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", types.ModuleType("e2b.exceptions"))
    assert oeaf._is_throttle_error(Exception("HTTP 429"))
    assert not oeaf._is_throttle_error(RuntimeError("template build failed"))
