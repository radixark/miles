"""Offline unit tests for the E2B-sandbox agent function (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers what a live episode cannot cheaply prove:
  - episode dispatch: this module's run_episode sends raw exec commands and
    scores via the standard `evaluate` action;
  - sandbox-create throttling: rate-limit errors (typed, textual, and the
    OPENENV_E2B_THROTTLE_PATTERNS extension for self-hosted providers) are
    retried with backoff and a bounded budget, anything else propagates
    immediately; a cancel mid-create reaps the orphaned sandbox.
"""

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openenv_agent_function as oaf  # noqa: E402
import openenv_e2b_agent_function as oeaf  # noqa: E402
from test_openenv_agent_function import _CLASSES, _FakeEnv, _FakePolicy, _FakeResult  # noqa: E402


def run_async(coro):
    """asyncio.run with the module's loop-bound state reset (fresh loop per test)."""
    oeaf._create_sem = None
    return asyncio.run(coro)


# --- episode dispatch ------------------------------------------------------


def test_e2b_leg_dispatch(monkeypatch):
    """The e2b run_episode: exec raw (server resolves the workdir), scoring
    via the standard `evaluate` action, no canonical exec, no rm-hack."""
    monkeypatch.setattr(oaf, "_load_tbench2", lambda: _CLASSES)

    @asynccontextmanager
    async def fake_episode_env(env_cls, metadata):
        yield env_cls()

    monkeypatch.setattr(oeaf, "_episode_env", fake_episode_env)

    reward, metrics = run_async(
        oeaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert any(a.action_type == "evaluate" for a in actions)
    assert not any("test.sh" in (a.command or "") for a in execs)
    assert not any("/tmp/tbench2_env_runs" in (a.command or "") for a in execs)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1


def test_e2b_leg_eval_error_yields_no_verdict(monkeypatch):
    """A server-side scoring failure (`evaluate` comes back with error set and
    no reward) surfaces as reward=None -- dropped by the training wrapper --
    not coerced into a false-negative 0.0."""

    class _EvalErrorEnv(_FakeEnv):
        async def step(self, action):
            if action.action_type == "evaluate":
                self.actions.append(action)
                res = _FakeResult()
                res.observation.error = "toolkit timeout"
                return res
            return await super().step(action)

    monkeypatch.setattr(oaf, "_load_tbench2", lambda: {"env": _EvalErrorEnv, "action": _CLASSES["action"]})

    @asynccontextmanager
    async def fake_episode_env(env_cls, metadata):
        yield env_cls()

    monkeypatch.setattr(oeaf, "_episode_env", fake_episode_env)

    reward, metrics = run_async(
        oeaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert reward is None
    assert metrics["turns"] == 2  # the episode itself completed; only scoring failed


# --- sandbox-create throttling ----------------------------------------------


class _Throttled(Exception):
    def __str__(self):
        return "rate limit exceeded, please retry"


def _patch_fast_backoff(monkeypatch):
    monkeypatch.setattr(oeaf, "_CREATE_BACKOFF_BASE_S", 0.001)
    monkeypatch.setattr(oeaf, "_CREATE_BACKOFF_CAP_S", 0.001)


def test_create_retries_through_throttling(monkeypatch):
    """Throttle errors are retried (with backoff) until the create succeeds."""
    _patch_fast_backoff(monkeypatch)
    calls = {"n": 0}

    def flaky_start(task_id, tasks_dir):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _Throttled()
        return (lambda: None), "http://sandbox:8000"

    monkeypatch.setattr(oeaf, "_start_sandbox", flaky_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    close_fn, url = run_async(oeaf._start_task_sandbox("t1"))
    assert url == "http://sandbox:8000"
    assert calls["n"] == 4  # 3 throttled attempts + 1 success


def test_create_gives_up_after_retry_budget(monkeypatch):
    """A create that is throttled past _CREATE_MAX_RETRIES raises the error."""
    _patch_fast_backoff(monkeypatch)
    monkeypatch.setattr(oeaf, "_CREATE_MAX_RETRIES", 2)
    calls = {"n": 0}

    def always_throttled(task_id, tasks_dir):
        calls["n"] += 1
        raise _Throttled()

    monkeypatch.setattr(oeaf, "_start_sandbox", always_throttled)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    with pytest.raises(_Throttled):
        run_async(oeaf._start_task_sandbox("t1"))
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_cancel_during_create_reaps_orphaned_sandbox(monkeypatch):
    """Cancelling an episode mid-create must not leak the sandbox: the worker
    thread finishes the create in the background and the reaper kills it."""
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def slow_start(task_id, tasks_dir):
        started.set()
        assert release.wait(5)
        return (lambda: closed.set()), "http://sandbox:8000"

    monkeypatch.setattr(oeaf, "_start_sandbox", slow_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    async def scenario():
        task = asyncio.create_task(oeaf._start_task_sandbox("t1"))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Only now does the in-flight create finish — after the awaiter is gone.
        release.set()

    run_async(scenario())
    assert closed.wait(5)  # the reaper killed the orphan


def test_create_non_throttle_error_propagates_immediately(monkeypatch):
    """Anything that is not a rate-limit error must not be retried."""
    _patch_fast_backoff(monkeypatch)
    calls = {"n": 0}

    def broken_start(task_id, tasks_dir):
        calls["n"] += 1
        raise RuntimeError("template build failed")

    monkeypatch.setattr(oeaf, "_start_sandbox", broken_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    with pytest.raises(RuntimeError):
        run_async(oeaf._start_task_sandbox("t1"))
    assert calls["n"] == 1


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


def test_is_throttle_error_typed_e2b_class():
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    ex = pytest.importorskip("e2b.exceptions")
    assert oeaf._is_throttle_error(ex.RateLimitException("slow down"))
    assert not oeaf._is_throttle_error(ex.AuthenticationException("bad key"))
