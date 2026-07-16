"""Offline unit tests for the openenv tbench2 adapter (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers the two things a live episode cannot cheaply prove:
  - backend dispatch: the per-task-sandbox leg and the shared-server leg of
    _multi_turn each use their own exec form and scoring path;
  - sandbox-create throttling: Daytona rate-limit errors are retried with
    backoff and a bounded budget, anything else propagates immediately; a
    cancel mid-create reaps the orphaned sandbox instead of leaking it.
"""

import asyncio
import sys
import threading
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openenv_agent_function as oaf  # noqa: E402


def run_async(coro):
    """asyncio.run with the module's loop-bound state reset (fresh loop per test)."""
    oaf._create_sem = None
    return asyncio.run(coro)


# --- fakes ---------------------------------------------------------------


class _FakeObs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResult:
    def __init__(self, output="", reward=None, instruction=""):
        self.observation = _FakeObs(output=output, instruction=instruction)
        if reward is not None:
            self.reward = reward


class _FakeEnv:
    """Records every step() action; answers both scoring protocols."""

    last_actions: list = []

    def __init__(self, base_url="", message_timeout_s=0):
        self.actions = []
        _FakeEnv.last_actions = self.actions

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def reset(self, task_id=None):
        return _FakeResult(instruction="do the thing")

    async def step(self, action):
        self.actions.append(action)
        if action.action_type == "evaluate":
            return _FakeResult(reward=1.0)
        if "test.sh" in (action.command or ""):
            return _FakeResult(output=f"{oaf._REWARD_MARKER}1.0")
        return _FakeResult(output="ok")


class _FakeAction:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


class _FakePolicy:
    """Turn 1: emit a bash command. Turn 2: TASK_COMPLETE."""

    def __init__(self):
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.n += 1
        text = "```bash\necho hi\n```" if self.n == 1 else "TASK_COMPLETE"
        msg = types.SimpleNamespace(
            content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


_CLASSES = {"env": _FakeEnv, "action": _FakeAction}


async def _run_episode():
    return await oaf._multi_turn(
        _CLASSES, _FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"}
    )


# --- backend dispatch ------------------------------------------------------


def test_shared_leg_dispatch(monkeypatch):
    """per_task off: exec prefixed with the task workdir, canonical-exec scoring,
    rm-hack present, standard `evaluate` never used."""
    monkeypatch.delenv("OPENENV_TB2_TASKS_DIR", raising=False)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(_run_episode())
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "cd /app && echo hi"
    assert any("bash /tests/test.sh" in a.command for a in execs)
    assert any("/tmp/tbench2_env_runs" in a.command for a in execs), "rm-hack missing"
    assert not any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1


def test_per_task_leg_dispatch(monkeypatch):
    """per_task on: exec raw (server resolves the workdir), scoring via the
    standard `evaluate` action, no canonical exec, no rm-hack."""
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    @asynccontextmanager
    async def fake_episode_env(env_cls, metadata):
        yield env_cls()

    monkeypatch.setattr(oaf, "_episode_env", fake_episode_env)

    reward, metrics = run_async(_run_episode())
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert any(a.action_type == "evaluate" for a in actions)
    assert not any("test.sh" in (a.command or "") for a in execs)
    assert not any("/tmp/tbench2_env_runs" in (a.command or "") for a in execs)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1


# --- sandbox-create throttling ----------------------------------------------


class _Throttled(Exception):
    def __str__(self):
        return "ThrottlerException: Too Many Requests"


def _patch_fast_backoff(monkeypatch):
    monkeypatch.setattr(oaf, "_CREATE_BACKOFF_BASE_S", 0.001)
    monkeypatch.setattr(oaf, "_CREATE_BACKOFF_CAP_S", 0.001)


def test_create_retries_through_throttling(monkeypatch):
    """Throttle errors are retried (with backoff) until the create succeeds."""
    _patch_fast_backoff(monkeypatch)
    calls = {"n": 0}

    def flaky_start(task_id, tasks_dir):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _Throttled()
        return (lambda: None), "http://sandbox:8000"

    monkeypatch.setattr(oaf, "_start_declarative", flaky_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    close_fn, url = run_async(oaf._start_task_sandbox("t1"))
    assert url == "http://sandbox:8000"
    assert calls["n"] == 4  # 3 throttled attempts + 1 success


def test_create_gives_up_after_retry_budget(monkeypatch):
    """A create that is throttled past _CREATE_MAX_RETRIES raises the error."""
    _patch_fast_backoff(monkeypatch)
    monkeypatch.setattr(oaf, "_CREATE_MAX_RETRIES", 2)
    calls = {"n": 0}

    def always_throttled(task_id, tasks_dir):
        calls["n"] += 1
        raise _Throttled()

    monkeypatch.setattr(oaf, "_start_declarative", always_throttled)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    with pytest.raises(_Throttled):
        run_async(oaf._start_task_sandbox("t1"))
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_cancel_during_create_reaps_orphaned_sandbox(monkeypatch):
    """Cancelling an episode mid-create must not leak the sandbox: the worker
    thread finishes the create in the background and the reaper deletes it."""
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def slow_start(task_id, tasks_dir):
        started.set()
        assert release.wait(5)
        return (lambda: closed.set()), "http://sandbox:8000"

    monkeypatch.setattr(oaf, "_start_declarative", slow_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    async def scenario():
        task = asyncio.create_task(oaf._start_task_sandbox("t1"))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Only now does the in-flight create finish — after the awaiter is gone.
        release.set()

    run_async(scenario())
    assert closed.wait(5)  # the reaper deleted the orphan


def test_create_non_throttle_error_propagates_immediately(monkeypatch):
    """Anything that is not a rate-limit error must not be retried."""
    _patch_fast_backoff(monkeypatch)
    calls = {"n": 0}

    def broken_start(task_id, tasks_dir):
        calls["n"] += 1
        raise RuntimeError("image build failed")

    monkeypatch.setattr(oaf, "_start_declarative", broken_start)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")

    with pytest.raises(RuntimeError):
        run_async(oaf._start_task_sandbox("t1"))
    assert calls["n"] == 1


def test_is_throttle_error_classification():
    assert oaf._is_throttle_error(_Throttled())
    assert oaf._is_throttle_error(Exception("HTTP 429"))
    assert not oaf._is_throttle_error(RuntimeError("image build failed"))


def test_is_throttle_error_typed_daytona_class():
    """The SDK's typed rate-limit error is recognized even when its message
    carries no throttle keywords; sibling error classes are not."""
    errors = pytest.importorskip("daytona.common.errors")
    assert oaf._is_throttle_error(errors.DaytonaRateLimitError("slow down"))
    assert not oaf._is_throttle_error(errors.DaytonaValidationError("bad params"))
