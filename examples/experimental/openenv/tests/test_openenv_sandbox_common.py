"""Offline unit tests for the shared sandbox layer (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers what every backend inherits, so it is proven once rather than once per
provider:
  - the backend registry, whose whole job is to refuse to guess: which provider
    runs decides whose quota a rollout spends, so an unnamed or unknown one
    must fail at launch rather than resolve to whichever backend came first;
  - episode dispatch: a backend's run_episode sends raw exec commands and scores
    via the standard `evaluate` action;
  - sandbox-create throttling: throttle errors (typed by the backend, or textual)
    are retried with backoff and a bounded budget, anything else propagates
    immediately; a cancel mid-create reaps the orphaned sandbox.

A backend's own tests cover only what is genuinely its own: which of its SDK's
errors count as throttling, and how its start hook wires the materialization.
"""

import asyncio
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openenv_agent_function as oaf  # noqa: E402
import openenv_sandbox_common as common  # noqa: E402
from test_openenv_agent_function import _CLASSES, _FakeEnv, _FakePolicy, _FakeResult  # noqa: E402

logger = logging.getLogger("test-backend")


class _Throttled(Exception):
    def __str__(self):
        return "rate limit exceeded, please retry"


def _backend(**overrides) -> common.SandboxBackend:
    """A backend whose provider-specific half is a stub.

    Every real backend is this object plus a start hook and a throttle classifier,
    so exercising it here is exercising all of them. Backoff is shrunk to keep
    the retry tests instant.
    """
    spec = {
        "provider": "Fake",
        "start_sandbox": lambda task_id, tasks_dir: ((lambda: None), "http://sandbox:8000"),
        "is_throttle": lambda exc: isinstance(exc, _Throttled),
        "logger": logger,
        "backoff_base_s": 0.001,
        "backoff_cap_s": 0.001,
    }
    return common.SandboxBackend(**{**spec, **overrides})


@asynccontextmanager
async def _fake_episode_env(env_cls, metadata):
    yield env_cls()


# --- backend registry -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("daytona", "daytona"),
        ("e2b", "e2b"),
        ("agentenv", "e2b"),
        ("  E2B  ", "e2b"),
        ("modal", "modal"),
    ],
)
def test_resolve_backend_normalizes_names_and_aliases(name, expected):
    assert common.resolve_backend(name) == expected


@pytest.mark.parametrize("name", [None, "", "   "])
def test_resolve_backend_refuses_to_pick_for_you(name):
    with pytest.raises(ValueError, match="no sandbox backend named"):
        common.resolve_backend(name)


def test_resolve_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        common.resolve_backend("nonesuch")


def test_every_registered_backend_names_an_importable_target():
    """The registry is what the launcher passes to --custom-agent-function-path."""
    for backend, path in common.AGENT_FUNCTIONS.items():
        module, _, func = path.rpartition(".")
        assert func == "run", backend
        assert module == common.AGENT_MODULES[backend], backend
        assert (Path(__file__).resolve().parent.parent / f"{module}.py").is_file(), backend


@pytest.mark.parametrize("name", sorted(common.AGENT_MODULES))
def test_every_backend_exposes_the_entry_points(name):
    """load_backend is how the sibling tools reach a provider; the module-level
    run/run_episode are how training and the eval driver reach it."""
    backend = common.load_backend(name)
    assert isinstance(backend, common.SandboxBackend)
    module = sys.modules[common.AGENT_MODULES[name]]
    assert module.run == backend.run
    assert module.run_episode == backend.run_episode


@pytest.mark.parametrize("name", sorted(common.AGENT_MODULES))
def test_operator_facing_help_names_every_backend(name):
    """The sibling tools tell an operator which backends exist. That list drifted
    once already (modal was added and the help text still said daytona/e2b), so
    it is generated from the registry where it can be, and asserted where it
    cannot."""
    assert name in common.backend_names()
    root = Path(__file__).resolve().parent.parent
    assert name in (root / "eval_tbench2_via_api.py").read_text().split('"""')[1]


def test_backend_knobs_read_the_providers_own_prefix(monkeypatch):
    """One provider's knobs must not move another's."""
    monkeypatch.setenv("OPENENV_FAKE_CREATE_CONCURRENCY", "9")
    knobs = common.backend_knobs("FAKE")
    assert knobs["create_concurrency"] == 9
    assert knobs["max_retries"] == 8  # unset falls back to the shared default


@pytest.mark.parametrize("name", sorted(common.AGENT_MODULES))
def test_every_backend_owns_its_provider_hooks(name):
    """Each backend must supply its OWN start hook and throttle classifier;
    inheriting a sibling's would silently run episodes on the wrong provider."""
    backend = common.load_backend(name)
    module = sys.modules[common.AGENT_MODULES[name]]
    assert backend.provider  # human-readable, used in the throttle log line
    assert backend.start_sandbox.__module__ == module.__name__
    assert backend.is_throttle.__module__ == module.__name__


# --- episode dispatch -------------------------------------------------------


def test_backend_dispatch(monkeypatch):
    """A backend's run_episode: exec raw (the server resolves the workdir), scoring
    via the standard `evaluate` action, no canonical exec, no rm-hack."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)
    backend = _backend()
    backend.episode_env = _fake_episode_env

    reward, metrics = asyncio.run(
        backend.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert any(a.action_type == "evaluate" for a in actions)
    assert not any("test.sh" in (a.command or "") for a in execs)
    assert not any("/tmp/tbench2_env_runs" in (a.command or "") for a in execs)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1


def test_backend_eval_error_yields_no_verdict(monkeypatch):
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

    monkeypatch.setattr(oaf, "load_tbench2", lambda: {"env": _EvalErrorEnv, "action": _CLASSES["action"]})
    backend = _backend()
    backend.episode_env = _fake_episode_env

    reward, metrics = asyncio.run(
        backend.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert reward is None
    assert metrics["turns"] == 2  # the episode itself completed; only scoring failed


def test_episode_env_requires_a_task_id():
    """The sandbox is built for ONE task, so an episode without a task id is a
    caller bug, not something to guess a default for."""
    backend = _backend()

    async def scenario():
        async with backend.episode_env(_FakeEnv, {}):
            pass

    with pytest.raises(ValueError, match="task_id"):
        asyncio.run(scenario())


# --- sandbox-create throttling ----------------------------------------------


def test_create_retries_through_throttling(monkeypatch):
    """Throttle errors are retried (with backoff) until the create succeeds."""
    calls = {"n": 0}

    def flaky_start(task_id, tasks_dir):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _Throttled()
        return (lambda: None), "http://sandbox:8000"

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    close_fn, url = asyncio.run(_backend(start_sandbox=flaky_start).start_task_sandbox("t1"))
    assert url == "http://sandbox:8000"
    assert calls["n"] == 4  # 3 throttled attempts + 1 success


def test_create_gives_up_after_retry_budget(monkeypatch):
    """A create that is throttled past max_retries raises the error."""
    calls = {"n": 0}

    def always_throttled(task_id, tasks_dir):
        calls["n"] += 1
        raise _Throttled()

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    backend = _backend(start_sandbox=always_throttled, max_retries=2)
    with pytest.raises(_Throttled):
        asyncio.run(backend.start_task_sandbox("t1"))
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_create_non_throttle_error_propagates_immediately(monkeypatch):
    """Anything that is not a throttle error must not be retried."""
    calls = {"n": 0}

    def broken_start(task_id, tasks_dir):
        calls["n"] += 1
        raise RuntimeError("image build failed")

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    with pytest.raises(RuntimeError):
        asyncio.run(_backend(start_sandbox=broken_start).start_task_sandbox("t1"))
    assert calls["n"] == 1


def test_cancel_during_create_reaps_orphaned_sandbox(monkeypatch):
    """Cancelling an episode mid-create must not leak the sandbox: the worker
    thread finishes the create in the background and the reaper closes it."""
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def slow_start(task_id, tasks_dir):
        started.set()
        assert release.wait(5)
        return (lambda: closed.set()), "http://sandbox:8000"

    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/nonexistent")
    backend = _backend(start_sandbox=slow_start)

    async def scenario():
        task = asyncio.create_task(backend.start_task_sandbox("t1"))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Only now does the in-flight create finish — after the awaiter is gone.
        release.set()

    asyncio.run(scenario())
    assert closed.wait(5)  # the reaper closed the orphan


# --- throttle-text classification -------------------------------------------


def test_throttle_text_matches_the_shared_vocabulary():
    for message in ("HTTP 429", "Too Many Requests", "rate limit exceeded"):
        assert common.throttle_text(Exception(message))
    assert not common.throttle_text(RuntimeError("image build failed"))


def test_throttle_text_takes_the_backends_own_patterns():
    assert common.throttle_text(Exception("ThrottlerException"), "throttler")
    assert not common.throttle_text(Exception("ThrottlerException"))


def test_throttle_text_extra_patterns_env(monkeypatch):
    """The env knob extends the retryable set for capacity errors whoever
    deployed the endpoint words their own way (a full self-hosted pool)."""
    env = "OPENENV_FAKE_THROTTLE_PATTERNS"
    monkeypatch.delenv(env, raising=False)
    assert not common.throttle_text(Exception("no node with sufficient capacity"), patterns_env_var=env)
    monkeypatch.setenv(env, "sufficient capacity, at capacity")
    assert common.throttle_text(Exception("no node with sufficient capacity"), patterns_env_var=env)
    assert common.throttle_text(Exception("cluster AT CAPACITY"), patterns_env_var=env)
    assert not common.throttle_text(RuntimeError("image build failed"), patterns_env_var=env)
