"""Offline unit tests for the openenv tbench2 adapter (no network, no GPU).

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the adapter:

    pytest examples/experimental/openenv/tests/ -q

Covers the shared-server leg of the agent loop (this module's run_episode):
its exec form, scoring path, and cleanup. The Daytona-sandbox leg's
dispatch and sandbox-create machinery live in
test_openenv_daytona_agent_function.py; the fakes below are shared with it.
"""

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openenv_agent_function as oaf  # noqa: E402


def run_async(coro):
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


# --- episode dispatch ------------------------------------------------------


def test_shared_leg_dispatch(monkeypatch):
    """The shared-server run_episode: exec prefixed with the task workdir,
    canonical-exec scoring, rm-hack present, standard `evaluate` never used."""
    monkeypatch.setattr(oaf, "_load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(
        oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "cd /app && echo hi"
    assert any("bash /tests/test.sh" in a.command for a in execs)
    assert any("/tmp/tbench2_env_runs" in a.command for a in execs), "rm-hack missing"
    assert not any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2 and metrics["tool_calls"] == 1
