"""Offline unit tests for the openenv tbench2 adapter (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

Covers the shared-server leg of the agent loop (this module's run_episode):
its exec form, scoring path, and cleanup. The sandbox leg's dispatch and
sandbox-create machinery live in test_openenv_sandbox_common.py; the fakes
below are shared with it.
"""

import asyncio
import types

import openenv_agent_function as oaf
import pytest


def run_async(coro):
    return asyncio.run(coro)


# --- fakes ---------------------------------------------------------------


class _FakeObs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResult:
    def __init__(self, output="", reward=None, instruction="", info=None):
        self.observation = _FakeObs(output=output, instruction=instruction, info=info or {})
        if reward is not None:
            self.reward = reward


class _FakeEnv:
    """Records every step() action; answers `evaluate` like a contract-carrying
    server (reward plus the canonical-harness marker)."""

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
            return _FakeResult(reward=1.0, info={"tests_passed": True, "harness": "tests/test.sh"})
        return _FakeResult(output="ok")


class _FakeAction:
    def __init__(self, action_type, command=None):
        self.action_type = action_type
        self.command = command


def _fake_completion(text, finish_reason="stop"):
    msg = types.SimpleNamespace(
        content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish_reason)])


class _FakePolicy:
    """Turn 1: emit a bash command. Turn 2: TASK_COMPLETE."""

    def __init__(self):
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.n += 1
        text = "```bash\necho hi\n```" if self.n == 1 else "TASK_COMPLETE"
        return _fake_completion(text)


class _NoCommandPolicy(_FakePolicy):
    """Emits only whitespace, so there is no command to execute."""

    async def _create(self, **kw):
        self.n += 1
        return _fake_completion(" \n")


_CLASSES = {"env": _FakeEnv, "action": _FakeAction}


# --- episode dispatch ------------------------------------------------------


def test_shared_leg_dispatch(monkeypatch):
    """The shared-server run_episode: exec commands pass through unmodified
    (the server resolves the workdir), scoring via the standard `evaluate`
    action — and the trial-dir purge (post_episode) runs, since the shared
    server outlives the episode."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(
        oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]

    assert reward == 1.0
    assert execs[0].command == "echo hi"
    assert any("/tmp/tbench2_env_runs" in (a.command or "") for a in execs), "trial-dir purge missing"
    assert any(a.action_type == "evaluate" for a in actions)
    assert metrics["turns"] == 2
    assert metrics["end_reason"] == "task_complete"
    assert metrics["tool_calls"] == 1


class _TruncatingPolicy(_FakePolicy):
    """Emits a command the model never finished writing."""

    async def _create(self, **kw):
        completion = await super()._create(**kw)
        completion.choices[0].finish_reason = "length"
        return completion


def test_length_capped_turn_ends_the_episode(monkeypatch):
    """A turn cut off by the token cap must not be executed: the command is
    truncated, so running it would send an arbitrary prefix to the sandbox."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    _, metrics = run_async(
        oaf.run_episode(_TruncatingPolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )

    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "length"
    assert metrics["tool_calls"] == 0
    execs = [a for a in _FakeEnv.last_actions if a.action_type == "exec"]
    assert not [a for a in execs if "echo hi" in (a.command or "")], "ran a command the model never finished"


def test_no_command_reports_end_reason(monkeypatch):
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    _, metrics = run_async(
        oaf.run_episode(_NoCommandPolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )

    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "no_command"
    assert metrics["tool_calls"] == 0


def test_max_turns_reports_end_reason(monkeypatch):
    monkeypatch.setenv("OPENENV_MAX_TURNS", "1")
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    policy = _FakePolicy()
    _, metrics = run_async(oaf.run_episode(policy, "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"}))

    assert policy.n == 1
    assert metrics["turns"] == 1
    assert metrics["end_reason"] == "max_turns"
    assert metrics["tool_calls"] == 1


def test_old_server_reward_is_not_trusted(monkeypatch):
    """A server without the canonical contract (e.g. an out-of-date install)
    answers `evaluate` with a plausible-looking reward but no harness marker
    (its info is {tests_passed, exit_code} from bare pytest). That reward must
    be dropped, not ingested: source preflight is impossible against a remote
    server."""

    class _OldServerEnv(_FakeEnv):
        async def step(self, action):
            if action.action_type == "evaluate":
                self.actions.append(action)
                return _FakeResult(reward=1.0, info={"tests_passed": True, "exit_code": 0})
            return await super().step(action)

    monkeypatch.setattr(oaf, "load_tbench2", lambda: {"env": _OldServerEnv, "action": _CLASSES["action"]})

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    reward, metrics = run_async(
        oaf.run_episode(_FakePolicy(), "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert reward is None
    assert metrics["turns"] == 2  # the episode itself completed; only scoring was rejected


class _TruncatedPolicy:
    """Every turn returns a command cut off by the per-turn cap."""

    def __init__(self):
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.n += 1
        text = "```bash\nmake -j && ./run_all_the"
        msg = types.SimpleNamespace(
            content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="length")])


def test_truncated_turn_ends_the_episode(monkeypatch):
    """A finish_reason="length" turn closes the trainable sample — collection
    keeps nothing past it, so the loop stops there. The cut-off command is not
    executed; scoring still runs."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    policy = _TruncatedPolicy()
    reward, metrics = run_async(
        oaf.run_episode(policy, "m", [{"role": "system", "content": "s"}], {}, {"task_id": "t1"})
    )
    assert policy.n == 1, "the loop must stop at the truncated turn"
    actions = _FakeEnv.last_actions
    execs = [a for a in actions if a.action_type == "exec"]
    assert all("/tmp/tbench2_env_runs" in (a.command or "") for a in execs), execs
    assert any(a.action_type == "evaluate" for a in actions), "scoring still runs"
    assert reward == 1.0 and metrics["turns"] == 1
    assert metrics["end_reason"] == "length"


class _BudgetPolicy:
    """Reports total_tokens off a per-turn sequence; None when emit_usage is off."""

    def __init__(self, totals, stop_after=None, emit_usage=True, finish="stop"):
        self.totals = totals
        self.stop_after = stop_after
        self.emit_usage = emit_usage
        self.finish = finish
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.n += 1
        text = "TASK_COMPLETE" if (self.stop_after and self.n == self.stop_after) else "```bash\necho hi\n```"
        msg = types.SimpleNamespace(
            content=text, model_dump=lambda exclude_none=True: {"role": "assistant", "content": text}
        )
        usage = (
            types.SimpleNamespace(total_tokens=self.totals[self.n - 1])
            if (self.emit_usage and self.n <= len(self.totals))
            else None
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason=self.finish)], usage=usage
        )


@pytest.mark.parametrize(
    "case_id, extra_meta, totals, stop_after, emit_usage, finish, exp_calls, exp_cmds, exp_reason, exp_tokens",
    [
        # totals are cumulative session totals, as the server reports them.
        ("crosses_mid_episode", {"max_seq_len": 1000}, [400, 1000], None, True, "stop", 2, 1, "max_seq_len", 1000),
        ("crosses_on_first_turn", {"max_seq_len": 1000}, [1200], None, True, "stop", 1, 0, "max_seq_len", 1200),
        ("stays_under_budget", {"max_seq_len": 100000}, [10, 20], 2, True, "stop", 2, 1, "task_complete", 20),
        ("no_budget_in_metadata", {}, [10000, 20000], 2, True, "stop", 2, 1, "task_complete", 20000),
        # A turn with no usage must not end the episode on an earlier turn's stale count.
        ("no_usage_reported", {"max_seq_len": 1}, [0, 0], 2, False, "stop", 2, 1, "task_complete", 0),
        # The per-turn cut closes the sample, so it wins even when the budget is also crossed.
        ("length_beats_budget", {"max_seq_len": 1000}, [1200], None, True, "length", 1, 0, "length", 1200),
    ],
)
def test_budget_capped_turn_ends_the_episode(
    monkeypatch,
    case_id,
    extra_meta,
    totals,
    stop_after,
    emit_usage,
    finish,
    exp_calls,
    exp_cmds,
    exp_reason,
    exp_tokens,
):
    """--max-seq-len is enforced after the episode, in collect_samples, so the loop sees no
    finish_reason and no error when a session crosses it. It has to watch usage.total_tokens
    itself or it generates turns that sample assembly throws away."""
    monkeypatch.setattr(oaf, "load_tbench2", lambda: _CLASSES)

    async def spying_with_env(env_cls, env_url, body):
        return await body(env_cls())

    monkeypatch.setattr(oaf, "_with_env", spying_with_env)

    metadata = {"task_id": "t1", **extra_meta}
    policy = _BudgetPolicy(totals, stop_after=stop_after, emit_usage=emit_usage, finish=finish)
    reward, metrics = run_async(oaf.run_episode(policy, "m", [{"role": "system", "content": "s"}], {}, metadata))

    assert policy.n == exp_calls
    cmds = [
        a.command
        for a in _FakeEnv.last_actions
        if a.action_type == "exec" and "/tmp/tbench2_env_runs" not in (a.command or "")
    ]
    assert len(cmds) == exp_cmds, "a turn stopped on the budget must not execute its command"
    assert any(a.action_type == "evaluate" for a in _FakeEnv.last_actions), "scoring still runs"
    assert reward == 1.0 and metrics["turns"] == exp_calls
    assert metrics["end_reason"] == exp_reason
    assert metrics["session_tokens"] == exp_tokens
