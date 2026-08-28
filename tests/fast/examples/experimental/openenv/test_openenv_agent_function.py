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

from miles.rollout.agent_function import InfraAbort


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
    assert metrics["no_verdict_reason"] == "non_canonical_verifier"
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


# --- training wrapper: failure semantics ------------------------------------


class _FakeAsyncOpenAI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url

    async def close(self):
        pass


def _episode(reward, metrics=None):
    async def run_episode_fn(policy, model_name, messages, request_kwargs, metadata):
        return reward, metrics or {"turns": 1}

    return run_episode_fn


def _train(monkeypatch, run_episode_fn):
    monkeypatch.setattr(oaf, "AsyncOpenAI", _FakeAsyncOpenAI)
    return run_async(
        oaf.run_for_training("http://t:1/sessions/s", [{"role": "user", "content": "q"}], {}, {}, run_episode_fn)
    )


def test_training_wrapper_returns_the_verdict(monkeypatch):
    result = _train(monkeypatch, _episode(1.0, {"turns": 3}))
    assert result == {"reward": 1.0, "exit_status": "Submitted", "eval_report": {}, "agent_metrics": {"turns": 3}}


def test_training_wrapper_scores_a_timeout_zero(monkeypatch):
    """The policy may be what is stalling, so a wall-clock overrun is a negative sample, not a discarded one."""

    async def slow(policy, model_name, messages, request_kwargs, metadata):
        await asyncio.sleep(10)

    monkeypatch.setattr(oaf, "_MAX_ROLLOUT_TIME_S", 0.01)
    result = _train(monkeypatch, slow)
    assert result["reward"] == 0.0
    assert result["exit_status"] == "TimeLimitExceeded"


def test_training_wrapper_scores_an_env_failure_zero(monkeypatch):
    """An env that broke mid-episode may have been broken by the agent (it is root in there)."""

    async def broken(policy, model_name, messages, request_kwargs, metadata):
        raise RuntimeError("websocket closed")

    result = _train(monkeypatch, broken)
    assert result["reward"] == 0.0
    assert result["exit_status"] == "AgentError"


def test_training_wrapper_scores_a_verifier_error_zero(monkeypatch):
    result = _train(monkeypatch, _episode(None, {"turns": 2, "no_verdict_reason": "eval_error"}))
    assert result["reward"] == 0.0
    assert result["exit_status"] == "VerifierError"


def test_training_wrapper_discards_a_non_canonical_verifier(monkeypatch):
    """A server without the test.sh contract is a deployment problem, not the policy's: discard, do not score 0."""
    with pytest.raises(InfraAbort) as excinfo:
        _train(monkeypatch, _episode(None, {"turns": 2, "no_verdict_reason": "non_canonical_verifier"}))
    assert excinfo.value.exit_status == "NonCanonicalVerifier"


def test_training_wrapper_lets_infra_abort_through(monkeypatch):
    """A sandbox that could not be created (raised by the backend) is not caught as an agent error."""

    async def no_sandbox(policy, model_name, messages, request_kwargs, metadata):
        raise InfraAbort("SandboxUnavailable")

    with pytest.raises(InfraAbort):
        _train(monkeypatch, no_sandbox)
