import asyncio
import importlib.util
import json
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.rollout.base_types import RolloutFnOutput
from miles.utils.test_utils.mock_sglang_server import ProcessResult

REPO_ROOT = Path(__file__).resolve().parents[4]

_SUBMITTED = {"reward": 1.0, "exit_status": "Submitted", "metadata": {}}
_DROP_CONNECTION = "drop connection"
_HANG = "hang"
_HTTP_500 = 500

_agent = None
_reward_hook = None


async def _run_agent(**kwargs):
    return await _agent(**kwargs)


async def _score(args, samples, **kwargs):
    return await _reward_hook(args, samples, **kwargs)


def _load(relative_path: str, name: str) -> ModuleType:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> RolloutEnvConfig:
    argv = extra_argv_for_variant("agentic_tool_call", custom_agent_function_path=f"{__name__}._run_agent")
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + argv
        + ["--rollout-batch-size", "1", "--n-samples-per-prompt", "2", "--custom-rm-path", f"{__name__}._score"],
        data_rows=[{"input": "What is 1+7?", "label": "8"}],
    )


async def _chat_call(url: str) -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{url}/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
    return response.status_code


async def _anthropic_call(url: str) -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{url}/v1/messages",
            json={"model": "default", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        )
    if response.status_code != 200:
        await _chat_call(url)
    return response.status_code


@contextmanager
def _service(
    first_outcome, *, model_call: Callable = _chat_call, model_statuses: list[int] | None = None
) -> Iterator[str]:
    outcomes = [first_outcome]
    loop = asyncio.new_event_loop()
    started = threading.Event()
    address = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            payload = json.loads(await reader.readexactly(length))
            status = await model_call(payload["base_url"])
            if model_statuses is not None:
                model_statuses.append(status)
            outcome = outcomes.pop() if outcomes else _SUBMITTED
            if outcome == _HANG:
                await asyncio.sleep(3600)
            elif outcome == _HTTP_500:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            elif outcome != _DROP_CONNECTION:
                body = json.dumps(outcome).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
                )
            await writer.drain()
        finally:
            writer.close()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        server = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", 0))
        address["url"] = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        started.set()
        loop.run_forever()
        server.close()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(server.wait_closed())
        loop.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    started.wait()
    try:
        yield address["url"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()


def _rollout(rollout_env, agent, reward_hook) -> RolloutFnOutput:
    global _agent, _reward_hook
    _agent, _reward_hook = agent, reward_hook
    rollout_env.mock_server.process_fn = lambda prompt: ProcessResult(text="TASK_COMPLETE", finish_reason="stop")
    return load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train")


def _kept_rewards(out: RolloutFnOutput) -> list[float]:
    return [sample.reward for group in out.samples for sub in group for sample in sub]


def _drops(out: RolloutFnOutput) -> dict[str, int]:
    return {key: count for key, count in out.metrics.items() if "dynamic_filter/drop_" in key}


def _assert_group_dropped(out: RolloutFnOutput, cause: str) -> None:
    assert _kept_rewards(out) == [1.0, 1.0]
    assert _drops(out) == {f"rollout/dynamic_filter/drop_group_has_aborted_{cause}": 1}


def _assert_scored_zero(out: RolloutFnOutput) -> None:
    assert sorted(_kept_rewards(out)) == [0.0, 1.0]
    assert _drops(out) == {}


@pytest.fixture
def swe(monkeypatch):
    agent_function = _load("examples/swe-agent-harbor-docker/swe_agent_function.py", "swe_agent_function_under_test")
    reward_hook = _load("examples/swe-agent-harbor-docker/generate.py", "swe_generate_under_test").reward_func
    return agent_function, reward_hook, monkeypatch


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
@pytest.mark.parametrize(
    ("exit_status", "cause"),
    [
        ("Error: NoResult", "NoResult"),
        ("Error: RuntimeError", "RuntimeError"),
        ("Flushed", "Flushed"),
        ("TaskNotFound", "TaskNotFound"),
        ("InvalidInstanceId", "InvalidInstanceId"),
        ("InvalidSessionServerId", "InvalidSessionServerId"),
        ("ImportError", "ImportError"),
    ],
)
def test_agent_server_status_without_a_verdict_drops_the_group(rollout_env, swe, exit_status, cause):
    agent_function, reward_hook, monkeypatch = swe
    failed = {"reward": 0.0, "exit_status": exit_status, "agent_metrics": {}, "eval_report": {}}

    with _service(failed) as url:
        monkeypatch.setenv("AGENT_SERVER_URL", url)
        out = _rollout(rollout_env, agent_function.run, reward_hook)

    _assert_group_dropped(out, cause)


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
@pytest.mark.parametrize("exit_status", ["TimeLimitExceeded", "SequenceLengthLimitExceeded", "AgentError", "Unknown"])
def test_agent_server_trial_verdict_stays_scored_zero(rollout_env, swe, exit_status):
    agent_function, reward_hook, monkeypatch = swe
    verdict = {"reward": 0.0, "exit_status": exit_status, "agent_metrics": {}, "eval_report": {}}

    with _service(verdict) as url:
        monkeypatch.setenv("AGENT_SERVER_URL", url)
        out = _rollout(rollout_env, agent_function.run, reward_hook)

    _assert_scored_zero(out)


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
@pytest.mark.parametrize("outcome", [_DROP_CONNECTION, _HTTP_500])
def test_agent_server_failed_call_drops_the_group(rollout_env, swe, outcome):
    agent_function, reward_hook, monkeypatch = swe

    with _service(outcome) as url:
        monkeypatch.setenv("AGENT_SERVER_URL", url)
        out = _rollout(rollout_env, agent_function.run, reward_hook)

    _assert_group_dropped(out, "AgentServerFailed")


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
def test_agent_server_client_timeout_stays_scored_zero(rollout_env, swe):
    agent_function, reward_hook, monkeypatch = swe

    with _service(_HANG) as url:
        monkeypatch.setenv("AGENT_SERVER_URL", url)
        monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "3")
        out = _rollout(rollout_env, agent_function.run, reward_hook)

    _assert_scored_zero(out)


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
def test_claude_code_through_the_agent_server_reaches_the_session_route(rollout_env, swe):
    agent_function, reward_hook, monkeypatch = swe
    statuses: list[int] = []

    async def agent(**kwargs):
        return await agent_function.run(**{**kwargs, "metadata": {**kwargs["metadata"], "agent_name": "claude-code"}})

    with _service(_SUBMITTED, model_call=_anthropic_call, model_statuses=statuses) as url:
        monkeypatch.setenv("AGENT_SERVER_URL", url)
        _rollout(rollout_env, agent, reward_hook)

    assert statuses == [200, 200]
