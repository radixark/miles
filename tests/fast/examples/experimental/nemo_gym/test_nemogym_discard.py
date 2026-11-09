import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import nemogym_agent_function
import nemogym_generate
import pytest
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.rollout.base_types import RolloutFnOutput
from miles.utils.test_utils.mock_sglang_server import ProcessResult

_SUBMITTED = {"reward": 1.0, "metadata": {}}
_HANG = "hang"
_HTTP_500 = 500


def _config() -> RolloutEnvConfig:
    argv = extra_argv_for_variant(
        "agentic_tool_call", custom_agent_function_path="nemogym_agent_function.run"
    )
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV
        + argv
        + [
            "--rollout-batch-size",
            "1",
            "--n-samples-per-prompt",
            "2",
            "--custom-rm-path",
            "nemogym_generate.reward_func",
        ],
        data_rows=[{"input": "What is 1+7?", "label": "8"}],
    )


@contextmanager
def _nemo_gym_server(first_outcome) -> Iterator[str]:
    outcomes = [first_outcome]
    loop = asyncio.new_event_loop()
    started = threading.Event()
    address = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        headers = await reader.readuntil(b"\r\n\r\n")
        length = next(
            int(line.split(b":", 1)[1])
            for line in headers.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        payload = json.loads(await reader.readexactly(length))
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(
                f"{payload['policy_base_url']}/chat/completions",
                json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
            )
        outcome = outcomes.pop() if outcomes else _SUBMITTED
        if outcome == _HANG:
            await asyncio.sleep(3600)
        elif outcome == _HTTP_500:
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
        else:
            body = json.dumps(outcome).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        await writer.drain()
        writer.close()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        server = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", 0))
        address["url"] = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    started.wait()
    try:
        yield address["url"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()


def _rollout(rollout_env) -> RolloutFnOutput:
    rollout_env.mock_server.process_fn = lambda prompt: ProcessResult(text="TASK_COMPLETE", finish_reason="stop")
    return load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train")


def _kept_rewards(out: RolloutFnOutput) -> list[float]:
    return sorted(sample.reward for group in out.samples for sub in group for sample in sub)


def _drops(out: RolloutFnOutput) -> dict[str, int]:
    return {key: count for key, count in out.metrics.items() if "dynamic_filter/drop_" in key}


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
def test_failed_call_drops_the_group(rollout_env, monkeypatch):
    with _nemo_gym_server(_HTTP_500) as url:
        monkeypatch.setenv("NEMO_GYM_URL", url)
        out = _rollout(rollout_env)

    assert _kept_rewards(out) == [1.0, 1.0]
    assert _drops(out) == {"rollout/dynamic_filter/drop_group_has_aborted_NemoGymFailed": 1}


@pytest.mark.parametrize("rollout_env", [pytest.param(_config(), id="rollout")], indirect=True)
def test_timeout_stays_scored_zero(rollout_env, monkeypatch):
    with _nemo_gym_server(_HANG) as url:
        monkeypatch.setenv("NEMO_GYM_URL", url)
        monkeypatch.setenv("NEMO_GYM_RUN_TIMEOUT", "3")
        out = _rollout(rollout_env)

    assert _kept_rewards(out) == [0.0, 1.0]
    assert _drops(out) == {}
