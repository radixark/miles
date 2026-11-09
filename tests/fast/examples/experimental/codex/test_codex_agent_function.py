import asyncio
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager

import psutil
import pytest
from tests.fast.fixtures.generation_fixtures import (
    DEFAULT_SAMPLING_PARAMS,
    MODEL_NAME,
    _call_generate,
    compute_generate_args,
    make_args,
    with_session_server,
)
from tests.fast.import_isolation_utils import REPO_ROOT

from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.types import Sample

sys.path.insert(0, str(REPO_ROOT / "examples" / "experimental" / "codex"))

_FAKE_CODEX = """#!{python}
import json, os, subprocess, sys, tomllib, urllib.error, urllib.request

config = tomllib.load(open(os.path.join(os.environ["CODEX_HOME"], "config.toml"), "rb"))
provider = config["model_providers"][config["model_provider"]]
assert provider["wire_api"] == "responses"
cwd = sys.argv[sys.argv.index("--cd") + 1]
tools = [{{"type": "function", "name": "exec_command", "parameters": {{"type": "object"}}}}]
items = [{{"type": "message", "role": "user", "content": [{{"type": "input_text", "text": sys.argv[-1]}}]}}]
print(json.dumps({{"type": "turn.started"}}), flush=True)
while True:
    body = json.dumps({{"model": config["model"], "input": items, "tools": tools, "store": False}}).encode()
    headers = {{"content-type": "application/json", "authorization": "Bearer " + os.environ[provider["env_key"]]}}
    request = urllib.request.Request(provider["base_url"] + "/responses", body, headers)
    try:
        output = json.load(urllib.request.urlopen(request, timeout=60))["output"]
    except urllib.error.HTTPError as error:
        print(json.dumps({{"type": "turn.failed", "error": {{"message": f"unexpected status {{error.code}}"}}}}), flush=True)
        sys.exit(1)
    items += output
    calls = [item for item in output if item["type"] == "function_call"]
    if not calls:
        break
    for call in calls:
        command = json.loads(call["arguments"])["cmd"]
        done = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        items.append({{"type": "function_call_output", "call_id": call["call_id"], "output": done.stdout}})
print(json.dumps({{"type": "turn.completed"}}), flush=True)
"""

_REAL_CODEX = shutil.which("codex")
_SLEEP_SECONDS = "47.2807"


def _tool_call(command: str) -> str:
    return f'<tool_call>\n{{"name": "exec_command", "arguments": {{"cmd": "{command}"}}}}\n</tool_call>'


def _write_42(prompt: str) -> ProcessResult:
    if "<tool_response>" in prompt:
        return ProcessResult(text="done", finish_reason="stop")
    return ProcessResult(text=_tool_call("echo 42 > answer.txt"), finish_reason="stop")


def _sleep(prompt: str) -> ProcessResult:
    return ProcessResult(text=_tool_call(f"sleep {_SLEEP_SECONDS}"), finish_reason="stop")


def _engine_error(prompt: str) -> ProcessResult:
    raise RuntimeError("engine failure")


@pytest.fixture(params=["fake", "real"])
def codex(request, tmp_path, monkeypatch):
    if request.param == "real" and _REAL_CODEX is None:
        pytest.skip("codex is not installed")
    if request.param == "fake":
        fake = tmp_path / "bin" / "codex"
        fake.parent.mkdir()
        fake.write_text(_FAKE_CODEX.format(python=sys.executable))
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    return request.param


@contextmanager
def _codex_env(process_fn, *, mode: str = "inline", latency: float = 0.0):
    SingletonMeta.clear_all_instances()
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn, latency=latency) as engine:
        port = find_available_port(31000)
        args = make_args(
            variant="agentic_tool_call",
            router_port=port,
            extra_argv=["--custom-agent-function-path", "codex_agent_function.run"]
            + ["--custom-agent-function-mode", mode],
        )
        with with_session_server(engine.url, args, port=port):
            yield compute_generate_args(args)
    SingletonMeta.clear_all_instances()


def _sample(marker: str, *, answer: str = "42", timeout_s: float = 60) -> Sample:
    prompt = [{"role": "user", "content": f"Write 42 into answer.txt. {marker}"}]
    return Sample(prompt=prompt, metadata={"answer": answer, "timeout_s": timeout_s}, index=0)


def _generate(args, sample: Sample) -> list[Sample]:
    return run(_call_generate(args, sample, DEFAULT_SAMPLING_PARAMS, variant="agentic_tool_call"))


@pytest.mark.parametrize(("answer", "reward"), [("42", 1.0), ("43", 0.0)])
def test_episode_through_responses_route_is_scored(codex, answer, reward):
    with _codex_env(_write_42) as args:
        (sample,) = _generate(args, _sample(uuid.uuid4().hex, answer=answer))
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == reward
    assert sample.metadata["exit_status"] == "Submitted"
    assert "exec_command" in sample.response


def test_engine_failure_discards_the_episode(codex):
    with _codex_env(_engine_error) as args:
        (sample,) = _generate(args, _sample(uuid.uuid4().hex))
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["exit_status"] == "CodexTurnFailed"


def test_wall_clock_cap_scores_zero(codex):
    with _codex_env(_sleep) as args:
        (sample,) = _generate(args, _sample(uuid.uuid4().hex, timeout_s=3))
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["exit_status"] == "TimeLimitExceeded"
    if codex == "real":
        assert _sleeping_tools() == []


def _sleeping_tools() -> list[psutil.Process]:
    procs = psutil.process_iter(["cmdline"])
    return [proc for proc in procs if any(_SLEEP_SECONDS in part for part in proc.info["cmdline"] or [])]


def _codex_processes(marker: str) -> list[psutil.Process]:
    found = []
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info["cmdline"] or []
        if "exec" in cmdline and any(marker in part for part in cmdline):
            found.append(proc)
    return found


async def _cancel_during_model_call(args, marker: str) -> tuple[list[psutil.Process], list[psutil.Process]]:
    task = asyncio.create_task(
        _call_generate(args, _sample(marker), DEFAULT_SAMPLING_PARAMS, variant="agentic_tool_call")
    )
    deadline = time.monotonic() + 30
    while not (procs := _codex_processes(marker)) and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    await asyncio.sleep(1)
    task.cancel()
    _, alive = await asyncio.to_thread(psutil.wait_procs, procs, timeout=10)
    with pytest.raises(asyncio.CancelledError):
        await task
    return procs, alive


@pytest.mark.parametrize("mode", ["inline", "subproc"])
def test_cancel_stops_codex_mid_model_call(codex, mode):
    marker = uuid.uuid4().hex
    with _codex_env(_write_42, mode=mode, latency=20) as args:
        procs, alive = run(_cancel_during_model_call(args, marker))
    assert procs
    assert alive == []


@pytest.mark.skipif(_REAL_CODEX is None, reason="codex is not installed")
def test_unreachable_model_discards_at_the_cap():
    import codex_agent_function

    from miles.rollout.agentic.agent_function import InfraAbort

    base_url = f"http://127.0.0.1:{find_available_port(39000)}/sessions/none"
    with pytest.raises(InfraAbort) as error:
        run(codex_agent_function.run(base_url, "hi", {}, {"answer": "42", "timeout_s": 15}))
    assert error.value.exit_status == "ModelUnreachable"
