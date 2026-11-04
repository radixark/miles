"""The agent-server client must not cap concurrent trials below what miles puts in flight.

Each Harbor trial holds one HTTP connection to the agent server for its whole run, so the
client's connection pool is a hard ceiling on concurrent trials.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENT_FUNCTION_SCRIPT = REPO_ROOT / "examples" / "swe-agent-harbor-docker" / "swe_agent_function.py"

# Above httpx's default pool size of 100, which is what a transport without limits falls back to.
NUM_TRIALS = 150


def _load_agent_function() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_agent_harbor_docker_agent_function", AGENT_FUNCTION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _peak_concurrent_trials(module: ModuleType, num_trials: int) -> int:
    """Start num_trials posts against a server that holds every request open; return how many arrive at once."""
    active = 0
    peak = 0
    release = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal active, peak
        while True:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            active += 1
            peak = max(peak, active)
            await release.wait()
            active -= 1
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
            await writer.drain()
            if reader.at_eof():
                break

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        posts = [
            asyncio.create_task(module._post_agent_server(f"http://127.0.0.1:{port}/run", {"i": i}))
            for i in range(num_trials)
        ]
        # Let every post that the pool admits reach the server.
        for _ in range(100):
            await asyncio.sleep(0.02)
            if peak >= num_trials:
                break
        observed = peak
        release.set()
        await asyncio.gather(*posts)
        return observed
    finally:
        await module._agent_server_client.aclose()
        server.close()
        await server.wait_closed()


def test_pool_admits_more_than_httpx_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_SERVER_MAX_CONNECTIONS", raising=False)
    module = _load_agent_function()

    assert asyncio.run(_peak_concurrent_trials(module, NUM_TRIALS)) == NUM_TRIALS


def test_max_connections_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SERVER_MAX_CONNECTIONS", "40")
    module = _load_agent_function()

    assert asyncio.run(_peak_concurrent_trials(module, NUM_TRIALS)) == 40


@pytest.mark.parametrize("value", ["forty", "0", "-5"])
def test_invalid_max_connections_falls_back_to_no_cap(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SERVER_MAX_CONNECTIONS", value)
    module = _load_agent_function()

    assert asyncio.run(_peak_concurrent_trials(module, NUM_TRIALS)) == NUM_TRIALS
