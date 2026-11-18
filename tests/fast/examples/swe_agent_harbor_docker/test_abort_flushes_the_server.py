import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.misc import call_agent_abort_hook

EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "swe-agent-harbor-docker"


@pytest.mark.parametrize("mode", ["subproc", "inline"])
def test_a_rollout_abort_cancels_the_call_and_still_flushes_the_agent_server(monkeypatch, mode):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    swe_agent_function = importlib.import_module("swe_agent_function")
    args = ImmutableNamespace(
        custom_agent_function_path=_hook("swe_agent_function.run"),
        rollout_dataset_agent_functions=[],
        session_server_instances=[SimpleNamespace(instance_id="instance-1")],
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    async def abort_mid_trial():
        trial_started, release_the_trial = asyncio.Event(), asyncio.Event()
        flushed: list[dict] = []

        async def hold_the_trial(request):
            trial_started.set()
            await release_the_trial.wait()
            return web.json_response({})

        async def flush(request):
            flushed.append(await request.json())
            return web.json_response({"cancelled": 1})

        app = web.Application()
        app.add_routes([web.post("/run", hold_the_trial), web.post("/flush", flush)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        monkeypatch.setenv("AGENT_SERVER_URL", f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}")
        try:
            call = asyncio.create_task(
                call_agent_function(
                    swe_agent_function.run,
                    mode=mode,
                    abort_scope="127.0.0.1:30000",
                    base_url="http://127.0.0.1:1/sessions/s1",
                    prompt="fix it",
                    request_kwargs={},
                    metadata={"session_server_instance_id": "instance-1"},
                )
            )
            await asyncio.wait_for(trial_started.wait(), timeout=60)
            await call_agent_abort_hook(args)
            with pytest.raises(InfraAbort) as raised:
                await asyncio.wait_for(call, timeout=10)
        finally:
            release_the_trial.set()
            await runner.cleanup()
        return raised.value.exit_status, flushed

    exit_status, flushed = asyncio.run(abort_mid_trial())

    assert exit_status == "Cancelled"
    assert flushed == [{"session_server_instance_id": "instance-1"}]
