"""Aggregate-saturation regression with the REAL tinker SDK over live HTTP:
two SamplingClients each admit 64 concurrent requests (the SDK's per-client
ceiling — it never bounded the aggregate), against the PRODUCTION router
transport and a shared slow router that holds every generation longer than
the old implicit 10-second pool deadline.

Before the admission/transport fix this exact load was the Tau sampling
cliff: the shared httpx client's default 100-connection pool timed request
#101+ out before it ever reached the router — exactly 100/128 succeeded and
28 died as terminal, empty-message failures ("sampling failed: ") the SDK
never retries. Now the frontend 429s the overflow BEFORE the request
consumes its seq identity, the SDK retries on the same seq ids with backoff,
and all 128 complete exactly once inside the configured bound.

Skipped when the ``tinker`` wheel is not installed; install tinker==0.24.1
(pinned in tests/ci/requirements-ci-cpu.txt) to run."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="stage-a-cpu")

import asyncio
import logging

import pytest

tinker = pytest.importorskip("tinker")

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, make_backend  # noqa: E402
from tinker import types  # noqa: E402

from miles.ray.tinker_frontend.http_server import TinkerFrontendHTTPServer  # noqa: E402

API_KEY = "tml-test-key"
BASE = "Qwen/Qwen3-0.6B"
CLIENTS = 2
PER_CLIENT = 64
ROUTER_DELAY_S = 11.0
CAP = 64


class SlowRouter:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/generate")
        async def generate(request: Request) -> dict:
            payload = await request.json()
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(self.delay_s)
                return {
                    "text": "ok",
                    "meta_info": {
                        "finish_reason": {"type": "length"},
                        "output_token_logprobs": [[-0.25, int(payload["input_ids"][0]) + 10_000, None]],
                        "prompt_tokens": len(payload["input_ids"]),
                    },
                }
            finally:
                self.active -= 1

        return app


def test_aggregate_sdk_load_completes_within_the_bound_instead_of_the_pool_cliff(tmp_path):
    logging.getLogger("tinker.lib.retry_handler").setLevel(logging.CRITICAL)
    logging.getLogger("tinker.lib.api_future_impl").setLevel(logging.ERROR)

    async def main():
        router = SlowRouter(ROUTER_DELAY_S)
        router_server = uvicorn.Server(
            uvicorn.Config(router.app(), host="127.0.0.1", port=0, log_level="critical", access_log=False)
        )
        serve_task = asyncio.get_running_loop().create_task(router_server.serve())
        while not router_server.started:
            if serve_task.done():
                serve_task.result()
            await asyncio.sleep(0.005)
        router_port = router_server.servers[0].sockets[0].getsockname()[1]

        backend = make_backend(
            router_url=f"http://127.0.0.1:{router_port}",
            save_root=str(tmp_path),
            multi_lora_n_adapters=16,
            tinker_api_key=API_KEY,
        )
        await backend.init()
        FakeDriver(backend)
        server = TinkerFrontendHTTPServer(backend, host="127.0.0.1", api_port=0)
        await server.start()
        frontend = server.frontend
        assert frontend.sampling_admission.capacity == CAP
        try:
            base_url = f"http://127.0.0.1:{server.actual_api_port}"
            service = await asyncio.to_thread(tinker.ServiceClient, base_url=base_url, api_key=API_KEY)
            clients = [
                await asyncio.to_thread(service.create_sampling_client, base_model=BASE) for _ in range(CLIENTS)
            ]
            holder = service._session_holder
            session = frontend.sessions.get(holder._session_id)
            heartbeat_before = session.last_heartbeat

            params = types.SamplingParams(max_tokens=1, seed=7)
            tasks = [
                asyncio.create_task(
                    client.sample_async(
                        prompt=types.ModelInput.from_ints([1_000 + index * PER_CLIENT + i]),
                        num_samples=1,
                        sampling_params=params,
                    )
                )
                for index, client in enumerate(clients)
                for i in range(PER_CLIENT)
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            failures = [item for item in outcomes if isinstance(item, BaseException)]
            assert not failures, [f"{type(item).__name__}: {item}" for item in failures[:3]]
            assert sum(isinstance(item, types.SampleResponse) for item in outcomes) == CLIENTS * PER_CLIENT

            assert frontend.sampling_admission.rejected > 0
            assert router.max_active <= CAP
            assert router.calls == CLIENTS * PER_CLIENT

            for client in clients:
                record = frontend.samplers.get(client._sampling_session_id)
                assert record.spent_fence == PER_CLIENT - 1 and not record.spent_sparse

            for _ in range(200):
                if frontend.sampling_admission.in_use == 0 and not frontend._sample_tasks:
                    break
                await asyncio.sleep(0.01)
            assert frontend.sampling_admission.in_use == 0
            assert not frontend._sample_tasks

            assert session.last_heartbeat > heartbeat_before
            holder.close()
            await asyncio.sleep(0.05)
        finally:
            await server.stop()
            await backend.close()
            router_server.should_exit = True
            await asyncio.gather(serve_task, return_exceptions=True)

    asyncio.run(main())
