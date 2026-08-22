"""Sampling context preflight (code-0815 §6.2): prompt + max_tokens must fit
the engine context limit, enforced as a typed 400 BEFORE the seq identity is
consumed — the engine itself silently truncates the decode budget of an
oversized request (near zero for an accumulated Tau context) and returns
garbage instead of failing. The limit is statically configured
(--tinker-sampling-max-context / --sglang-context-length) or discovered
lazily from the router's /get_server_info; while unknown, the preflight
admits everything (permissive, never a false reject)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
from types import SimpleNamespace

import pytest
from tests.fast.ray.tinker_frontend.fake_stack import make_backend

from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.http_server import resolve_sampling_max_context
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend, _context_limit_from_server_info

BASE = "Qwen/Qwen3-0.6B"


class InfoTransport:
    """Immediate one-token generations; server_info returns the given dict or
    raises the given exception, counting calls."""

    def __init__(self, info: dict | None = None, info_exc: Exception | None = None) -> None:
        self.info = info
        self.info_exc = info_exc
        self.generate_calls = 0
        self.info_calls = 0

    async def generate(self, payload: dict) -> dict:
        self.generate_calls += 1
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 1000, None]],
            }
        }

    async def server_info(self) -> dict:
        self.info_calls += 1
        if self.info_exc is not None:
            raise self.info_exc
        return self.info

    async def close(self) -> None:
        pass


class NoInfoTransport(InfoTransport):
    """A transport predating the server_info seam (duck-typed injectors)."""

    server_info = None


async def make_frontend(transport, max_context=None, cap=8):
    backend = make_backend()
    await backend.init()
    frontend = TinkerFrontend(
        backend,
        poll_window_s=0.2,
        poll_interval_s=0.001,
        sampling_transport=transport,
        sampling_max_active_subgenerations=cap,
        sampling_max_context=max_context,
    )
    session_id = frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]
    sampler_id = frontend.create_sampling_session(
        wire.CreateSamplingSessionRequest(session_id=session_id, sampling_session_seq_id=0, base_model=BASE)
    )["sampling_session_id"]
    return backend, frontend, sampler_id


def sample_request(sampler_id, seq=0, prompt_len=2, max_tokens=1, num_samples=1):
    return wire.SampleRequest.model_validate(
        {
            "sampling_session_id": sampler_id,
            "seq_id": seq,
            "num_samples": num_samples,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": list(range(5, 5 + prompt_len))}]},
            "sampling_params": {"max_tokens": max_tokens},
        }
    )


async def retrieve(frontend, request_id):
    return await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=request_id))


async def wait_discovery(frontend, timeout_s=2.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while frontend._context_limit is None and frontend._context_discovery_task is not None:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("context discovery never settled")
        await asyncio.sleep(0.001)


class TestConfiguredLimit:
    def test_oversized_is_a_typed_400_before_identity_and_the_boundary_is_inclusive(self):
        async def main():
            transport = InfoTransport()
            backend, frontend, sampler_id = await make_frontend(transport, max_context=64)
            try:
                with pytest.raises(ApiError) as excinfo:
                    frontend.sample(sample_request(sampler_id, seq=0, prompt_len=60, max_tokens=8))
                assert excinfo.value.status_code == 400
                assert "context limit of 64" in excinfo.value.detail
                # BEFORE identity consumption (like the num_samples cap): no
                # future record, no spent mark, no admission side effects —
                # the client can resubmit the SAME seq with a smaller budget.
                assert frontend.futures.get(f"{sampler_id}:s0") is None
                assert not frontend.samplers.get(sampler_id).is_spent(0)
                assert frontend.sampling_admission.rejected == 0
                assert transport.generate_calls == 0

                # prompt + max_tokens == limit must be ADMITTED: the engine
                # serves exactly context_len total tokens.
                fits = frontend.sample(sample_request(sampler_id, seq=0, prompt_len=56, max_tokens=8))
                assert (await retrieve(frontend, fits["request_id"]))["type"] == "sample"
                assert transport.generate_calls == 1
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_a_configured_limit_never_queries_the_transport(self):
        async def main():
            transport = InfoTransport(info={"context_length": 999})
            backend, frontend, sampler_id = await make_frontend(transport, max_context=64)
            try:
                done = frontend.sample(sample_request(sampler_id, seq=0))
                await retrieve(frontend, done["request_id"])
                assert transport.info_calls == 0
                assert frontend._context_limit == 64
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_capabilities_advertise_the_known_limit(self):
        async def main():
            backend, frontend, _ = await make_frontend(InfoTransport(), max_context=64)
            try:
                [model] = frontend.capabilities()["supported_models"]
                assert model["max_context_length"] == 64
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestDiscovery:
    def test_tighter_discovered_limit_wins(self):
        async def main():
            transport = InfoTransport(info={"context_length": 128, "max_req_input_len": 100})
            backend, frontend, sampler_id = await make_frontend(transport)
            try:
                [model] = frontend.capabilities()["supported_models"]
                assert model["max_context_length"] is None  # unknown until discovered
                done = frontend.sample(sample_request(sampler_id, seq=0))  # triggers discovery
                await wait_discovery(frontend)
                assert frontend._context_limit == 106
                await retrieve(frontend, done["request_id"])

                with pytest.raises(ApiError, match="context limit of 106"):
                    frontend.sample(sample_request(sampler_id, seq=1, prompt_len=120, max_tokens=16))
                assert transport.info_calls == 1  # discovered exactly once
                [model] = frontend.capabilities()["supported_models"]
                assert model["max_context_length"] == 106
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_null_context_length_reconstructs_from_max_req_input_len(self):
        # sglang launched WITHOUT --context-length echoes null and derives the
        # limit from the model config; the scheduler still reports
        # max_req_input_len = min(ctx - 1, kv - 1) - 5, so ctx comes back as
        # max_req_input_len + 6 (folding in a tighter KV-pool bound).
        assert _context_limit_from_server_info({"context_length": None, "max_req_input_len": 122}) == 128
        assert _context_limit_from_server_info({"context_length": 256, "max_req_input_len": 122}) == 128
        assert _context_limit_from_server_info({"context_length": True, "max_req_input_len": True}) is None
        assert _context_limit_from_server_info({"status": "ready"}) is None
        assert _context_limit_from_server_info(["not", "a", "dict"]) is None

    def test_preflight_is_permissive_until_discovery_lands(self):
        async def main():
            release = asyncio.Event()

            class SlowInfoTransport(InfoTransport):
                async def server_info(self):
                    self.info_calls += 1
                    await release.wait()
                    return {"context_length": 8}

            transport = SlowInfoTransport()
            backend, frontend, sampler_id = await make_frontend(transport)
            try:
                # The limit (8) would reject this — but it is not known yet,
                # and rejecting against a guess would break working clients.
                admitted = frontend.sample(sample_request(sampler_id, seq=0, prompt_len=100, max_tokens=50))
                assert (await retrieve(frontend, admitted["request_id"]))["type"] == "sample"
                release.set()
                await wait_discovery(frontend)
                with pytest.raises(ApiError, match="context limit of 8"):
                    frontend.sample(sample_request(sampler_id, seq=1, prompt_len=100, max_tokens=50))
            finally:
                release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_discovery_failure_disables_the_preflight_after_bounded_attempts(self):
        async def main():
            transport = InfoTransport(info_exc=RuntimeError("router not ready"))
            backend, frontend, sampler_id = await make_frontend(transport)
            try:
                for seq in range(TinkerFrontend._CONTEXT_DISCOVERY_MAX_ATTEMPTS + 2):
                    done = frontend.sample(sample_request(sampler_id, seq=seq, prompt_len=100, max_tokens=100))
                    await wait_discovery(frontend)
                    assert (await retrieve(frontend, done["request_id"]))["type"] == "sample"
                # Bounded: no per-sample hammering of a dead info endpoint.
                assert transport.info_calls == TinkerFrontend._CONTEXT_DISCOVERY_MAX_ATTEMPTS
                assert frontend._context_limit is None  # preflight stays off
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_a_transport_without_server_info_disables_the_preflight(self):
        async def main():
            transport = NoInfoTransport()
            backend, frontend, sampler_id = await make_frontend(transport)
            try:
                done = frontend.sample(sample_request(sampler_id, seq=0, prompt_len=100, max_tokens=100))
                assert (await retrieve(frontend, done["request_id"]))["type"] == "sample"
                assert frontend._context_discovery_task is None
                assert frontend._context_limit is None
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestLaunchResolution:
    def test_the_tinker_flag_wins_then_the_sglang_context_then_discovery(self):
        flagged = SimpleNamespace(tinker_sampling_max_context=32768, sglang_context_length=65536)
        assert resolve_sampling_max_context(flagged) == 32768
        deployed = SimpleNamespace(tinker_sampling_max_context=None, sglang_context_length=65536)
        assert resolve_sampling_max_context(deployed) == 65536
        bare = SimpleNamespace(tinker_sampling_max_context=None)  # no sglang attr at all
        assert resolve_sampling_max_context(bare) is None


class TestTransportDiscoveryHop:
    """The production transport's server_info against both live shapes
    (verified on H200): a bare engine answers /get_server_info directly;
    sglang-router >= 0.3 answers with router metadata and keeps the engine
    one hop away behind /workers."""

    @staticmethod
    async def _serve(app):
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical", access_log=False))
        task = asyncio.get_running_loop().create_task(server.serve())
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.005)
        return server, task, server.servers[0].sockets[0].getsockname()[1]

    def test_router_metadata_hops_to_the_first_healthy_worker(self):
        from fastapi import FastAPI

        from miles.ray.tinker_frontend.sampling import SGLangRouterSamplingTransport

        async def main():
            worker = FastAPI()

            @worker.get("/get_server_info")
            async def worker_info() -> dict:
                return {"context_length": 8192, "max_req_input_len": 8186}

            worker_server, worker_task, worker_port = await self._serve(worker)

            router = FastAPI()

            @router.get("/get_server_info")
            async def router_info() -> dict:
                # sglang-router 0.3.x: router metadata, no engine fields.
                return {"router_manager": True, "routers_count": 1, "workers_count": 1}

            @router.get("/workers")
            async def workers() -> dict:
                return {
                    "workers": [
                        {"url": "http://127.0.0.1:1", "is_healthy": False},  # skipped: unhealthy
                        {"url": f"http://127.0.0.1:{worker_port}", "is_healthy": True},
                    ]
                }

            router_server, router_task, router_port = await self._serve(router)
            transport = SGLangRouterSamplingTransport(f"http://127.0.0.1:{router_port}")
            try:
                info = await transport.server_info()
                assert info["context_length"] == 8192
            finally:
                await transport.close()
                router_server.should_exit = True
                worker_server.should_exit = True
                await asyncio.gather(router_task, worker_task, return_exceptions=True)

        asyncio.run(main())

    def test_engine_shape_answers_without_a_hop(self):
        from fastapi import FastAPI

        from miles.ray.tinker_frontend.sampling import SGLangRouterSamplingTransport

        async def main():
            engine = FastAPI()
            workers_calls = 0

            @engine.get("/get_server_info")
            async def engine_info() -> dict:
                # A launch-derived engine: context_length null but the
                # scheduler field present — must NOT trigger the hop.
                return {"context_length": None, "max_req_input_len": 40954}

            @engine.get("/workers")
            async def workers() -> dict:
                nonlocal workers_calls
                workers_calls += 1
                return {"workers": []}

            engine_server, engine_task, engine_port = await self._serve(engine)
            transport = SGLangRouterSamplingTransport(f"http://127.0.0.1:{engine_port}")
            try:
                info = await transport.server_info()
                assert info["max_req_input_len"] == 40954
                assert workers_calls == 0
            finally:
                await transport.close()
                engine_server.should_exit = True
                await asyncio.gather(engine_task, return_exceptions=True)

        asyncio.run(main())
