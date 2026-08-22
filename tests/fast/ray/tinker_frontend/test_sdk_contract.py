"""Contract tests: the REAL, unmodified ``tinker`` SDK (pinned wire behavior
of 0.24.1) drives the frontend over a live localhost HTTP server.

The stack is the production one minus GPUs and Ray: TinkerFrontendHTTPServer
-> TinkerFrontend -> real MultiLoraOperationBackend (registry + ledger + validation),
executed by the FakeDriver (the documented trainer verbs), sampling proxied
to a stub sglang router. The SDK is never mocked, monkeypatched, or called
below its public surface (the one exception: models.unload is a low-level
``AsyncTinker`` resource because no high-level client exposes it).

Skipped when the ``tinker`` wheel is not installed (hosted CPU CI); install
``tinker==0.24.1`` to run.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=180, suite="stage-a-cpu")

import asyncio
import threading
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")

import uvicorn  # noqa: E402
from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, FakeRouter, make_backend  # noqa: E402
from tinker import types  # noqa: E402

from miles.ray.tinker_frontend.http_server import TinkerFrontendHTTPServer  # noqa: E402

API_KEY = "tml-test-key"
BASE = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    def run(coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    router = FakeRouter()
    router_server = uvicorn.Server(
        uvicorn.Config(router.app(), host="127.0.0.1", port=0, log_level="warning", access_log=False)
    )

    router_task: dict = {}

    async def start_router():
        task = asyncio.get_running_loop().create_task(router_server.serve())
        router_task["serve"] = task
        while not router_server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        return router_server.servers[0].sockets[0].getsockname()[1]

    router_port = run(start_router())
    backend = make_backend(
        router_url=f"http://127.0.0.1:{router_port}",
        save_root=str(tmp_path_factory.mktemp("tinker-save")),
        multi_lora_n_adapters=16,
        tinker_api_key=API_KEY,
    )
    run(backend.init())
    driver = FakeDriver(backend)

    async def spawn_driver():
        return asyncio.get_running_loop().create_task(driver.run(interval=0.002))

    driver_task = run(spawn_driver())
    server = TinkerFrontendHTTPServer(backend, host="127.0.0.1", api_port=0)
    run(server.start())
    yield SimpleNamespace(
        base_url=f"http://127.0.0.1:{server.actual_api_port}",
        backend=backend,
        driver=driver,
        router=router,
        frontend=server.frontend,
        run=run,
    )

    # Teardown must AWAIT what it cancels: dropping the driver/router tasks
    # pending prints "Task was destroyed but it is pending!" and can mask
    # exactly the shutdown/task-leak bug class these tests exist to catch.
    async def stop_background_tasks():
        driver_task.cancel()
        await asyncio.gather(driver_task, return_exceptions=True)
        router_server.should_exit = True
        await asyncio.gather(router_task["serve"], return_exceptions=True)

    run(stop_background_tasks())
    run(server.stop())
    run(backend.close())
    loop.call_soon_threadsafe(loop.stop)


@pytest.fixture()
def service_client(stack):
    return tinker.ServiceClient(base_url=stack.base_url, api_key=API_KEY)


def make_datum(tokens, weights=None, targets=None):
    targets = targets if targets is not None else tokens[1:] + [99]
    weights = weights if weights is not None else [1.0] * len(tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={"target_tokens": targets, "weights": weights},
    )


class TestBootstrap:
    def test_capabilities_list_the_deployment_base_model(self, service_client):
        capabilities = service_client.get_server_capabilities()
        assert [m.model_name for m in capabilities.supported_models] == [BASE]

    def test_a_wrong_api_key_is_a_clean_auth_failure(self, stack):
        bad = tinker.ServiceClient(base_url=stack.base_url, api_key="tml-wrong-key")
        with pytest.raises(Exception, match="401|X-API-Key"):
            bad.get_server_capabilities()


class TestTrainingChain:
    def test_fb_optim_forward_chain(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        assert client.get_info().lora_rank == 8

        data = [make_datum([1, 2, 3]), make_datum([4, 5, 6, 7])]
        fb_future = client.forward_backward(data, "cross_entropy")
        optim_future = client.optim_step(types.AdamParams(learning_rate=1e-4))
        fb = fb_future.result()
        optim = optim_future.result()

        rows = [output["logprobs"].tolist() for output in fb.loss_fn_outputs]
        assert rows == [[-0.5] * 3, [-0.5] * 4]  # step clock 0 at execution
        assert fb.metrics["loss:sum"] == pytest.approx(3.5)
        assert fb.metrics["unmasked_tokens:sum"] == pytest.approx(7.0)
        assert optim.metrics["grad_norm"] == pytest.approx(0.125)

        # After the optim step the weights moved; forward sees the new step
        # and (JSON legacy /forward path) recomputed metrics come back.
        forward = client.forward([make_datum([1, 2, 3])], "cross_entropy").result()
        assert forward.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx([-0.51] * 3)
        assert forward.metrics["loss:sum"] == pytest.approx(1.53)

    def test_multi_chunk_forward_backward_posts_out_of_order(self, service_client):
        # >1024 datums forces the SDK to split into chunks and (parallel
        # chunk mode) POST the first chunk LAST: the ledger's gap buffer
        # must reorder execution and the combiner must reassemble rows.
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        count = 1030
        data = [make_datum([10, 11]) for _ in range(count)]
        result = client.forward_backward(data, "cross_entropy").result()
        assert len(result.loss_fn_outputs) == count
        assert result.metrics["unmasked_tokens:sum"] == pytest.approx(2.0 * count)

    def test_importance_sampling_and_ppo(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": [2, 3, 99],
                "logprobs": [-0.4, -0.4, -0.4],
                "advantages": [0.0, 1.0, 1.0],
            },
        )
        is_result = client.forward_backward([datum], "importance_sampling").result()
        assert "loss:sum" in is_result.metrics
        ppo_result = client.forward_backward(
            [datum], "ppo", loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        ).result()
        assert "loss:sum" in ppo_result.metrics

    def test_user_error_is_typed_and_leaves_no_gap(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        bad = make_datum([1, 2, 3], targets=[9, 3, 99])  # active non-next-token target
        with pytest.raises(tinker.RequestFailedError, match="next input"):
            client.forward_backward([bad], "cross_entropy").result()
        # The rejected seq consumed its ordinal: the run continues — but the
        # failed fb poisoned its gradient window (#2258 §5), so the window's
        # optim_step discards instead of stepping the surviving gradients.
        good = client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
        assert len(good.loss_fn_outputs) == 1
        with pytest.raises(tinker.RequestFailedError, match="gradient window"):
            client.optim_step(types.AdamParams()).result()
        # The discard reset the window: the next round steps normally.
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
        assert client.optim_step(types.AdamParams()).result().metrics["grad_norm"] == pytest.approx(0.125)

    def test_failed_chunk_never_partial_steps_the_window(self, stack, service_client):
        # The cookbook pattern: submit the optim before awaiting the fb. With
        # >1024 datums the SDK splits chunks (first chunk posted LAST); the bad
        # datum rides the second chunk, so one chunk fails while the other
        # lands. The optim_step MUST fail and the step clock MUST hold still.
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        data = [make_datum([10, 11]) for _ in range(1024)]
        data.append(make_datum([1, 2, 3], targets=[9, 3, 99]))
        fb_future = client.forward_backward(data, "cross_entropy")
        optim_future = client.optim_step(types.AdamParams(learning_rate=1e-4))
        with pytest.raises(tinker.RequestFailedError, match="next input"):
            fb_future.result()
        with pytest.raises(tinker.RequestFailedError, match="gradient window"):
            optim_future.result()
        name = client.model_id.split(":")[0]  # session id
        [record] = [
            r for n, r in stack.backend.registry.records.items() if r.config.metadata.get("session_id") == name
        ]
        assert record.step == 0

    def test_backpressure_429_retries_to_success(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)

        async def throttle():
            stack.driver.paused = True
            stack.backend.operations.max_pending = 1

        async def release():
            stack.driver.paused = False
            stack.backend.operations.max_pending = 256

        stack.run(throttle())
        try:
            fb_future = client.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
            optim_future = client.optim_step(types.AdamParams())  # 429s, SDK backs off
            stack.run(asyncio.sleep(0.2))
        finally:
            stack.run(release())
        assert len(fb_future.result().loss_fn_outputs) == 1
        assert optim_future.result().metrics["grad_norm"] == pytest.approx(0.125)


class TestCheckpoints:
    def test_save_then_resume_with_optimizer(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
        client.optim_step(types.AdamParams()).result()
        path = client.save_state("resume-me").result().path
        assert path.startswith("tinker://") and path.endswith("/weights/resume-me")

        # weights_info -> create_model -> load_weights(optimizer=True) chain.
        resumed = service_client.create_training_client_from_state_with_optimizer(path)
        assert resumed.get_info().lora_rank == 8
        result = resumed.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
        # Step clock restored to 1: the fake driver's logprobs move with it.
        assert result.loss_fn_outputs[0]["logprobs"].tolist() == pytest.approx([-0.51] * 3)

    def test_weights_only_resume_is_a_typed_rejection(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        path = client.save_state("no-optim").result().path
        with pytest.raises(tinker.RequestFailedError, match="weights-only"):
            service_client.create_training_client_from_state(path)

    def test_immutable_states_and_load_after_unload(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        client.save_state("once").result()
        with pytest.raises(tinker.RequestFailedError, match="immutable"):
            client.save_state("once").result()


class TestSampling:
    def test_publish_then_sample(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
        client.optim_step(types.AdamParams()).result()
        sampling = client.save_weights_and_get_sampling_client()
        response = sampling.sample(
            prompt=types.ModelInput.from_ints([5, 6, 7]),
            num_samples=2,
            sampling_params=types.SamplingParams(max_tokens=3, temperature=0.5, top_p=0.9),
        ).result()
        assert len(response.sequences) == 2
        for sequence in response.sequences:
            assert sequence.tokens == [1000, 1001, 1002]
            assert sequence.logprobs == [-0.25, -0.5, -0.75]
            assert sequence.stop_reason == "length"
        generated = stack.router.requests[-1]
        assert generated["lora_path"].startswith("__miles_adapter_")
        assert generated["extra_key"].endswith(":v1")
        assert generated["sampling_params"] == {
            "max_new_tokens": 3,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": -1,
        }
        assert sampling.get_base_model() == BASE

    def test_base_model_sampling_session(self, stack, service_client):
        sampling = service_client.create_sampling_client(base_model=BASE)
        response = sampling.sample(
            prompt=types.ModelInput.from_ints([8]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        assert response.sequences[0].tokens == [1000, 1001]
        assert "lora_path" not in stack.router.requests[-1]

    def test_compute_logprobs_scores_every_prompt_token(self, stack, service_client):
        sampling = service_client.create_sampling_client(base_model=BASE)
        prompt = [5, 6, 7, 8]
        logprobs = sampling.compute_logprobs(types.ModelInput.from_ints(prompt)).result()
        # Exact alignment with the router's per-position scores; position 0 has no context.
        assert logprobs == [None, -0.125, -0.25, -0.375]
        assert len(logprobs) == len(prompt)
        assert all(isinstance(lp, float) for lp in logprobs[1:])
        sent = stack.router.requests[-1]
        assert sent["input_ids"] == prompt
        assert sent["logprob_start_len"] == 0 and sent["return_logprob"] is True
        # The 0.24.1 SDK's compute_logprobs wire form is a 1-sample, 1-token generation.
        assert sent["sampling_params"]["max_new_tokens"] == 1

    def test_sample_with_prompt_logprobs_returns_both(self, service_client):
        sampling = service_client.create_sampling_client(base_model=BASE)
        response = sampling.sample(
            prompt=types.ModelInput.from_ints([5, 6, 7]),
            num_samples=2,
            sampling_params=types.SamplingParams(max_tokens=3),
            include_prompt_logprobs=True,
        ).result()
        assert len(response.sequences) == 2
        assert response.sequences[0].tokens == [1000, 1001, 1002]
        assert response.prompt_logprobs == [None, -0.125, -0.25]

    def test_topk_prompt_logprobs_is_a_typed_rejection(self, service_client):
        sampling = service_client.create_sampling_client(base_model=BASE)
        future = sampling.sample(
            prompt=types.ModelInput.from_ints([5, 6]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
            topk_prompt_logprobs=2,
        )
        with pytest.raises(tinker.RequestFailedError, match="topk_prompt_logprobs"):
            future.result()

    def test_stale_ephemeral_sampler_fails_loud_after_republish(self, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=8)
        old = client.save_weights_and_get_sampling_client()
        client.save_weights_and_get_sampling_client()  # republish supersedes
        future = old.sample(
            prompt=types.ModelInput.from_ints([5]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        )
        with pytest.raises(tinker.RequestFailedError, match="republished"):
            future.result()

    def test_oversized_context_is_a_typed_rejection_not_silent_truncation(self, stack, service_client):
        # The FakeRouter serves /get_server_info with max_req_input_len=4090
        # (context_length null, the launch-derived default): the frontend
        # reconstructs an engine context of 4096 and must reject a prompt +
        # max_tokens over it LOUDLY — the engine itself would silently clamp
        # the decode budget (the observed 65,235-token Tau prompt against a
        # 65,536 context) and return garbage.
        sampling = service_client.create_sampling_client(base_model=BASE)
        small = sampling.sample(  # triggers (and must precede) discovery
            prompt=types.ModelInput.from_ints([9]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        assert small.sequences[0].tokens == [1000, 1001]

        async def discovered():
            for _ in range(200):
                if stack.frontend._context_limit is not None:
                    return stack.frontend._context_limit
                await asyncio.sleep(0.01)
            raise TimeoutError("context discovery never landed")

        assert stack.run(discovered()) == stack.router.max_req_input_len + 6 == 4096

        with pytest.raises(Exception, match="context limit of 4096"):
            sampling.sample(
                prompt=types.ModelInput.from_ints(list(range(1, 4001))),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=2048),
            ).result()
        # The rejection consumed nothing: the same client keeps sampling.
        again = sampling.sample(
            prompt=types.ModelInput.from_ints([11]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2),
        ).result()
        assert again.sequences[0].stop_reason in ("length", "stop")


class TestUnload:
    def test_low_level_unload_retires_the_registration(self, stack, service_client):
        client = service_client.create_lora_training_client(base_model=BASE, rank=4)
        model_id = client.model_id

        async def unload_and_poll():
            from tinker._client import AsyncTinker

            low_level = AsyncTinker(base_url=stack.base_url, api_key=API_KEY)
            future = await low_level.models.unload(request=types.UnloadModelRequest(model_id=model_id))
            for _ in range(200):
                raw = await low_level.futures.with_raw_response.retrieve(
                    request=types.FutureRetrieveRequest(request_id=future.request_id)
                )
                body = await raw.json()
                if body.get("type") != "try_again":
                    return body
                await asyncio.sleep(0.02)
            raise TimeoutError("unload future never resolved")

        body = stack.run(unload_and_poll())
        assert body == {"type": "unload_model", "model_id": model_id}
        with pytest.raises(tinker.RequestFailedError):
            client.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
