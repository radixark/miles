"""TinkerFrontend against a real backend + fake driver: the future protocol,
seq->ordinal mapping (incl. out-of-order chunk arrival and rejected-seq
contiguity), idempotent retries, checkpoints, publish->sample, and fences."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-a-cpu")

import asyncio

import pytest
from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, FakeRouter, make_backend

from miles.ray.multi_lora.config import AdapterRunConfig
from miles.ray.multi_lora.operations import OperationBackpressure
from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend

BASE = "Qwen/Qwen3-0.6B"


class Stack:
    def __init__(self, frontend, driver, router):
        self.frontend = frontend
        self.driver = driver
        self.router = router
        self.session_id = frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]

    async def create_model(self, model_seq_id=0, rank=8, **lora_overrides):
        request = wire.CreateModelRequest(
            session_id=self.session_id,
            model_seq_id=model_seq_id,
            base_model=BASE,
            lora_config=wire.LoraConfig(rank=rank, **lora_overrides),
        )
        future = await self.frontend.create_model(request)
        body = await self.retrieve(future["request_id"])
        assert body == {"type": "create_model", "model_id": f"{self.session_id}:train:{model_seq_id}"}
        return body["model_id"]

    async def retrieve(self, request_id):
        return await self.frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=request_id))

    def fb_request(self, model_id, seq_id, tokens=(1, 2, 3), weights=(0.0, 1.0, 1.0), targets=None):
        targets = targets if targets is not None else list(tokens[1:]) + [99]
        return wire.ForwardBackwardRequest.model_validate(
            {
                "forward_backward_input": {
                    "data": [
                        {
                            "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(tokens)}]},
                            "loss_fn_inputs": {
                                "target_tokens": {"data": targets, "dtype": "int64", "shape": [len(targets)]},
                                "weights": {"data": list(weights), "dtype": "float32", "shape": [len(weights)]},
                            },
                        }
                    ],
                    "loss_fn": "cross_entropy",
                },
                "model_id": model_id,
                "seq_id": seq_id,
            }
        )

    def optim_request(self, model_id, seq_id, lr=1e-4):
        return wire.OptimStepRequest.model_validate(
            {"adam_params": {"learning_rate": lr}, "model_id": model_id, "seq_id": seq_id}
        )


class RouterSamplingTransport:
    """SamplingTransport-shaped fake: the tests exercise the REAL transport
    seam (no method monkeypatching), routing /generate to the FakeRouter."""

    def __init__(self, router):
        self.router = router
        self.closed = False

    async def generate(self, payload: dict) -> dict:
        self.router.requests.append(payload)
        return self.router.response_for(payload)

    async def close(self) -> None:
        self.closed = True


def run(scenario, poll_window_s=5.0, **backend_overrides):
    async def main():
        router = FakeRouter()
        backend = make_backend(**backend_overrides)
        await backend.init()
        driver = FakeDriver(backend)
        frontend = TinkerFrontend(
            backend,
            poll_window_s=poll_window_s,
            poll_interval_s=0.002,
            sampling_transport=RouterSamplingTransport(router),
        )
        stack = Stack(frontend, driver, router)
        driver_task = asyncio.create_task(driver.run(interval=0.002))
        try:
            await asyncio.wait_for(scenario(stack), timeout=30)
        finally:
            driver_task.cancel()
            await asyncio.gather(driver_task, return_exceptions=True)
            await frontend.close()
            await backend.close()

    asyncio.run(main())


class TestTrainingChain:
    def test_out_of_order_chunks_then_optim(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            # The SDK posts the first chunk LAST: submit seq 2 before seq 1,
            # and the optim (seq 3) before either result is retrieved.
            fb2 = stack.frontend.forward_backward(stack.fb_request(model_id, 2, tokens=(5, 6, 7)))
            fb1 = stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            optim = stack.frontend.optim_step(stack.optim_request(model_id, 3))
            body1 = await stack.retrieve(fb1["request_id"])
            body2 = await stack.retrieve(fb2["request_id"])
            body3 = await stack.retrieve(optim["request_id"])
            for body in (body1, body2):
                (row,) = [output["logprobs"]["data"] for output in body["loss_fn_outputs"]]
                assert row == [-0.5, -0.5, -0.5]  # step clock 0 at execution
                assert body["metrics"]["loss:sum"] == pytest.approx(1.0)
                assert body["metrics"]["unmasked_tokens:sum"] == pytest.approx(3.0)
            assert body3 == {"type": "optim_step", "metrics": {"grad_norm": 0.125, "learning_rate": 1e-4}}
            # The optim step moved the weights: same payload, new logprobs.
            fb4 = stack.frontend.forward_backward(stack.fb_request(model_id, 4))
            body4 = await stack.retrieve(fb4["request_id"])
            assert body4["loss_fn_outputs"][0]["logprobs"]["data"] == [-0.51, -0.51, -0.51]

        run(scenario)

    def test_forward_recomputes_metrics_and_takes_no_dirty_pin(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            forward = stack.frontend.forward(
                wire.ForwardRequest.model_validate(
                    {
                        **stack.fb_request(model_id, 1).model_dump(exclude={"forward_backward_input"}),
                        "forward_input": stack.fb_request(model_id, 1).forward_backward_input.model_dump(),
                        "seq_id": 1,
                    }
                )
            )
            body = await stack.retrieve(forward["request_id"])
            assert body["metrics"]["loss:sum"] == pytest.approx(1.0)
            # No unstepped gradients: save_state right after a forward works.
            save = stack.frontend.save_weights(
                wire.SaveWeightsRequest(model_id=model_id, path="after-forward", seq_id=2)
            )
            saved = await stack.retrieve(save["request_id"])
            assert saved["type"] == "save_weights"

        run(scenario)

    def test_idempotent_retry_and_conflict(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            request = stack.fb_request(model_id, 1)
            first = stack.frontend.forward_backward(request)
            again = stack.frontend.forward_backward(request)
            assert again == first
            with pytest.raises(ApiError) as excinfo:
                stack.frontend.forward_backward(stack.fb_request(model_id, 1, tokens=(7, 8, 9)))
            assert excinfo.value.status_code == 422
            body = await stack.retrieve(first["request_id"])
            replay = await stack.retrieve(first["request_id"])
            assert replay == body

        run(scenario)

    def test_rejected_seq_still_consumes_its_ordinal(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            fb1 = stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            # seq 2 is a boundary rejection (active target not next-token).
            bad = stack.frontend.forward_backward(
                stack.fb_request(model_id, 2, tokens=(1, 2, 3), weights=(1.0, 1.0, 1.0), targets=[9, 3, 99])
            )
            fb3 = stack.frontend.forward_backward(stack.fb_request(model_id, 3))
            failed = await stack.retrieve(bad["request_id"])
            assert failed["category"] == "user" and "next input" in failed["error"]
            # seq 3 executes: the rejected ordinal did not leave a gap.
            assert (await stack.retrieve(fb3["request_id"]))["type"] == "forward_backward"
            assert (await stack.retrieve(fb1["request_id"]))["type"] == "forward_backward"

        run(scenario)

    def test_failed_chunk_poisons_the_gradient_window(self):
        # #2258 §5: one rejected chunk of a multi-chunk fb must fail the
        # window's optim_step (discard, no partial step); the consumed poison
        # resets the window for the next round.
        async def scenario(stack):
            model_id = await stack.create_model()
            good = stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            bad = stack.frontend.forward_backward(
                stack.fb_request(model_id, 2, weights=(1.0, 1.0, 1.0), targets=[9, 3, 99])
            )
            optim = stack.frontend.optim_step(stack.optim_request(model_id, 3))
            assert (await stack.retrieve(good["request_id"]))["type"] == "forward_backward"
            assert (await stack.retrieve(bad["request_id"]))["category"] == "user"
            poisoned = await stack.retrieve(optim["request_id"])
            assert poisoned["category"] == "user" and "gradient window" in poisoned["error"]
            record = stack.frontend.backend.registry.find(stack.frontend.models.get(model_id).name)
            assert record.step == 0  # the step clock never advanced

            # The discard reset the window: a clean fb+optim round succeeds.
            fb4 = stack.frontend.forward_backward(stack.fb_request(model_id, 4))
            optim5 = stack.frontend.optim_step(stack.optim_request(model_id, 5))
            assert (await stack.retrieve(fb4["request_id"]))["type"] == "forward_backward"
            assert (await stack.retrieve(optim5["request_id"]))["type"] == "optim_step"
            assert record.step == 1

        run(scenario)

    def test_backpressure_is_retryable_not_terminal(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            stack.driver.paused = True
            stack.frontend.backend.operations.max_pending = 1
            stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            with pytest.raises(OperationBackpressure):
                stack.frontend.optim_step(stack.optim_request(model_id, 2))
            stack.driver.paused = False
            # The SDK backs off and resends the identical request until admitted.
            for _ in range(500):
                try:
                    retried = stack.frontend.optim_step(stack.optim_request(model_id, 2))
                    break
                except OperationBackpressure:
                    await asyncio.sleep(0.005)
            assert (await stack.retrieve(retried["request_id"]))["type"] == "optim_step"

        run(scenario)


class TestCheckpoints:
    def test_save_load_roundtrip_mints_and_resolves_tinker_paths(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            save = stack.frontend.save_weights(wire.SaveWeightsRequest(model_id=model_id, path="ckpt-0", seq_id=1))
            saved = await stack.retrieve(save["request_id"])
            path = saved["path"]
            assert path.startswith("tinker://") and path.endswith("/weights/ckpt-0")
            info = stack.frontend.weights_info(wire.WeightsInfoRequest(tinker_path=path))
            assert info == {
                "base_model": BASE,
                "is_lora": True,
                "lora_rank": 8,
                "train_unembed": None,
                "train_mlp": None,
                "train_attn": None,
            }
            load = stack.frontend.load_weights(
                wire.LoadWeightsRequest(model_id=model_id, path=path, optimizer=True, seq_id=2)
            )
            loaded = await stack.retrieve(load["request_id"])
            assert loaded == {"type": "load_weights", "path": path, "model_id": model_id}

        run(scenario)

    def test_weights_only_load_is_a_typed_user_failure_without_a_gap(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            save = stack.frontend.save_weights(wire.SaveWeightsRequest(model_id=model_id, path="s0", seq_id=1))
            path = (await stack.retrieve(save["request_id"]))["path"]
            load = stack.frontend.load_weights(
                wire.LoadWeightsRequest(model_id=model_id, path=path, optimizer=False, seq_id=2)
            )
            failed = await stack.retrieve(load["request_id"])
            assert failed["category"] == "user" and "weights-only" in failed["error"]
            fb = stack.frontend.forward_backward(stack.fb_request(model_id, 3))
            assert (await stack.retrieve(fb["request_id"]))["type"] == "forward_backward"

        run(scenario)

    def test_ttl_is_a_typed_rejection_no_reaper_runs(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            save = stack.frontend.save_weights(
                wire.SaveWeightsRequest(model_id=model_id, path="t0", seq_id=1, ttl_seconds=3600)
            )
            failed = await stack.retrieve(save["request_id"])
            assert failed["category"] == "user" and "ttl_seconds" in failed["error"]

        run(scenario)

    def test_load_failure_redacts_the_backend_path(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            save = stack.frontend.save_weights(wire.SaveWeightsRequest(model_id=model_id, path="lost", seq_id=1))
            tinker_path = (await stack.retrieve(save["request_id"]))["path"]
            backend_path = stack.frontend.checkpoints.get(tinker_path).backend_path
            del stack.driver.saved_states[backend_path]  # the artifact vanished server-side
            load = stack.frontend.load_weights(
                wire.LoadWeightsRequest(model_id=model_id, path=tinker_path, optimizer=True, seq_id=2)
            )
            failed = await stack.retrieve(load["request_id"])
            assert failed["category"] == "user"
            assert tinker_path in failed["error"] and backend_path not in failed["error"]

        run(scenario)

    def test_overwrite_and_unknown_paths_are_typed_rejections(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            save = stack.frontend.save_weights(
                wire.SaveWeightsRequest(model_id=model_id, path="x", seq_id=1, overwrite=True)
            )
            assert "immutable" in (await stack.retrieve(save["request_id"]))["error"]
            load = stack.frontend.load_weights(
                wire.LoadWeightsRequest(model_id=model_id, path="tinker://nope/weights/x", optimizer=True, seq_id=2)
            )
            assert "unknown checkpoint" in (await stack.retrieve(load["request_id"]))["error"]
            with pytest.raises(ApiError) as excinfo:
                stack.frontend.weights_info(wire.WeightsInfoRequest(tinker_path="tinker://nope/weights/x"))
            assert excinfo.value.status_code == 404

        run(scenario)


class TestSampling:
    async def publish(self, stack, model_id, seq_id, sampling_session_seq_id):
        publish = stack.frontend.save_weights_for_sampler(
            wire.SaveWeightsForSamplerRequest(
                model_id=model_id, seq_id=seq_id, sampling_session_seq_id=sampling_session_seq_id
            )
        )
        body = await stack.retrieve(publish["request_id"])
        assert body["type"] == "save_weights_for_sampler" and body["path"] is None
        return body["sampling_session_id"]

    def sample_request(self, sampler_id, seq_id=0, num_samples=1, **params):
        return wire.SampleRequest.model_validate(
            {
                "sampling_session_id": sampler_id,
                "seq_id": seq_id,
                "num_samples": num_samples,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [5, 6]}]},
                "sampling_params": {"max_tokens": 3, **params},
            }
        )

    def test_publish_then_sample_carries_serving_identity(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            sampler_id = await self.publish(stack, model_id, seq_id=1, sampling_session_seq_id=0)
            future = stack.frontend.sample(self.sample_request(sampler_id, num_samples=2))
            body = await stack.retrieve(future["request_id"])
            assert body["type"] == "sample" and len(body["sequences"]) == 2
            assert body["sequences"][0]["tokens"] == [1000, 1001, 1002]
            request = stack.router.requests[0]
            assert request["lora_path"].startswith("__miles_adapter_")
            assert request["extra_key"].endswith(":v1")
            assert request["return_logprob"] is True
            info = stack.frontend.get_sampler(sampler_id)
            assert info["base_model"] == BASE

        run(scenario)

    def test_client_supplied_routing_identity_never_reaches_the_router(self):
        # rid/lora_path/extra_key are the server-derived serving identity: a
        # client posting them (top-level or smuggled into sampling_params)
        # must never see its values on the router payload — the wire models
        # drop unknown fields and the sglang params are rebuilt from an
        # allowlist. This test locks that construction.
        async def scenario(stack):
            model_id = await stack.create_model()
            sampler_id = await self.publish(stack, model_id, seq_id=1, sampling_session_seq_id=0)
            request = wire.SampleRequest.model_validate(
                {
                    "sampling_session_id": sampler_id,
                    "seq_id": 0,
                    "num_samples": 1,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [5, 6]}]},
                    "sampling_params": {"max_tokens": 3, "lora_path": "../../pwn", "extra_key": "x", "rid": "x"},
                    "lora_path": "../../pwn",
                    "extra_key": "hijacked",
                    "rid": "chosen-rid",
                }
            )
            future = stack.frontend.sample(request)
            body = await stack.retrieve(future["request_id"])
            assert body["type"] == "sample"
            sent = stack.router.requests[0]
            assert sent["lora_path"].startswith("__miles_adapter_")
            assert sent["extra_key"] != "hijacked" and sent["rid"] != "chosen-rid"
            assert set(sent["sampling_params"]) == {"max_new_tokens", "temperature", "top_p", "top_k"}

        run(scenario)

    def test_republish_makes_the_old_session_fail_loud(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            old = await self.publish(stack, model_id, seq_id=1, sampling_session_seq_id=0)
            await self.publish(stack, model_id, seq_id=2, sampling_session_seq_id=1)
            future = stack.frontend.sample(self.sample_request(old))
            body = await stack.retrieve(future["request_id"])
            assert body["category"] == "user" and "republished" in body["error"]

        run(scenario)

    def test_republish_mid_generation_fails_the_inflight_sample(self):
        # TOCTOU fence: the pre-dispatch version check alone would let a
        # sample straddling a republish resolve as if it came from the pinned
        # version; the post-generation re-check fails it loudly.
        async def scenario(stack):
            model_id = await stack.create_model()
            sampler_id = await self.publish(stack, model_id, seq_id=1, sampling_session_seq_id=0)
            name = stack.frontend.samplers.get(sampler_id).name

            gate = asyncio.Event()
            transport = stack.frontend.sampling_transport
            original = transport.generate

            async def delayed(payload):
                await gate.wait()
                return await original(payload)

            transport.generate = delayed
            future = stack.frontend.sample(self.sample_request(sampler_id))
            await asyncio.sleep(0.02)  # the sample task is awaiting /generate
            stack.frontend.backend.registry.record_weight_update([name])  # republish lands mid-flight
            gate.set()
            body = await stack.retrieve(future["request_id"])
            assert body["category"] == "user" and "republished while this sample was in flight" in body["error"]

        run(scenario)

    def test_named_sampler_path_is_a_typed_rejection(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            publish = stack.frontend.save_weights_for_sampler(
                wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, path="final")
            )
            body = await stack.retrieve(publish["request_id"])
            assert body["category"] == "user" and "latest-only" in body["error"]

        run(scenario)

    def test_base_model_session_and_unsupported_probes(self):
        async def scenario(stack):
            request = wire.CreateSamplingSessionRequest(
                session_id=stack.session_id, sampling_session_seq_id=0, base_model=BASE
            )
            sampler_id = stack.frontend.create_sampling_session(request)["sampling_session_id"]
            assert stack.frontend.create_sampling_session(request)["sampling_session_id"] == sampler_id
            future = stack.frontend.sample(self.sample_request(sampler_id, num_samples=2, seed=40))
            body = await stack.retrieve(future["request_id"])
            assert body["type"] == "sample"
            assert "lora_path" not in stack.router.requests[-1]
            # Deterministic yet diverse: each fanned-out sample gets seed + i.
            seeds = sorted(r["sampling_params"]["sampling_seed"] for r in stack.router.requests[-2:])
            assert seeds == [40, 41]
            calls = len(stack.router.requests)
            overflow = self.sample_request(sampler_id, seq_id=1, num_samples=2, seed=2**63 - 1)
            failed = await stack.retrieve(stack.frontend.sample(overflow)["request_id"])
            assert failed["category"] == "user" and "signed 64-bit" in failed["error"]
            assert len(stack.router.requests) == calls

            probe = self.sample_request(sampler_id, seq_id=2)
            probe.prompt_logprobs = True
            body = await stack.retrieve(stack.frontend.sample(probe)["request_id"])
            # Prompt scoring rides the same generate: one entry per prompt token, first None.
            assert body["type"] == "sample" and body["prompt_logprobs"] == [None, -0.125]
            assert stack.router.requests[-1]["logprob_start_len"] == 0
            assert all("logprob_start_len" not in r for r in stack.router.requests[:-1])

            topk_probe = self.sample_request(sampler_id, seq_id=3)
            topk_probe.topk_prompt_logprobs = 2
            failed = await stack.retrieve(stack.frontend.sample(topk_probe)["request_id"])
            assert failed["category"] == "user" and "topk_prompt_logprobs" in failed["error"]

        run(scenario)


class TestReplayExpiry:
    """Delivered-then-evicted results must answer with a typed 410 tombstone:
    the bytes are gone and re-execution would break idempotency."""

    def test_training_resubmit_after_eviction_is_410_not_conflict(self):
        async def scenario(stack):
            stack.frontend.futures.max_delivered = 1
            model_id = await stack.create_model()
            first = stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            await stack.retrieve(first["request_id"])
            second = stack.frontend.forward_backward(stack.fb_request(model_id, 2))
            await stack.retrieve(second["request_id"])  # evicts seq 1's record

            with pytest.raises(ApiError) as repoll:
                await stack.retrieve(first["request_id"])
            assert repoll.value.status_code == 410 and "already delivered" in repoll.value.detail
            # The identical re-submit must not surface as a fatal 422 conflict
            # blaming the client ("retries must be identical" — it was).
            with pytest.raises(ApiError) as resubmit:
                stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            assert resubmit.value.status_code == 410
            # A DIFFERENT payload at the spent identity is still a conflict.
            with pytest.raises(ApiError) as conflict:
                stack.frontend.forward_backward(stack.fb_request(model_id, 1, tokens=(7, 8, 9)))
            assert conflict.value.status_code == 422

        run(scenario)

    def test_sample_resubmit_after_eviction_never_regenerates(self):
        async def scenario(stack):
            stack.frontend.futures.max_delivered = 1
            model_id = await stack.create_model()
            publish = stack.frontend.save_weights_for_sampler(
                wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
            )
            sampler_id = (await stack.retrieve(publish["request_id"]))["sampling_session_id"]
            request = wire.SampleRequest.model_validate(
                {
                    "sampling_session_id": sampler_id,
                    "seq_id": 0,
                    "num_samples": 1,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [5, 6]}]},
                    "sampling_params": {"max_tokens": 3},
                }
            )
            future = stack.frontend.sample(request)
            await stack.retrieve(future["request_id"])
            generated = len(stack.router.requests)
            fb = stack.frontend.forward_backward(stack.fb_request(model_id, 2))
            await stack.retrieve(fb["request_id"])  # evicts the sample record
            with pytest.raises(ApiError) as excinfo:
                stack.frontend.sample(request)  # same seq: must NOT re-generate
            assert excinfo.value.status_code == 410
            await asyncio.sleep(0.05)
            assert len(stack.router.requests) == generated

        run(scenario)


class TestLifecycle:
    def test_unload_fences_and_resolves(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            unload = await stack.frontend.unload_model(wire.UnloadModelRequest(model_id=model_id))
            body = await stack.retrieve(unload["request_id"])
            assert body == {"type": "unload_model", "model_id": model_id}
            follow_up = stack.frontend.forward_backward(stack.fb_request(model_id, 1))
            failed = await stack.retrieve(follow_up["request_id"])
            assert failed["category"] == "user"

        run(scenario)

    def test_create_model_rejections(self):
        async def scenario(stack):
            base = wire.CreateModelRequest(
                session_id=stack.session_id, model_seq_id=0, base_model=BASE, lora_config=wire.LoraConfig(rank=8)
            )
            for broken, match in (
                (base.model_copy(update={"base_model": "other/model"}), 400),
                (base.model_copy(update={"lora_config": wire.LoraConfig(rank=8, seed=7)}), 400),
                (base.model_copy(update={"lora_config": wire.LoraConfig(rank=8, train_mlp=False)}), 400),
                (base.model_copy(update={"lora_config": None}), 400),
                (base.model_copy(update={"session_id": "sess-unknown"}), 404),
            ):
                with pytest.raises(ApiError) as excinfo:
                    await stack.frontend.create_model(broken)
                assert excinfo.value.status_code == match

        run(scenario)

    def test_unknown_future_is_410(self):
        async def scenario(stack):
            with pytest.raises(ApiError) as excinfo:
                await stack.retrieve("nope")
            assert excinfo.value.status_code == 410

        run(scenario)

    def test_stale_model_handle_never_binds_to_a_same_name_successor(self):
        # Anti-ABA: operations are pinned to (name, registration_id); after
        # the name is re-registered, the stale handle fences as a typed user
        # failure and the successor's ledger stays untouched.
        async def scenario(stack):
            model_id = await stack.create_model()
            record = stack.frontend.models.get(model_id)
            name, rid1 = record.name, record.registration_id
            unload = await stack.frontend.unload_model(wire.UnloadModelRequest(model_id=model_id))
            await stack.retrieve(unload["request_id"])
            for _ in range(500):
                if stack.frontend.backend.registry.find(name) is None:
                    break
                await asyncio.sleep(0.005)
            await stack.frontend.backend.register(name, AdapterRunConfig(rank=8))  # operator reuses the name
            rid2 = stack.frontend.backend.registry.find(name).registration_id
            assert rid2 != rid1

            submitted = stack.frontend.optim_step(stack.optim_request(model_id, 1))
            body = await stack.retrieve(submitted["request_id"])
            assert body["category"] == "user" and "fenced" in body["error"]
            assert stack.frontend.backend.operations.queue_view(name, rid2) == []

        run(scenario)

    def test_unsupported_sdk_version_is_rejected_at_bootstrap(self):
        async def scenario(stack):
            for request in (
                lambda: stack.frontend.client_config(wire.ClientConfigRequest(sdk_version="0.25.0")),
                lambda: stack.frontend.create_session(wire.CreateSessionRequest(sdk_version="0.25.0")),
                lambda: stack.frontend.create_session(wire.CreateSessionRequest()),  # unknown client
            ):
                with pytest.raises(ApiError) as excinfo:
                    request()
                assert excinfo.value.status_code == 400 and "tinker==0.24.1" in excinfo.value.detail

        run(scenario)

    def test_healthz_reports_readiness_not_liveness(self):
        async def scenario(stack):
            assert stack.frontend.health() == {"status": "ok"}  # the fake driver marked ready
            stack.frontend.backend.trainer_ready = False
            with pytest.raises(ApiError) as excinfo:
                stack.frontend.health()
            assert excinfo.value.status_code == 503
            stack.frontend.backend.mark_trainer_ready()
            assert stack.frontend.health() == {"status": "ok"}

        run(scenario)

    def test_rejected_flood_backpressures_instead_of_growing_without_bound(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            stack.driver.paused = True  # nothing drains, nothing is retrieved
            stack.frontend.backend.operations.max_unacked_results = 8
            accepted = throttled = 0
            for seq in range(1, 101):
                bad = stack.fb_request(model_id, seq, weights=(1.0, 1.0, 1.0), targets=[9, 3, 99])
                try:
                    stack.frontend.forward_backward(bad)
                    accepted += 1
                except OperationBackpressure:
                    throttled += 1
            assert accepted == 8 and throttled == 92
            assert len(stack.frontend.futures.records) <= 8 + 1  # +1: the create_model future

        run(scenario)

    def test_bootstrap_surfaces(self):
        async def scenario(stack):
            assert stack.frontend.health() == {"status": "ok"}
            capabilities = stack.frontend.capabilities()
            assert capabilities["supported_models"][0]["model_name"] == BASE
            config = stack.frontend.client_config(wire.ClientConfigRequest(sdk_version="0.24.1"))
            assert config["proto_write_fwdbwd"] is False and config["pjwt_auth_enabled"] is False
            assert stack.frontend.session_heartbeat(wire.SessionHeartbeatRequest(session_id=stack.session_id)) == {
                "type": "session_heartbeat"
            }

        run(scenario)

    def test_get_info(self):
        async def scenario(stack):
            model_id = await stack.create_model()
            info = stack.frontend.get_info(wire.GetInfoRequest(model_id=model_id))
            assert info["model_id"] == model_id and info["lora_rank"] == 8
            assert info["model_data"]["model_name"] == BASE

        run(scenario)


async def until_terminal(stack, request_id):
    while (body := await stack.retrieve(request_id)).get("type") == "try_again":
        pass
    return body


class TestCapacityQueue:
    def test_unbound_create_reports_paused_capacity_until_the_slot_frees(self):
        # Fixed residency, SDK-visible: with one trainer slot, a second
        # registration queues UNBOUND — its create future long-polls as
        # 'paused_capacity' (never an early success), its operations enqueue
        # into the ordered ledger but never execute, and only the incumbent's
        # full retirement/cleanup binds the queued registration, resolves the
        # create future, and drains the queued work.
        async def scenario(stack):
            model_a = await stack.create_model(model_seq_id=0)
            future_b = await stack.frontend.create_model(
                wire.CreateModelRequest(
                    session_id=stack.session_id,
                    model_seq_id=1,
                    base_model=BASE,
                    lora_config=wire.LoraConfig(rank=8),
                )
            )
            model_b = f"{stack.session_id}:train:1"
            paused = {"type": "try_again", "queue_state": "paused_capacity"}
            assert await stack.retrieve(future_b["request_id"]) == paused

            # The paused registration accepts operations, but nothing runs:
            # the forward_backward future stays pending, and the create future
            # still reports paused_capacity (no early create success).
            fb_b = stack.frontend.forward_backward(stack.fb_request(model_b, 1))
            assert (await stack.retrieve(fb_b["request_id"]))["type"] == "try_again"
            assert await stack.retrieve(future_b["request_id"]) == paused

            # A's retirement frees the slot: the driver binds and loads B, the
            # create future resolves, and the queued forward_backward executes.
            unload = await stack.frontend.unload_model(wire.UnloadModelRequest(model_id=model_a))
            assert await until_terminal(stack, unload["request_id"]) == {
                "type": "unload_model",
                "model_id": model_a,
            }
            assert await until_terminal(stack, future_b["request_id"]) == {
                "type": "create_model",
                "model_id": model_b,
            }
            body = await until_terminal(stack, fb_b["request_id"])
            (row,) = [output["logprobs"]["data"] for output in body["loss_fn_outputs"]]
            assert row == [-0.5, -0.5, -0.5]  # executed at B's fresh step clock

        run(scenario, poll_window_s=0.2, multi_lora_n_adapters=1)


def test_seq_to_ordinal_documented_mapping():
    # The D5 mapping is 1:1 by design; keep it explicit and grep-able.
    from miles.ray.tinker_frontend import service

    assert "ordinal = seq_id" in service.__doc__


def test_frontend_reads_the_backend_facade_only():
    """§4.2/§3.7 dependency rule (codex-rollout-fullparameter-design-0810):
    the frontend consumes projections and verbs — a facade fake needs no
    .registry, .operations, or .router_url fields. Enforced structurally:
    the service source never dereferences backend internals."""
    import inspect

    from miles.ray.tinker_frontend import service

    source = inspect.getsource(service)
    # Match dereferences of the injected backend (self.backend.<internal>),
    # not module paths: importing the OperationBackpressure TYPE from
    # miles.ray.multi_lora.operations is part of the frontend's wire
    # contract (429 + Retry-After), not a reach into backend state.
    for internal in ("self.backend.registry", "self.backend.operations", "self.backend.router_url"):
        assert internal not in source, f"frontend must not read {internal}"


def test_injected_sampling_transport_receives_the_exact_router_payload():
    """§4.6/§8.2: sampling stays frontend -> router through the injected
    transport — /asample answers with a future immediately (the transport is
    awaited by a background task), the payload matches the direct-router wire
    shape exactly, and no rollout component is ever involved."""
    import asyncio

    from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, FakeRouter, make_backend

    from miles.ray.tinker_frontend.service import TinkerFrontend

    class FakeTransport:
        def __init__(self, router):
            self.router = router
            self.payloads = []
            self.release = asyncio.Event()

        async def generate(self, payload):
            self.payloads.append(payload)
            await self.release.wait()
            return self.router.response_for(payload)

        async def close(self):
            pass

    async def main():
        router = FakeRouter()
        backend = make_backend()
        await backend.init()
        driver = FakeDriver(backend)
        transport = FakeTransport(router)
        frontend = TinkerFrontend(backend, poll_window_s=0.3, poll_interval_s=0.002, sampling_transport=transport)
        stack = Stack(frontend, driver, router)
        driver_task = asyncio.create_task(driver.run(interval=0.002))
        try:
            model_id = await stack.create_model()
            publish = frontend.save_weights_for_sampler(
                wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
            )
            publish_body = await stack.retrieve(publish["request_id"])
            sampler_id = publish_body["sampling_session_id"]

            request = wire.SampleRequest.model_validate(
                {
                    "sampling_session_id": sampler_id,
                    "seq_id": 0,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                    "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                    "num_samples": 1,
                }
            )
            future = frontend.sample(request)
            assert future["request_id"]  # the future returns IMMEDIATELY
            for _ in range(200):
                if transport.payloads:
                    break
                await asyncio.sleep(0.002)
            [payload] = transport.payloads
            # The exact direct-router wire shape: tokenized prompt, sglang
            # params, logprobs on, registration-scoped rid + cache key.
            assert payload["input_ids"] == [1, 2, 3]
            assert payload["return_logprob"] is True
            assert payload["sampling_params"]["max_new_tokens"] == 4
            assert payload["lora_path"].startswith("__miles_adapter_")
            assert payload["rid"].count("::") == 2
            transport.release.set()
            body = await stack.retrieve(future["request_id"])
            assert body["sequences"]
        finally:
            driver_task.cancel()
            await frontend.close()
            await backend.close()

    asyncio.run(main())
