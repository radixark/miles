"""Frontend failure-path contracts (external adversarial review): lost-response
publish retries, sampler-identity retention, sibling cancellation, shutdown
barriers, bounded-idempotency fences, and the exact SDK patch pin — the
behaviors happy-path/equivalence tests cannot see."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-a-cpu")

import asyncio
import time

import pytest
from tests.fast.ray.tinker_frontend.fake_stack import make_backend

from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend

BASE = "Qwen/Qwen3-0.6B"


class StaticTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def generate(self, payload: dict) -> dict:
        self.calls += 1
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 1000, None]],
            }
        }

    async def close(self) -> None:
        self.closed = True


async def make_frontend(transport):
    backend = make_backend()
    await backend.init()
    frontend = TinkerFrontend(backend, poll_window_s=0.2, poll_interval_s=0.001, sampling_transport=transport)
    session_id = frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]
    return backend, frontend, session_id


async def create_ready_model(backend, frontend, session_id):
    submitted = await frontend.create_model(
        wire.CreateModelRequest(
            session_id=session_id,
            model_seq_id=0,
            base_model=BASE,
            lora_config=wire.LoraConfig(rank=8),
        )
    )
    model_id = f"{session_id}:train:0"
    model = frontend.models.get(model_id)
    backend.registry.mark_ready([model.name])
    await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=submitted["request_id"]))
    return model_id, model


def base_sampler(frontend, session_id, seq=0):
    return frontend.create_sampling_session(
        wire.CreateSamplingSessionRequest(
            session_id=session_id,
            sampling_session_seq_id=seq,
            base_model=BASE,
        )
    )["sampling_session_id"]


def sample_request(sampler_id, seq=0, num_samples=1):
    return wire.SampleRequest.model_validate(
        {
            "sampling_session_id": sampler_id,
            "seq_id": seq,
            "num_samples": num_samples,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [5, 6]}]},
            "sampling_params": {"max_tokens": 1},
        }
    )


class TestExactSdkPin:
    def test_only_the_pinned_patch_version_is_accepted(self):
        async def main():
            backend = make_backend()
            frontend = TinkerFrontend(backend, sampling_transport=StaticTransport())
            try:
                for version in ("0.24.0", "0.24.2"):
                    with pytest.raises(ApiError, match="0.24.1"):
                        frontend.create_session(wire.CreateSessionRequest(sdk_version=version))
                assert frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]
            finally:
                await frontend.close()

        asyncio.run(main())


class TestPublishRetryIdempotency:
    def test_lost_response_retry_replays_the_original_future(self):
        async def main():
            backend, frontend, session_id = await make_frontend(StaticTransport())
            try:
                model_id, _ = await create_ready_model(backend, frontend, session_id)
                first = frontend.save_weights_for_sampler(
                    wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
                )
                retry = frontend.save_weights_for_sampler(
                    wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=1)
                )
                assert retry == first
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_a_different_operation_at_the_same_seq_still_conflicts(self):
        async def main():
            backend, frontend, session_id = await make_frontend(StaticTransport())
            try:
                model_id, _ = await create_ready_model(backend, frontend, session_id)
                frontend.save_weights_for_sampler(
                    wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
                )
                with pytest.raises(ApiError) as excinfo:
                    frontend.save_weights_for_sampler(
                        wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, path="named")
                    )
                assert excinfo.value.status_code == 422
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestSamplerIdentityRetention:
    def test_publish_cannot_overwrite_an_existing_base_sampler(self):
        async def main():
            backend, frontend, session_id = await make_frontend(StaticTransport())
            try:
                sampler_id = base_sampler(frontend, session_id, seq=0)
                model_id, model = await create_ready_model(backend, frontend, session_id)
                publish = frontend.save_weights_for_sampler(
                    wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
                )
                claimed = backend.claim_ready_control_operations()["operations"]
                backend.registry.record_weight_update([model.name])
                backend.complete_control_operations({claimed[0]["operation_id"]: {"ok": True}})
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=publish["request_id"]))
                assert body["category"] == "user" and "already exists" in body["error"]
                assert frontend.samplers.get(sampler_id).name is None
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_publish_completion_after_parent_reap_cannot_recreate_a_sampler(self):
        async def main():
            backend, frontend, session_id = await make_frontend(StaticTransport())
            try:
                model_id, model = await create_ready_model(backend, frontend, session_id)
                publish = frontend.save_weights_for_sampler(
                    wire.SaveWeightsForSamplerRequest(model_id=model_id, seq_id=1, sampling_session_seq_id=0)
                )
                frontend.reap_once(now=time.time() + frontend.session_idle_ttl_s + 1)
                assert frontend.sessions.get(session_id) is None

                claimed = backend.claim_ready_control_operations()["operations"]
                backend.registry.record_weight_update([model.name])
                backend.complete_control_operations({claimed[0]["operation_id"]: {"ok": True}})
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=publish["request_id"]))
                assert body["category"] == "user" and "parent session expired" in body["error"]
                assert not frontend.samplers.records
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_sample_identity_does_not_reexecute_after_tombstone_rollover(self):
        async def main():
            transport = StaticTransport()
            backend, frontend, session_id = await make_frontend(transport)
            frontend.futures.max_delivered = 1
            frontend.futures.max_expired = 1
            try:
                sampler_id = base_sampler(frontend, session_id)
                for seq in range(3):
                    future = frontend.sample(sample_request(sampler_id, seq=seq))
                    await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=future["request_id"]))
                assert transport.calls == 3

                retried = frontend.sample(sample_request(sampler_id, seq=0))
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=retried["request_id"]))
                assert transport.calls == 3
                assert body["category"] == "user" and "already executed" in body["error"]
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class PartialFailureTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = asyncio.Event()
        self.second_cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, payload: dict) -> dict:
        index = self.calls
        self.calls += 1
        if index == 0:
            await self.second_started.wait()
            raise RuntimeError("first generation failed")
        self.second_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.second_cancelled.set()
            raise
        return await StaticTransport().generate(payload)

    async def close(self) -> None:
        pass


class BlockingTransport:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, payload: dict) -> dict:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def close(self) -> None:
        pass


class TestAsyncLifecycle:
    def test_partial_multisample_failure_cancels_sibling_generation(self):
        async def main():
            transport = PartialFailureTransport()
            backend, frontend, session_id = await make_frontend(transport)
            try:
                sampler_id = base_sampler(frontend, session_id)
                future = frontend.sample(sample_request(sampler_id, num_samples=2))
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=future["request_id"]))
                assert body["category"] == "server"
                assert transport.second_cancelled.is_set()
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_close_awaits_inflight_sample_cancellation_and_gates_new_ones(self):
        async def main():
            transport = BlockingTransport()
            backend, frontend, session_id = await make_frontend(transport)
            sampler_id = base_sampler(frontend, session_id)
            future = frontend.sample(sample_request(sampler_id))
            await transport.started.wait()
            try:
                await frontend.close()
                assert transport.cancelled.is_set()
                assert not frontend._sample_tasks
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=future["request_id"]))
                assert body["category"] == "server" and "shutting down" in body["error"]
                with pytest.raises(ApiError) as excinfo:
                    frontend.sample(sample_request(sampler_id, seq=1))
                assert excinfo.value.status_code == 503
                await frontend.close()
            finally:
                await backend.close()

        asyncio.run(main())

    def test_close_terminalizes_sample_cancelled_before_its_first_step(self):
        async def main():
            transport = BlockingTransport()
            backend, frontend, session_id = await make_frontend(transport)
            sampler_id = base_sampler(frontend, session_id)
            future = frontend.sample(sample_request(sampler_id))
            try:
                await frontend.close()
                assert not transport.started.is_set()
                assert frontend.sampling_admission.in_use == 0
                assert frontend.sampling_stats.failures_by_class == {"Cancelled": 1}
                body = await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=future["request_id"]))
                assert body["category"] == "server" and "shutting down" in body["error"]
            finally:
                await backend.close()

        asyncio.run(main())
