"""Sampling admission + transport bound (the Tau sampling-stall P0 fix):
global weighted fail-fast admission (429 BEFORE identity consumption), the
transport's hard in-flight invariant, permit release on every exit path
(success, failure, stale, sibling cancellation, shutdown), and typed error
classification for empty-message exceptions like httpx.PoolTimeout."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import httpx
import pytest
from tests.fast.ray.tinker_frontend.fake_stack import make_backend

from miles.ray.multi_lora.operations import OperationBackpressure
from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.sampling import SGLangRouterSamplingTransport
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend

BASE = "Qwen/Qwen3-0.6B"


class GatedTransport:
    """Counts calls; holds every generation until released."""

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, payload: dict) -> dict:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 1000, None]],
            }
        }

    async def close(self) -> None:
        pass


class FailingTransport:
    """Raises the given exception on every call; counts calls."""

    def __init__(self, exc: BaseException) -> None:
        self.calls = 0
        self.exc = exc

    async def generate(self, payload: dict) -> dict:
        self.calls += 1
        raise self.exc

    async def close(self) -> None:
        pass


async def make_frontend(transport, cap: int):
    backend = make_backend()
    await backend.init()
    frontend = TinkerFrontend(
        backend,
        poll_window_s=0.2,
        poll_interval_s=0.001,
        sampling_transport=transport,
        sampling_max_active_subgenerations=cap,
    )
    session_id = frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]
    sampler_id = frontend.create_sampling_session(
        wire.CreateSamplingSessionRequest(session_id=session_id, sampling_session_seq_id=0, base_model=BASE)
    )["sampling_session_id"]
    return backend, frontend, sampler_id


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


async def retrieve(frontend, request_id):
    return await frontend.retrieve_future(wire.FutureRetrieveRequest(request_id=request_id))


async def drain_callbacks():
    # Permit release rides task done-callbacks: one loop tick behind the
    # terminal resolution a retriever can already observe.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestWeightedAdmission:
    def test_num_samples_weighs_the_quota(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                first = frontend.sample(sample_request(sampler_id, seq=0, num_samples=3))
                assert frontend.sampling_admission.in_use == 3
                # 2 more sub-generations would exceed 4: rejected by WEIGHT,
                # not request count...
                with pytest.raises(OperationBackpressure):
                    frontend.sample(sample_request(sampler_id, seq=1, num_samples=2))
                # ...while weight 1 still fits.
                second = frontend.sample(sample_request(sampler_id, seq=2, num_samples=1))
                assert frontend.sampling_admission.in_use == 4
                transport.release.set()
                assert (await retrieve(frontend, first["request_id"]))["type"] == "sample"
                assert (await retrieve(frontend, second["request_id"]))["type"] == "sample"
                await drain_callbacks()
                assert frontend.sampling_admission.in_use == 0
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_backpressure_precedes_identity_consumption_and_the_retry_runs_once(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=1)
            try:
                first = frontend.sample(sample_request(sampler_id, seq=0))
                with pytest.raises(OperationBackpressure):
                    frontend.sample(sample_request(sampler_id, seq=1))
                # The 429 left NO trace of seq 1: no future record, no spent
                # mark — the SDK's backoff retry of the same seq id is safe.
                assert frontend.futures.get(f"{sampler_id}:s1") is None
                assert not frontend.samplers.get(sampler_id).is_spent(1)
                assert frontend.sampling_admission.rejected == 1

                transport.release.set()
                assert (await retrieve(frontend, first["request_id"]))["type"] == "sample"
                await drain_callbacks()
                retried = frontend.sample(sample_request(sampler_id, seq=1))
                assert (await retrieve(frontend, retried["request_id"]))["type"] == "sample"
                assert transport.calls == 2  # exactly once per admitted generation
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_exact_replay_bypasses_a_full_quota(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=1)
            try:
                done = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, done["request_id"])
                assert body["type"] == "sample"
                await drain_callbacks()

                transport.release.clear()
                transport.started.clear()
                frontend.sample(sample_request(sampler_id, seq=1))  # quota now full
                await transport.started.wait()
                assert frontend.sampling_admission.in_use == 1
                # An exact retry of the delivered seq 0 must replay its result
                # regardless of load: no 429, no permit, no re-generation.
                replay = frontend.sample(sample_request(sampler_id, seq=0))
                assert replay["request_id"] == done["request_id"]
                assert await retrieve(frontend, replay["request_id"]) == body
                assert frontend.sampling_admission.in_use == 1
                assert transport.calls == 2  # seq 0 once + seq 1 once
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_spent_but_evicted_seq_answers_typed_terminal_without_a_permit(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=1)
            frontend.futures.max_delivered = 1
            frontend.futures.max_expired = 1
            try:
                for seq in range(3):  # rolls seq 0's record AND tombstone off
                    done = frontend.sample(sample_request(sampler_id, seq=seq))
                    await retrieve(frontend, done["request_id"])
                await drain_callbacks()
                calls = transport.calls

                transport.release.clear()
                transport.started.clear()
                frontend.sample(sample_request(sampler_id, seq=3))  # quota now full
                await transport.started.wait()
                resent = frontend.sample(sample_request(sampler_id, seq=0))  # no 429
                body = await retrieve(frontend, resent["request_id"])
                assert body["category"] == "user" and "already executed" in body["error"]
                assert transport.calls == calls + 1  # only seq 3 ran
                assert frontend.sampling_admission.in_use == 1  # no permit taken
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_num_samples_over_capacity_is_a_nonretryable_400(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                with pytest.raises(ApiError) as excinfo:
                    frontend.sample(sample_request(sampler_id, seq=0, num_samples=5))
                # A request that can never fit must not 429 forever (the SDK
                # would retry indefinitely): typed 400, identity unconsumed.
                assert excinfo.value.status_code == 400 and "exceeds" in excinfo.value.detail
                assert not frontend.samplers.get(sampler_id).is_spent(0)
                assert frontend.sampling_admission.rejected == 0

                fits = frontend.sample(sample_request(sampler_id, seq=0, num_samples=4))
                assert (await retrieve(frontend, fits["request_id"]))["type"] == "sample"
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestPermitLifecycle:
    def test_transport_failure_releases_permits_and_names_the_exception_class(self):
        async def main():
            # str(httpx.PoolTimeout("")) is empty: without the class name the
            # old message was an undiagnosable "sampling failed: ".
            transport = FailingTransport(httpx.PoolTimeout(""))
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                failed = frontend.sample(sample_request(sampler_id, seq=0, num_samples=2))
                body = await retrieve(frontend, failed["request_id"])
                assert body["category"] == "server"
                assert "sampling failed (PoolTimeout):" in body["error"]
                await drain_callbacks()
                assert frontend.sampling_admission.in_use == 0  # weight-2 release
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_ambiguous_midbody_failure_is_terminal_and_never_reissued(self):
        async def main():
            # Whether the router executed is unknowable after a mid-body
            # reset: auto-resending could duplicate a stochastic generation.
            transport = FailingTransport(httpx.RemoteProtocolError("peer closed connection mid-body"))
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                failed = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, failed["request_id"])
                assert body["category"] == "server" and "(RemoteProtocolError)" in body["error"]
                await drain_callbacks()
                assert transport.calls == 1  # exactly one attempt, no auto-retry
                assert frontend.sampling_admission.in_use == 0
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_stale_registration_releases_the_permit_before_any_router_call(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            # Simulate an ephemeral sampler whose registration was retired.
            record = frontend.samplers.get(sampler_id)
            record.name, record.registration_id = "ghost", "r-gone"
            try:
                stale = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, stale["request_id"])
                assert body["category"] == "user" and "no longer live" in body["error"]
                await drain_callbacks()
                assert transport.calls == 0
                assert frontend.sampling_admission.in_use == 0
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_shutdown_cancellation_drains_permits_deterministically(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            future = frontend.sample(sample_request(sampler_id, seq=0, num_samples=3))
            await transport.started.wait()
            assert frontend.sampling_admission.in_use == 3
            try:
                # close() cancels AND awaits the sample tasks; the permits are
                # verifiably back by the time it returns (a task cancelled
                # before its first step still runs its done-callbacks).
                await frontend.close()
                assert frontend.sampling_admission.in_use == 0
                body = await retrieve(frontend, future["request_id"])
                assert body["category"] == "server" and "shutting down" in body["error"]
            finally:
                await backend.close()

        asyncio.run(main())


class TestTransportBound:
    def test_limits_and_timeouts_match_the_configured_bound(self):
        transport = SGLangRouterSamplingTransport("http://router:9/", max_inflight=7)
        assert transport.base_url == "http://router:9"
        assert transport.limits.max_connections == 7
        assert transport.limits.max_keepalive_connections == 7
        # pool=None is only legal because the gate bounds in-flight requests
        # to max_connections: nothing ever queues on the pool itself.
        assert transport.timeout.pool is None
        assert transport._gate._value == 7
        assert transport.timeout.connect == 10.0
        assert transport.timeout.read == 600.0
        assert transport.timeout.write == 60.0

    def test_sibling_cancellation_releases_the_gate(self):
        async def main():
            transport = SGLangRouterSamplingTransport("http://unused:9", max_inflight=1)
            started = asyncio.Event()

            class HangingClient:
                async def post(self, url, json):
                    started.set()
                    await asyncio.Event().wait()

                async def aclose(self):
                    pass

            transport._http = HangingClient()
            holder = asyncio.create_task(transport.generate({}))
            await started.wait()
            waiter = asyncio.create_task(transport.generate({}))  # queued on the gate
            await asyncio.sleep(0)
            assert transport._gate.locked()
            # The _run_sample failure path cancels siblings: one holding the
            # permit, one still waiting for it — both must leave a clean gate.
            waiter.cancel()
            holder.cancel()
            await asyncio.gather(holder, waiter, return_exceptions=True)
            assert not transport._gate.locked()
            assert transport._gate._value == 1
            await transport.close()

        asyncio.run(main())
