"""Orphan reaper + sampling observability (code-0815 §7 / §6.1).

The reaper frees bytes and capacity without permitting re-execution: a reaped
sample's seq stays spent while its parent session is live, a reaped result
leaves a fingerprint tombstone, and reaped parent sessions retire their whole
sampler namespace fail-closed. Unpolled operation futures are polled on the vanished
client's behalf, which stores the terminal bytes BEFORE acking the ledger —
the existing retention order, so the unacked-results budget drains without
ever acking an undelivered result away."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio
import logging
import time

import httpx
import pytest
from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, make_backend

from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend

BASE = "Qwen/Qwen3-0.6B"
SERVICE_LOGGER = "miles.ray.tinker_frontend.service"


class GatedTransport:
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
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def generate(self, payload: dict) -> dict:
        raise self.exc

    async def close(self) -> None:
        pass


async def make_frontend(transport, cap=4, **ttl_overrides):
    backend = make_backend()
    await backend.init()
    frontend = TinkerFrontend(
        backend,
        poll_window_s=0.2,
        poll_interval_s=0.001,
        sampling_transport=transport,
        sampling_max_active_subgenerations=cap,
        **ttl_overrides,
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
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestOrphanedSamples:
    def test_unpolled_sample_is_cancelled_typed_and_its_identity_stays_spent(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                submitted = frontend.sample(sample_request(sampler_id, seq=0, num_samples=3))
                request_id = submitted["request_id"]
                await transport.started.wait()
                assert frontend.sampling_admission.in_use == 3
                task = frontend._sample_task_by_request[request_id]

                counts = frontend.reap_once(now=time.time() + frontend.future_unpolled_ttl_s + 1)
                assert counts["cancelled_samples"] == 1
                await asyncio.gather(task, return_exceptions=True)
                await drain_callbacks()

                assert frontend.sampling_admission.in_use == 0
                assert request_id not in frontend._sample_task_by_request
                body = await retrieve(frontend, request_id)
                assert body["category"] == "server" and "orphaned" in body["error"]
                assert frontend.samplers.get(sampler_id).is_spent(0)

                calls = transport.calls
                replay = frontend.sample(sample_request(sampler_id, seq=0, num_samples=3))
                assert replay["request_id"] == request_id
                assert (await retrieve(frontend, request_id)) == body
                assert transport.calls == calls
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_prestart_orphan_cancellation_still_terminalizes_the_future(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                submitted = frontend.sample(sample_request(sampler_id, seq=0))
                request_id = submitted["request_id"]
                task = frontend._sample_task_by_request[request_id]
                counts = frontend.reap_once(now=time.time() + frontend.future_unpolled_ttl_s + 1)
                assert counts["cancelled_samples"] == 1
                await asyncio.gather(task, return_exceptions=True)
                await drain_callbacks()

                assert not transport.started.is_set()
                assert frontend.sampling_admission.in_use == 0
                assert frontend.sampling_stats.failures_by_class == {"Cancelled": 1}
                body = await retrieve(frontend, request_id)
                assert body["category"] == "server" and "orphaned" in body["error"]
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_an_actively_polled_sample_is_never_an_orphan(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                submitted = frontend.sample(sample_request(sampler_id, seq=0))
                await transport.started.wait()
                record = frontend.futures.get(submitted["request_id"])
                record.created_at -= frontend.future_unpolled_ttl_s * 10
                await retrieve(frontend, submitted["request_id"])

                counts = frontend.reap_once()
                assert counts["cancelled_samples"] == 0
                assert not frontend._sample_task_by_request[submitted["request_id"]].done()

                transport.release.set()
                assert (await retrieve(frontend, submitted["request_id"]))["type"] == "sample"
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_ttl_zero_disables_reaping(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(
                transport, cap=4, session_idle_ttl_s=0.0, future_unpolled_ttl_s=0.0, future_undelivered_ttl_s=0.0
            )
            try:
                frontend.sample(sample_request(sampler_id, seq=0))
                await transport.started.wait()
                counts = frontend.reap_once(now=time.time() + 10_000_000)
                assert counts == {"sessions": 0, "cancelled_samples": 0, "undelivered": 0}
                assert len(frontend.sessions.records) == 1
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestUndeliveredResults:
    def test_reaped_result_leaves_a_typed_tombstone_and_never_reexecutes(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4, session_idle_ttl_s=0)
            frontend.futures.max_expired = 1
            try:
                submitted = frontend.sample(sample_request(sampler_id, seq=0))
                request_id = submitted["request_id"]
                for _ in range(200):
                    record = frontend.futures.get(request_id)
                    if record.terminal is not None:
                        break
                    await asyncio.sleep(0.001)
                assert record.terminal is not None
                calls = transport.calls

                counts = frontend.reap_once(now=time.time() + frontend.future_undelivered_ttl_s + 1)
                assert counts["undelivered"] == 1
                assert frontend.futures.get(request_id) is None

                with pytest.raises(ApiError) as repoll:
                    await retrieve(frontend, request_id)
                assert repoll.value.status_code == 410 and "reaped" in repoll.value.detail
                with pytest.raises(ApiError) as resent:
                    frontend.sample(sample_request(sampler_id, seq=0))
                assert resent.value.status_code == 410
                assert transport.calls == calls

                done = frontend.sample(sample_request(sampler_id, seq=1))
                await retrieve(frontend, done["request_id"])
                await drain_callbacks()
                second = frontend.futures.get(done["request_id"])
                frontend.futures.reap_undelivered(second)
                assert frontend.futures.reaped_fingerprint(request_id) is None
                calls = transport.calls
                fenced = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, fenced["request_id"])
                assert body["category"] == "user" and "already executed" in body["error"]
                assert transport.calls == calls
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_delivered_results_stay_in_the_replay_window(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                submitted = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, submitted["request_id"])
                assert body["type"] == "sample"
                counts = frontend.reap_once(now=time.time() + 10_000_000)
                assert counts["undelivered"] == 0
                assert (await retrieve(frontend, submitted["request_id"])) == body
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestIdleSessions:
    def test_idle_session_retires_all_child_samplers_fail_closed(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                session_id = frontend.samplers.get(sampler_id).session_id
                sampler_ids = [sampler_id]
                for seq in range(1, 257):
                    sampler_ids.append(
                        frontend.create_sampling_session(
                            wire.CreateSamplingSessionRequest(
                                session_id=session_id,
                                sampling_session_seq_id=seq,
                                base_model=BASE,
                            )
                        )["sampling_session_id"]
                    )
                done = frontend.sample(sample_request(sampler_id, seq=0))
                body = await retrieve(frontend, done["request_id"])
                await drain_callbacks()

                counts = frontend.reap_once(now=time.time() + frontend.session_idle_ttl_s + 1)
                assert counts["sessions"] == 1
                with pytest.raises(ApiError) as heartbeat:
                    frontend.session_heartbeat(wire.SessionHeartbeatRequest(session_id=session_id))
                assert heartbeat.value.status_code == 404
                assert all(frontend.samplers.get(sampler) is None for sampler in sampler_ids)
                calls = transport.calls
                assert (await retrieve(frontend, done["request_id"])) == body
                with pytest.raises(ApiError) as get_sampler:
                    frontend.get_sampler(sampler_id)
                assert get_sampler.value.status_code == 404
                with pytest.raises(ApiError) as resubmit:
                    frontend.sample(sample_request(sampler_id, seq=0))
                assert resubmit.value.status_code == 404
                assert transport.calls == calls
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_a_heartbeating_session_is_not_reaped(self):
        async def main():
            backend, frontend, sampler_id = await make_frontend(GatedTransport(), cap=4)
            try:
                session_id = frontend.samplers.get(sampler_id).session_id
                frontend.sessions.get(session_id).last_heartbeat = time.time()
                assert frontend.reap_once()["sessions"] == 0
                assert frontend.sessions.get(session_id) is not None
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestVanishedClientOperations:
    def test_unpolled_operation_future_is_resolved_and_acked_then_tombstoned(self):
        async def main():
            backend = make_backend()
            await backend.init()
            driver = FakeDriver(backend)
            frontend = TinkerFrontend(backend, poll_window_s=0.5, poll_interval_s=0.002)
            session_id = frontend.create_session(wire.CreateSessionRequest(sdk_version="0.24.1"))["session_id"]
            driver_task = asyncio.create_task(driver.run(interval=0.002))
            try:
                create = await frontend.create_model(
                    wire.CreateModelRequest(
                        session_id=session_id, model_seq_id=0, base_model=BASE, lora_config=wire.LoraConfig(rank=8)
                    )
                )
                model_body = await retrieve(frontend, create["request_id"])
                model_id = model_body["model_id"]
                fb = frontend.forward_backward(
                    wire.ForwardBackwardRequest.model_validate(
                        {
                            "forward_backward_input": {
                                "data": [
                                    {
                                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                                        "loss_fn_inputs": {
                                            "target_tokens": {"data": [2, 3, 99], "dtype": "int64", "shape": [3]},
                                            "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32", "shape": [3]},
                                        },
                                    }
                                ],
                                "loss_fn": "cross_entropy",
                            },
                            "model_id": model_id,
                            "seq_id": 1,
                        }
                    )
                )
                operation_id = fb["request_id"]
                for _ in range(500):
                    view = backend.operation_view(operation_id)
                    if view is not None and view["state"] == "SUCCEEDED":
                        break
                    await asyncio.sleep(0.002)
                assert backend.operation_view(operation_id)["state"] == "SUCCEEDED"

                frontend.reap_once(now=time.time() + frontend.future_unpolled_ttl_s + 1)
                record = frontend.futures.get(operation_id)
                assert record.terminal is not None and record.terminal["type"] == "forward_backward"
                assert backend.operation_view(operation_id) is None

                assert (await retrieve(frontend, operation_id))["type"] == "forward_backward"
            finally:
                driver_task.cancel()
                await asyncio.gather(driver_task, return_exceptions=True)
                await frontend.close()
                await backend.close()

        asyncio.run(main())


class TestMaintenanceLoop:
    def test_start_is_idempotent_and_close_tears_it_down(self):
        async def main():
            backend, frontend, _ = await make_frontend(GatedTransport(), cap=4)
            try:
                frontend.start_maintenance()
                task = frontend._maintenance_task
                assert task is not None
                frontend.start_maintenance()
                assert frontend._maintenance_task is task
            finally:
                await frontend.close()
                await backend.close()
            assert frontend._maintenance_task is None
            assert task.cancelled()

        asyncio.run(main())


class TestSamplingMetrics:
    def test_admission_counters_and_high_water(self):
        async def main():
            transport = GatedTransport()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                first = frontend.sample(sample_request(sampler_id, seq=0, num_samples=3))
                admission = frontend.sampling_admission
                assert (admission.admitted, admission.admitted_weight, admission.peak_in_use) == (1, 3, 3)
                second = frontend.sample(sample_request(sampler_id, seq=1, num_samples=1))
                assert (admission.admitted, admission.admitted_weight, admission.peak_in_use) == (2, 4, 4)
                transport.release.set()
                await retrieve(frontend, first["request_id"])
                await retrieve(frontend, second["request_id"])
                await drain_callbacks()
                assert admission.in_use == 0 and admission.peak_in_use == 4
                assert frontend.sampling_stats.completed == 2
                assert frontend.sampling_stats.failed == 0
            finally:
                transport.release.set()
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_terminal_failures_are_counted_by_exception_class(self):
        async def main():
            backend, frontend, sampler_id = await make_frontend(FailingTransport(httpx.PoolTimeout("")), cap=4)
            try:
                failed = frontend.sample(sample_request(sampler_id, seq=0, num_samples=2))
                body = await retrieve(frontend, failed["request_id"])
                assert body["category"] == "server"
                await drain_callbacks()
                assert frontend.sampling_stats.failed == 1
                assert frontend.sampling_stats.failures_by_class == {"PoolTimeout": 1}
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_per_request_latencies_are_stamped(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            try:
                done = frontend.sample(sample_request(sampler_id, seq=0))
                await retrieve(frontend, done["request_id"])
                await drain_callbacks()
                record = frontend.futures.get(done["request_id"])
                assert record.first_result_at is not None and record.resolved_at is not None
                assert record.created_at <= record.first_result_at <= record.resolved_at
                stats = frontend.sampling_stats
                assert stats.first_result_count == 1
                assert stats.total_s_max >= stats.first_result_s_max >= 0.0
            finally:
                await frontend.close()
                await backend.close()

        asyncio.run(main())

    def test_summary_logs_only_when_something_changed(self):
        async def main():
            transport = GatedTransport()
            transport.release.set()
            backend, frontend, sampler_id = await make_frontend(transport, cap=4)
            logger = logging.getLogger(SERVICE_LOGGER)
            captured: list[str] = []

            class Capture(logging.Handler):
                def emit(self, record):
                    captured.append(record.getMessage())

            handler = Capture(level=logging.INFO)
            logger.addHandler(handler)
            previous_level = logger.level
            logger.setLevel(logging.INFO)
            try:
                done = frontend.sample(sample_request(sampler_id, seq=0))
                await retrieve(frontend, done["request_id"])
                await drain_callbacks()
                frontend._log_sampling_summary()
                summaries = [line for line in captured if "sampling summary" in line]
                assert len(summaries) == 1
                assert "admitted=1" in summaries[0] and "completed=1" in summaries[0]
                frontend._log_sampling_summary()
                assert len([line for line in captured if "sampling summary" in line]) == 1
            finally:
                logger.removeHandler(handler)
                logger.setLevel(previous_level)
                await frontend.close()
                await backend.close()

        asyncio.run(main())
