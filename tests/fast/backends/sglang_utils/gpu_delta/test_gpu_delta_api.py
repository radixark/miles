"""GPU-delta HTTP contract and tokenizer transaction coordination, without engines."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware
from sglang.utils import TypeBasedDispatcher
from miles.backends.sglang_utils.gpu_delta import integration
from miles.backends.sglang_utils.gpu_delta import io as io
from miles.backends.sglang_utils.gpu_delta import tokenizer as tokenizer
from miles.backends.sglang_utils.gpu_delta.io import (
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaInfoReqInput,
    GpuDeltaReqOutput,
    UpdateWeightsFromGpuDeltaReqInput,
)
from miles.backends.sglang_utils.gpu_delta.tokenizer import GpuDeltaCommunicator


@pytest.fixture
def http_delta():
    class CustomRoute(APIRoute):
        pass

    app = FastAPI()
    app.router.route_class = CustomRoute
    state = SimpleNamespace(tokenizer_manager=None)
    integration.register_gpu_delta_routes(app, lambda: state)
    add_api_key_middleware(app, api_key="ordinary-key", admin_api_key="admin-key")
    calls = []
    reply = {"success": True, "message": "", "participants": []}

    async def dispatch(obj):
        calls.append(obj)
        return reply

    # Global state is populated at startup, after the routes are registered.
    state.tokenizer_manager = SimpleNamespace(gpu_delta=SimpleNamespace(request=dispatch))
    with TestClient(app, headers={"Authorization": "Bearer admin-key"}) as client:
        yield client, app, calls, reply, CustomRoute


def test_delta_routes_preserve_typed_requests_auth_and_app_route_class(http_delta):
    client, app, calls, reply, route_class = http_delta
    session = {"session_id": "publication-1"}
    receipts = [
        {
            "identity": {"engine_id": "e0", "rank_id": "original"},
            "state": "APPLIED",
            "session_id": "publication-1",
            "target_version": 1,
        }
    ]
    publication = session | {
        "manifest_path": "/immutable/manifest.json",
        "manifest_sha256": "a" * 64,
        "stream_id": "run-1",
        "base_stream_id": None,
        "base_version": 0,
        "target_version": 1,
        "plan_digest": "b" * 64,
        "participants": [],
    }
    cases = [
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            publication,
        ),
        (
            "update_weights_from_gpu_delta",
            UpdateWeightsFromGpuDeltaReqInput,
            publication
            | {
                "release_state": True,
                "pause_mode": "in_place",
                "abort_all_requests": True,
                "weight_version": "actor-42",
                "validate_payload_hash": False,
            },
        ),
        ("clear_gpu_delta_state", ClearGpuDeltaStateReqInput, {}),
        ("get_gpu_delta_info", GetGpuDeltaInfoReqInput, {"engine_id": "e0"}),
    ]
    assert {route.path for route in app.routes if "gpu_delta" in route.path} == {
        "/update_weights_from_gpu_delta",
        "/clear_gpu_delta_state",
        "/get_gpu_delta_info",
    }
    for name, request_type, payload in cases:
        route = next(route for route in app.routes if route.path == f"/{name}")
        assert isinstance(route, route_class)
        assert route.methods == {"POST"}
        assert route.endpoint._auth_level is AuthLevel.ADMIN_OPTIONAL
        rejected = client.post(
            route.path,
            json=payload,
            headers={"Authorization": "Bearer ordinary-key"},
        )
        assert rejected.status_code == 401
        result = client.post(route.path, json=payload)
        assert result.status_code == 200 and result.json() == reply
        assert isinstance(calls[-1], request_type)
        received = msgpack_decode(msgpack_encode(calls[-1]))
        assert type(received) is request_type
        assert received == calls[-1]
        if request_type is UpdateWeightsFromGpuDeltaReqInput:
            assert received.validate_payload_hash is payload.get("validate_payload_hash", True)
            assert received.pause_mode == payload.get("pause_mode", "retract")
    response = GpuDeltaReqOutput(success=True, message="", participants=receipts, rid="control-reply")
    received = msgpack_decode(msgpack_encode(response))
    assert type(received) is GpuDeltaReqOutput and received == response
    assert len(calls) == len(cases)
    assert calls[0].release_state is False
    rejected = client.post("/update_weights_from_gpu_delta", json={"manifest_path": "/manifest.json"})
    assert rejected.status_code == 422 and len(calls) == len(cases)
    # The scheduler pauses only by retraction or in place; SGLang's abort mode is not one of them.
    rejected = client.post("/update_weights_from_gpu_delta", json=publication | {"pause_mode": "abort"})
    assert rejected.status_code == 422 and len(calls) == len(cases)

    # A real scheduler failure retains the same typed HTTP route and response.
    reply.update(success=False, message="delta payload checksum failed")
    result = client.post("/update_weights_from_gpu_delta", json=publication)
    assert result.status_code == 409 and result.json() == reply
    assert len(calls) == len(cases) + 1


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(tokenizer, "get_serving", lambda: SimpleNamespace(tokenizer_worker_num=1))
    sent, versions = [], []
    manager = SimpleNamespace(
        is_pause=False,
        mm_processor=None,
        is_pause_cond=asyncio.Condition(),
        _dispatch_to_scheduler=sent.append,
        _result_dispatcher=TypeBasedDispatcher([]),
        recv_from_detokenizer=SimpleNamespace(getsockopt_string=lambda _: "ipc:///gpu-delta-tokenizer-test"),
        auto_create_handle_loop=lambda: None,
        _update_weight_version_if_provided=versions.append,
    )
    integration.init_tokenizer_control(manager, 1)
    return manager.gpu_delta, sent, versions


@pytest.fixture
def publication():
    return dict(
        session_id="publication-1",
        manifest_path="/immutable/manifest.json",
        manifest_sha256="a" * 64,
        stream_id="run-1",
        base_stream_id="run-1",
        base_version=2,
        target_version=7,
        plan_digest="b" * 64,
        participants=[dict(engine_id="e0", rank_id=f"r{i}") for i in range(3)],
    )


_RESET_PHASES = (
    io.AbortGpuDeltaReqInput,
    io.ClearGpuDeltaStateReqInput,
    io.ReleaseGpuDeltaCacheReqInput,
    io.GetGpuDeltaInfoReqInput,
)


def reset_reply(obj, publication):
    state = dict(zip(_RESET_PHASES, ("ABORTED", "CLEARED", "CLEARED", "IDLE"), strict=True))[type(obj)]
    stream, version = (
        (publication["stream_id"], publication["target_version"])
        if publication.get("reverse")
        else (publication["base_stream_id"], publication["base_version"])
    )
    return {
        "success": True,
        "message": "",
        "participants": [
            dict(
                identity=rank,
                state=state,
                committed_stream_id=stream,
                stream_id=stream,
                version=version,
                plan_digest=publication["plan_digest"],
            )
            for rank in publication["participants"]
        ],
    }


def reply(control, obj, state, ranks=(0,)):
    control.manager._result_dispatcher(
        io.GpuDeltaReqOutput(
            rid=obj.rid,
            success=True,
            message="",
            participants=[
                {
                    "identity": {"engine_id": "e0", "rank_id": f"r{rank}"},
                    "session_id": obj.session_id,
                    "state": state,
                    "target_version": 7,
                    "version": 7,
                }
                for rank in ranks
            ],
        )
    )


@pytest.mark.parametrize("ranks_per_route", [2, 4], ids=["tp2", "cp2-tp2"])
def test_update_waits_for_every_rank_before_apply_and_resume(control, publication, monkeypatch, ranks_per_route):
    routes = (
        tuple(range(ranks_per_route)),
        tuple(range(ranks_per_route, 2 * ranks_per_route)),
    )
    publication["participants"] = [dict(engine_id="e0", rank_id=f"r{rank}") for rank in range(2 * ranks_per_route)]

    async def run():
        facade, _, versions = control
        setup_calls = []
        facade.manager.auto_create_handle_loop = lambda: setup_calls.append("loop")

        def serving():
            setup_calls.append("workers")
            return SimpleNamespace(tokenizer_worker_num=1)

        monkeypatch.setattr(tokenizer, "get_serving", serving)
        sent = asyncio.Queue()
        facade.communicator._send = sent.put_nowait
        integration.update_tokenizer_fan_out(facade.manager, 2)
        task = asyncio.create_task(facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication)))
        prepare = await sent.get()
        assert type(prepare) is io.PrepareGpuDeltaReqInput
        assert prepare.validate_payload_hash is True
        assert not facade.manager.is_pause
        reply(facade, prepare, "PREPARED", ranks=routes[0])
        await asyncio.sleep(0)
        assert sent.empty() and not task.done()
        # A duplicate cannot stand in for a slower rank's completion.
        reply(facade, prepare, "PREPARED", ranks=routes[0])
        await asyncio.sleep(0)
        assert sent.empty() and not task.done()
        reply(facade, prepare, "PREPARED", ranks=routes[1])
        apply = await sent.get()
        assert type(apply) is io.ApplyGpuDeltaReqInput
        assert facade.manager.is_pause
        # A timeout can race the original HTTP call. Reconciliation waits for its
        # complete transaction, including resume on the last rank.
        lookup = asyncio.create_task(
            facade.request(io.GetGpuDeltaInfoReqInput(engine_id="e0", session_id=publication["session_id"]))
        )
        reply(facade, apply, "APPLIED", ranks=routes[0])
        await asyncio.sleep(0)
        assert sent.empty() and facade.manager.is_pause
        reply(facade, apply, "APPLIED", ranks=routes[1])
        resume = await sent.get()
        assert type(resume) is io.ResumeGpuDeltaReqInput
        assert facade.manager.is_pause and versions == []
        reply(facade, resume, "RESUMED", ranks=routes[0])
        await asyncio.sleep(0)
        assert not task.done() and facade.manager.is_pause and versions == []
        reply(facade, resume, "RESUMED", ranks=routes[1])
        result = await task
        assert result["success"] and not facade.manager.is_pause
        assert versions == ["7"]
        info = await sent.get()
        assert isinstance(info, io.GetGpuDeltaInfoReqInput)
        for ranks in routes:
            reply(facade, info, "RESUMED", ranks=ranks)
        assert (await lookup)["activation"] == result
        assert [r["state"] for r in result["applied_participants"]] == ["APPLIED"] * (2 * ranks_per_route)
        assert [r["state"] for r in result["participants"]] == ["RESUMED"] * (2 * ranks_per_route)
        assert setup_calls == ["loop", "workers"] * 2

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    "failure,release_state,pause_mode,abort_all_requests,weight_version,pause_at",
    [
        (None, False, "retract", False, None, None),
        (None, True, "in_place", True, "0", None),
        (None, False, "in_place", False, "actor-42", None),
        (None, True, "retract", False, None, "entry"),
        (None, False, "in_place", False, None, "prepare"),
        ("prepare", True, "retract", False, None, None),
        ("apply", True, "retract", False, None, None),
        ("resume", True, "retract", False, None, None),
        ("uncertain_apply", True, "retract", False, None, None),
        ("clear", True, "retract", False, None, None),
        ("release", True, "retract", False, None, None),
    ],
)
def test_update_preserves_receipts_options_and_failure_boundaries(
    control,
    publication,
    monkeypatch,
    failure,
    release_state,
    pause_mode,
    abort_all_requests,
    weight_version,
    pause_at,
):
    # The same canonical input contract supports fresh-base and rolling updates.
    publication["validate_payload_hash"] = False
    if release_state:
        publication.update(base_stream_id=None, base_version=0)
    now = [0.0]
    monkeypatch.setattr(tokenizer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    serving_version = "7" if weight_version is None else weight_version
    calls = []

    async def run():
        facade, _, versions = control
        facade.manager.is_pause = pause_at == "entry"
        cache_clears, aborted = [], []
        facade.manager.mm_processor = SimpleNamespace(clear_preprocess_cache=lambda: cache_clears.append("clear"))

        def abort(abort_all):
            assert abort_all and facade.manager.is_pause_cond.locked()
            aborted.append("abort")

        facade.manager.abort_request = abort

        async def request(obj):
            calls.append(obj)
            if failure == "prepare" and isinstance(obj, _RESET_PHASES):
                return reset_reply(obj, publication)
            if type(obj) is io.PrepareGpuDeltaReqInput:
                assert not facade.manager.is_pause_cond.locked()
                assert facade.manager.is_pause is (pause_at == "entry")
                assert obj == io.PrepareGpuDeltaReqInput(**publication, http_worker_ipc=facade.reply_endpoint)
                if pause_at == "prepare":
                    facade.manager.is_pause = True
                state, phase, duration = "PREPARED", "prepare", 5
            elif isinstance(obj, io.ApplyGpuDeltaReqInput):
                assert facade.manager.is_pause
                assert obj.pause_mode == pause_mode
                assert aborted == (["abort"] if abort_all_requests else [])
                if failure == "uncertain_apply":
                    raise OSError("lost apply acknowledgment")
                state, phase, duration = "APPLIED", "apply", 7
            elif isinstance(obj, io.ResumeGpuDeltaReqInput):
                assert facade.manager.is_pause
                assert obj.keep_pause is (pause_at is not None)
                assert obj.weight_version == weight_version
                state, phase, duration = "RESUMED", "resume", 11
            elif isinstance(obj, io.AbortGpuDeltaReqInput):
                state, phase, duration = "ABORTED", "abort", 0
            else:
                assert facade.manager.is_pause is (pause_at is not None)
                assert versions == [serving_version]
                state, phase, duration = "CLEARED", "clear", 13
                if isinstance(obj, io.ReleaseGpuDeltaCacheReqInput):
                    phase, duration = "release", 17
                    assert isinstance(calls[-2], io.ClearGpuDeltaStateReqInput)
                else:
                    assert isinstance(obj, io.ClearGpuDeltaStateReqInput)
            now[0] += duration
            return {
                "success": failure != phase,
                "message": "" if failure != phase else "injected failure",
                "participants": [
                    dict(
                        identity=identity,
                        session_id=publication["session_id"],
                        state=state,
                        version=7,
                        target_version=7,
                    )
                    for identity in publication["participants"]
                ],
            }

        facade._request = request
        result = await facade.request(
            io.UpdateWeightsFromGpuDeltaReqInput(
                **publication,
                release_state=release_state,
                pause_mode=pause_mode,
                abort_all_requests=abort_all_requests,
                weight_version=weight_version,
            )
        )
        assert result["success"] is (failure is None)
        assert cache_clears == (
            ["clear"] if pause_mode == "retract" and failure in {None, "resume", "clear", "release"} else []
        )
        assert aborted == (["abort"] if abort_all_requests and failure != "prepare" else [])
        if failure is None:
            assert versions == [serving_version]
            expected = [
                io.PrepareGpuDeltaReqInput,
                io.ApplyGpuDeltaReqInput,
                io.ResumeGpuDeltaReqInput,
            ]
            if release_state:
                expected += [
                    io.ClearGpuDeltaStateReqInput,
                    io.ReleaseGpuDeltaCacheReqInput,
                ]
            assert [type(obj) for obj in calls] == expected
            assert facade.manager.is_pause is (pause_at is not None)
            assert all(rank["state"] == "RESUMED" for rank in result["participants"])
            assert all(rank["state"] == "APPLIED" for rank in result["applied_participants"])
            assert result["timings"] == {
                "prepare_s": 5.0,
                "apply_s": 7.0,
                "resume_s": 11.0,
                "activation_s": 23.0,
            }
            assert now[0] == (53.0 if release_state else 23.0)
        else:
            assert any(isinstance(obj, io.AbortGpuDeltaReqInput) for obj in calls) is (failure == "prepare")
            assert facade.manager.is_pause is (failure in {"apply", "resume", "uncertain_apply"})
            if failure == "clear":
                assert not any(isinstance(obj, io.ReleaseGpuDeltaCacheReqInput) for obj in calls)
            elif failure not in {"release", "prepare"}:
                assert not any(isinstance(obj, io.ClearGpuDeltaStateReqInput) for obj in calls)
            assert versions == ([serving_version] if failure in {"clear", "release"} else [])
            if failure != "prepare":
                count = len(calls)
                denied = await facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
                assert not denied["success"] and "replace the engine" in denied["message"] and len(calls) == count

    asyncio.run(run())


def test_public_request_rejects_multiple_tokenizer_workers_before_dispatch(control, publication, monkeypatch):
    facade, sent, versions = control
    monkeypatch.setattr(tokenizer, "get_serving", lambda: SimpleNamespace(tokenizer_worker_num=2))
    result = asyncio.run(facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication)))
    assert not result["success"] and "one Python tokenizer worker" in result["message"]
    assert sent == versions == [] and not facade.manager.is_pause


@pytest.mark.parametrize("still_timed_out", [False, True])
def test_unsettled_preparation_can_retry_once_after_verified_reset(control, publication, monkeypatch, still_timed_out):
    monkeypatch.setattr(tokenizer, "_PREPARE_TIMEOUT_S", 0.01)

    async def run():
        facade, _, versions = control
        calls = []

        async def request(obj):
            calls.append(type(obj))
            if type(obj) is io.PrepareGpuDeltaReqInput and (still_timed_out or len(calls) == 1):
                await asyncio.Future()
            if isinstance(obj, _RESET_PHASES):
                return reset_reply(obj, publication)
            return {"success": True, "message": "", "participants": [{"version": publication["target_version"]}]}

        facade._request = request
        result = await facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
        assert result["success"] is (not still_timed_out)
        if still_timed_out:
            assert "TimeoutError" in result["message"]
        expected = [io.PrepareGpuDeltaReqInput, *_RESET_PHASES, io.PrepareGpuDeltaReqInput]
        expected += list(_RESET_PHASES) if still_timed_out else [io.ApplyGpuDeltaReqInput, io.ResumeGpuDeltaReqInput]
        assert calls == expected
        assert not facade.manager.is_pause and versions == ([] if still_timed_out else ["7"])

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("phase", ["prepare", "pause_lock", "apply"])
def test_cancel_only_aborts_before_apply_and_never_reopens_admission(control, publication, phase):
    async def run():
        facade, _, versions = control
        reached = asyncio.Event()
        sent = asyncio.Queue()
        facade.communicator._send = sent.put_nowait
        if phase == "pause_lock":

            class BusyPauseCondition(asyncio.Condition):
                async def __aenter__(self):
                    reached.set()
                    return await super().__aenter__()

            facade.manager.is_pause_cond = BusyPauseCondition()
            await facade.manager.is_pause_cond.acquire()

        update = io.UpdateWeightsFromGpuDeltaReqInput(**publication)
        task = asyncio.create_task(facade.request(update))
        prepare = await sent.get()
        assert type(prepare) is io.PrepareGpuDeltaReqInput
        ranks = range(len(publication["participants"]))
        if phase != "prepare":
            reply(facade, prepare, "PREPARED", ranks)
            if phase == "pause_lock":
                await reached.wait()
            else:
                delayed_apply = await sent.get()
                assert type(delayed_apply) is io.ApplyGpuDeltaReqInput
        task.cancel()
        if phase != "apply":
            for expected in _RESET_PHASES:
                cleanup = await sent.get()
                assert type(cleanup) is expected
                facade.communicator.handle_recv(
                    io.GpuDeltaReqOutput(rid=cleanup.rid, **reset_reply(cleanup, publication))
                )
        with pytest.raises(asyncio.CancelledError):
            await task
        assert facade.manager.is_pause is (phase == "apply")
        assert versions == [] and sent.empty()
        if phase == "pause_lock":
            facade.manager.is_pause_cond.release()

        if phase == "apply":
            # The original apply is still in transit. A later HTTP request cannot trust
            # a PREPARED reply as proof that sending the same XOR again would be safe.
            lookup = asyncio.create_task(
                facade.request(io.GetGpuDeltaInfoReqInput(engine_id="e0", session_id=update.session_id))
            )
            info = await sent.get()
            description = reset_reply(info, publication)
            for row in description["participants"]:
                row["state"] = "PREPARED"
            facade.communicator.handle_recv(io.GpuDeltaReqOutput(rid=info.rid, **description))
            assert (await lookup)["participants"] == description["participants"]
            denied = await facade.request(update)
            assert not denied["success"] and "replace the engine" in denied["message"] and sent.empty()
            reply(facade, delayed_apply, "APPLIED", ranks)
        else:
            # Verified pre-write cancellation permits the caller's next activation.
            next_update = asyncio.create_task(facade.request(update))
            for request_type, state in (
                (io.PrepareGpuDeltaReqInput, "PREPARED"),
                (io.ApplyGpuDeltaReqInput, "APPLIED"),
                (io.ResumeGpuDeltaReqInput, "RESUMED"),
            ):
                obj = await sent.get()
                assert type(obj) is request_type
                reply(facade, obj, state, ranks)
            assert (await next_update)["success"]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("outcome", ["recovered", "still_failing", "permanent_peer", "changed_source", "failed_drain"])
def test_retry_only_verified_prewrite_storage_failures(control, publication, outcome):
    # Reverse application reads the target endpoint. Original-rank verification must use that same endpoint.
    publication["reverse"] = True

    async def run():
        facade, _, versions = control
        calls = []

        async def request(obj):
            calls.append(type(obj))
            if isinstance(obj, _RESET_PHASES):
                result = reset_reply(obj, publication)
                if outcome == "permanent_peer" and isinstance(obj, io.AbortGpuDeltaReqInput):
                    result["participants"][-1]["retryable_prepare_error"] = False
                elif outcome == "changed_source" and isinstance(obj, io.GetGpuDeltaInfoReqInput):
                    result["participants"][-1]["stream_id"] = "foreign"
                elif outcome == "failed_drain" and isinstance(obj, io.ClearGpuDeltaStateReqInput):
                    result.update(success=False, message="drain failed")
                return result
            if type(obj) is io.PrepareGpuDeltaReqInput:
                assert obj.session_id == publication["session_id"]
                if calls.count(type(obj)) == 1 or outcome == "still_failing":
                    return {
                        "success": False,
                        "message": "storage read failed",
                        "participants": [
                            dict(
                                identity=obj.participants[0],
                                session_id=obj.session_id,
                                state="FAILED",
                                retryable_prepare_error=True,
                            )
                        ],
                    }
            return {"success": True, "message": "", "participants": [{"version": publication["base_version"]}]}

        facade._request = request
        result = await facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
        assert result["success"] is (outcome == "recovered")
        reset = list(_RESET_PHASES)
        if outcome == "failed_drain":
            reset = reset[:2]
        expected = [io.PrepareGpuDeltaReqInput, *reset]
        if outcome in {"recovered", "still_failing"}:
            expected.append(io.PrepareGpuDeltaReqInput)
            expected += [io.ApplyGpuDeltaReqInput, io.ResumeGpuDeltaReqInput] if outcome == "recovered" else reset
        assert calls == expected
        assert versions == ([str(publication["base_version"])] if outcome == "recovered" else [])
        assert not facade.manager.is_pause
        if outcome in {"changed_source", "failed_drain"}:
            denied = await facade.request(io.UpdateWeightsFromGpuDeltaReqInput(**publication))
            assert not denied["success"] and "replace the engine" in denied["message"] and calls == expected

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_late_reply_after_cancellation_cannot_acknowledge_resume():
    async def scenario():
        sent = []
        comm = GpuDeltaCommunicator(sent.append, 1)
        old = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        old_rid = sent[-1].rid
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        resume = asyncio.create_task(comm(SimpleNamespace(rid=None)))
        await asyncio.sleep(0)
        comm.handle_recv(SimpleNamespace(rid=old_rid, state="APPLIED"))
        await asyncio.sleep(0)
        assert not resume.done()
        comm.handle_recv(SimpleNamespace(rid=sent[-1].rid, state="RESUMED"))
        assert (await resume)[0].state == "RESUMED"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["prepare", "replacement"])
def test_prepare_failure_does_not_wait_for_other_ranks(publication, failure):
    async def run():
        sent = asyncio.Queue()
        comm = GpuDeltaCommunicator(sent.put_nowait, 1)
        task = asyncio.create_task(comm(io.PrepareGpuDeltaReqInput(**publication)))
        request = await sent.get()
        row = {
            "identity": None if failure == "prepare" else {"rank_id": "replacement"},
            "state": "FAILED" if failure == "prepare" else "PREPARED",
        }
        comm.handle_recv(
            io.GpuDeltaReqOutput(
                rid=request.rid,
                success=failure != "prepare",
                message="prepare failed" if failure == "prepare" else "",
                participants=[row],
            )
        )
        results = await task
        assert len(results) == 1 and not results[0].success
        expected = {
            "prepare": "prepare failed",
            "replacement": "delta prepare replied from another receiver incarnation",
        }
        assert results[0].message == expected[failure]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
