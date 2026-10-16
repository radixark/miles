"""Independent engine phase ordering and failure boundaries without GPUs or HTTP."""

import asyncio
import copy

import pytest
from tests.fast.backends.sglang_utils import test_sglang_api_client

from miles.backends.training_utils.weight_update.protocols.gpu_delta import session


def _setup(failure=None, failed_engine=1):
    events, clients, descriptions = [], [], []
    for engine in range(2):
        identities = [
            {
                "engine_id": f"engine-{engine}",
                "host_cache_id": f"engine-host-{engine}",
                "rank_id": f"rank-{engine}-{rank}",
                "pid": 100 + rank,
                "start_ticks": 456,
                "tp_rank": rank,
                "dp_rank": rank,
                "pp_rank": 0,
            }
            for rank in range(2)
        ]
        views = [{"id": f"tp-{rank}", "slices": [[rank * 2, rank * 2 + 2]]} for rank in range(2)]
        descriptions.append(
            {
                "success": True,
                "participants": [
                    {
                        "identity": identity,
                        "plan": {
                            "tensors": [
                                {"name": "w", "dtype": "U8", "shape": [4], "encoding": "xor_bytes", "views": [view]}
                            ],
                        },
                    }
                    for identity, view in zip(identities, views, strict=True)
                ],
            }
        )
        clients.append(_Engine(engine, identities, events, failure, failed_engine))
    _, _, digest = session.merge_plans(descriptions)
    publication = {
        "codec": "snappy-zstd",
        "manifest_path": "/shared/version/manifest.json",
        "manifest_sha256": "f" * 64,
        "stream_id": "stream",
        "base_version": 0,
        "target_version": 1,
        "plan_digest": digest,
    }
    return clients, descriptions, publication, events


class _Engine:
    def __init__(self, index, identities, events, failure, failed_engine):
        self.index, self.identities, self.events, self.failure = index, identities, events, failure
        self.failed_engine = failed_engine
        self.args = None
        self.polls = 0
        self.prepared = False
        self.model_version = None

    def _response(self, state):
        receipts = [
            {
                "identity": identity,
                "state": state,
                **{
                    key: self.args[key]
                    for key in (
                        "session_id",
                        "manifest_sha256",
                        "stream_id",
                        "base_version",
                        "target_version",
                        "plan_digest",
                    )
                },
            }
            for identity in self.identities
        ]
        return {"success": True, "participants": receipts}

    async def prepare_gpu_delta(self, **kwargs):
        self.args = kwargs
        assert list(kwargs["participants"]) == self.identities
        await asyncio.sleep(0.01 if self.index else 0)
        if self.failure == "prepare" and self.index == self.failed_engine:
            raise RuntimeError("prepare rejected")
        return self._response("PREPARING")

    async def get_gpu_delta_status(self, **kwargs):
        self.polls += 1
        if self.failure == "prepare_worker" and self.index == self.failed_engine:
            return {"success": False, "message": "CPU preparation failed"}
        if self.index == 1 and self.polls == 1:
            return self._response("PREPARING")
        if not self.prepared:
            self.events.append((self.index, "prepared"))
            self.prepared = True
        return self._response("PREPARED")

    async def apply_gpu_delta(self, **kwargs):
        assert (self.index, "prepared") in self.events
        assert kwargs == {"session_id": self.args["session_id"]}
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append((self.index, "applied"))
        if self.failure == "apply" and self.index == self.failed_engine:
            return {"success": False, "message": "apply failed"}
        reply = self._response("APPLIED")
        for receipt in reply["participants"]:
            receipt["result"] = {"large_nested_diagnostics": [1, 2, 3]}
        return reply

    async def resume_gpu_delta(self, **kwargs):
        assert (self.index, "applied") in self.events
        self.model_version = kwargs.get("weight_version")
        assert set(kwargs) <= {"session_id", "weight_version"}
        assert kwargs["session_id"] == self.args["session_id"]
        if self.failure == "resume" and self.index == self.failed_engine:
            raise RuntimeError("resume reply lost")
        self.events.append((self.index, "resumed"))
        reply = self._response("RESUMED")
        for receipt in reply["participants"]:
            receipt["scheduler_timing"] = {"blocked_s": 1.0 + self.index}
        return reply

    async def abort_gpu_delta(self, **kwargs):
        self.events.append((self.index, "abort"))
        return {"success": True}


@pytest.mark.parametrize("weight_version", [None, "42"])
def test_fast_engine_resumes_while_other_engine_still_prepares(monkeypatch, weight_version):
    clients, descriptions, publication, events = _setup()
    cohort = session.negotiate_cohort(descriptions)
    monkeypatch.setattr(session, "merge_plans", lambda *args, **kwargs: pytest.fail("Immutable plan renegotiated"))
    result = asyncio.run(
        session.activate_publication(clients, cohort, publication, session_id="s", weight_version=weight_version)
    )
    assert [client.model_version for client in clients] == [weight_version, weight_version]
    assert all(receipt["target_version"] == 1 for receipt in result["resumed_receipts"])
    assert len(result["receipts"]) == 4
    assert all(r["state"] == "APPLIED" and "result" in r for r in result["receipts"])
    assert len(result["resumed_receipts"]) == 4
    assert [r["scheduler_timing"]["blocked_s"] for r in result["resumed_receipts"]] == [1.0, 1.0, 2.0, 2.0]
    assert sum(event == "resumed" for _, event in events) == 2
    assert clients[1].polls >= 2
    assert events.index((0, "resumed")) < events.index((1, "prepared"))
    assert {r["engine_id"] for r in result["engine_timings"]} == {"engine-0", "engine-1"}
    assert set(result["coordinator_timings"]) == {"activation_s"}


@pytest.mark.parametrize("failure", ["prepare", "prepare_worker", "apply"])
def test_failure_is_scoped_to_its_engine_without_blind_replay(failure):
    clients, descriptions, publication, events = _setup(failure)
    with pytest.raises(RuntimeError):
        asyncio.run(
            session.activate_publication(clients, session.negotiate_cohort(descriptions), publication, session_id="s")
        )
    assert (0, "resumed") in events and (1, "resumed") not in events
    if failure.startswith("prepare"):
        assert (1, "abort") in events and (0, "abort") not in events
        assert (1, "applied") not in events
    else:
        assert not any(event == "abort" for _, event in events)


def test_uncertain_resume_is_terminal_without_abort_or_replay():
    clients, descriptions, publication, events = _setup("resume")
    with pytest.raises(RuntimeError, match="resume reply lost"):
        asyncio.run(
            session.activate_publication(clients, session.negotiate_cohort(descriptions), publication, session_id="s")
        )
    assert sum(event == "applied" for _, event in events) == 2
    assert not any(event == "abort" for _, event in events)
    assert (0, "resumed") in events


def test_early_failure_settles_other_engine_before_returning():
    clients, descriptions, publication, events = _setup("apply", failed_engine=0)
    with pytest.raises(RuntimeError, match="apply failed"):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(descriptions), publication))
    assert events.index((0, "applied")) < events.index((1, "resumed"))
    assert (0, "resumed") not in events
    assert not any(event == "abort" for _, event in events)


def test_common_plan_deduplicates_replicas_and_rejects_view_conflicts_before_preparation():
    clients, descriptions, publication, events = _setup()
    plan, identities, digest = session.merge_plans(descriptions)
    assert len(plan) == 1 and len(plan[0]["views"]) == 2 and len(identities) == 4
    assert digest == publication["plan_digest"]
    broken = copy.deepcopy(descriptions)
    broken[1]["participants"][0]["plan"]["tensors"][0]["views"][0]["slices"] = [[1, 2]]
    with pytest.raises(ValueError, match="conflict"):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(broken), publication))
    assert not events
    assert all(client.args is None for client in clients)


@pytest.mark.parametrize("field,value", [("format", "fp8-block128-ue8m0"), ("checkpoint_dtype", "F32")])
def test_common_plan_binds_optional_canonical_representation(field, value):
    _, descriptions, _, _ = _setup()
    previous, _, previous_digest = session.merge_plans(descriptions)
    for description in descriptions:
        for participant in description["participants"]:
            participant["plan"]["tensors"][0][field] = value
    plan, _, digest = session.merge_plans(descriptions)
    assert field not in previous[0] and plan[0][field] == value and digest != previous_digest
    del descriptions[1]["participants"][0]["plan"]["tensors"][0][field]
    for ordered in (descriptions, descriptions[::-1]):
        with pytest.raises(ValueError, match="Receiver canonical plans differ"):
            session.merge_plans(ordered)


def test_bounded_wait_cancels_inflight_status_requests():
    clients, descriptions, publication, events = _setup()

    async def hanging_status(**kwargs):
        try:
            await asyncio.Future()
        finally:
            events.append((0, "status_cancelled"))

    clients[0].get_gpu_delta_status = hanging_status
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(session._wait_prepared(clients[0], session_id="s", timeout=0.01))
    assert events == [(0, "status_cancelled")]


client = test_sglang_api_client.client
recorder = test_sglang_api_client.recorder


@pytest.mark.parametrize(
    "one_shot,options,policy",
    [
        pytest.param(False, {}, (True, False), id="staged-defaults"),
        pytest.param(False, {"flush_cache": False}, (False, False), id="staged-no-flush"),
        pytest.param(False, {"abort_all_requests": True}, (True, True), id="staged-abort"),
        pytest.param(True, {}, (True, False), id="one-shot-defaults"),
        pytest.param(True, {"release_state": False}, (True, False), id="one-shot-retain-state"),
        pytest.param(True, {"flush_cache": False}, (False, False), id="one-shot-no-flush"),
        pytest.param(True, {"abort_all_requests": True}, (True, True), id="one-shot-abort"),
    ],
)
async def test_gpu_delta_request_lifetime_and_cache_policy(client, recorder, one_shot, options, policy):
    expected = {"flush_cache": policy[0], "abort_all_requests": policy[1]}
    if one_shot:
        await client.update_weights_from_gpu_delta("/checkpoint/gpu-delta/manifest.json", **options)
        endpoint = "update_weights_from_gpu_delta"
        expected.update(
            manifest_path="/checkpoint/gpu-delta/manifest.json", release_state=options.get("release_state", True)
        )
    else:
        await client.apply_gpu_delta("publication-1", **options)
        endpoint = "apply_gpu_delta"
        expected["session_id"] = "publication-1"
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{test_sglang_api_client.SERVER_URL}/{endpoint}")
    assert kwargs["json"] == expected
