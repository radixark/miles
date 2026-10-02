"""Multi-engine phase ordering and failure boundaries without GPUs or HTTP."""

import asyncio
import copy

import pytest

from miles.backends.training_utils.weight_update import gpu_delta_session as session


def _setup(failure=None):
    events, clients, descriptions = [], [], []
    for engine in range(2):
        identities = [
            {
                "engine_id": f"engine-{engine}",
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
                            ]
                        },
                    }
                    for identity, view in zip(identities, views, strict=True)
                ],
            }
        )
        clients.append(_Engine(engine, identities, events, failure))
    plan, cohort, digest = session.merge_plans(descriptions)
    publication = {
        "manifest_path": "/shared/version/manifest.json",
        "manifest_sha256": "f" * 64,
        "stream_id": "stream",
        "base_version": 0,
        "target_version": 1,
        "plan_digest": digest,
    }
    return clients, descriptions, publication, events


class _Engine:
    def __init__(self, index, identities, events, failure):
        self.index, self.identities, self.events, self.failure = index, identities, events, failure
        self.args = None
        self.polls = 0
        self.paused = False
        self.prepared = False

    def _response(self, state):
        receipts = [
            {
                "identity": identity,
                "state": state,
                **{
                    key: self.args[key]
                    for key in ("session_id", "manifest_sha256", "stream_id", "base_version", "target_version")
                },
            }
            for identity in self.identities
        ]
        return {"success": True, "participants": receipts}

    async def prepare_weights_from_delta(self, **kwargs):
        self.args = kwargs
        assert kwargs["participants"] == self.identities
        assert len(kwargs["cohort"]) == 4
        await asyncio.sleep(0.01 if self.index else 0)
        if self.failure == "prepare" and self.index == 1:
            raise RuntimeError("prepare rejected")
        return self._response("PREPARING")

    async def pause_generation(self, **kwargs):
        assert sum(event == "prepared" for _, event in self.events) == 2
        self.paused = True
        self.polls = 0
        self.events.append((self.index, "pause_ack"))
        return {"success": True}

    async def get_weights_delta_status(self, **kwargs):
        self.polls += 1
        if not self.paused:
            if self.index == 1 and self.polls == 1:
                return self._response("PREPARING")
            if not self.prepared:
                self.events.append((self.index, "prepared"))
                self.prepared = True
            return self._response("PREPARED")
        if self.index == 1 and self.polls == 1:
            return self._response("PREPARED")
        self.events.append((self.index, "quiesced"))
        return self._response("QUIESCED")

    async def update_weights_from_delta(self, **kwargs):
        assert len({index for index, event in self.events if event == "quiesced"}) == 2
        assert len(kwargs["receipts"]) == 4
        assert all(r["state"] == "QUIESCED" for r in kwargs["receipts"])
        self.events.append((self.index, "applied"))
        if self.failure == "apply" and self.index == 1:
            raise RuntimeError("apply failed")
        reply = self._response("APPLIED")
        if self.failure == "identity" and self.index == 1:
            reply["participants"][0]["identity"] = self.identities[0] | {"rank_id": "replacement"}
        return reply

    async def commit_weights_from_delta(self, **kwargs):
        assert sum(event == "applied" for _, event in self.events) == 2
        assert len(kwargs["receipts"]) == 4 and all(r["state"] == "APPLIED" for r in kwargs["receipts"])
        self.events.append((self.index, "committed"))
        if self.failure == "commit" and self.index == 1:
            raise RuntimeError("commit failed")
        return self._response("COMMITTED")

    async def continue_generation(self, **kwargs):
        assert sum(event == "committed" for _, event in self.events) == 2
        assert len(kwargs["delta_commit_receipts"]) == 4
        assert all(r["state"] == "COMMITTED" for r in kwargs["delta_commit_receipts"])
        self.events.append((self.index, "resumed"))
        reply = self._response("RESUMED")
        for receipt in reply["participants"]:
            receipt["scheduler_timing"] = {"blocked_s": 1.0 + self.index}
        return reply

    async def abort_weights_from_delta(self, **kwargs):
        self.events.append((self.index, "abort"))
        return {"success": True}


def test_every_engine_prepared_then_quiesced_then_applied_then_committed_before_resume():
    clients, descriptions, publication, events = _setup()
    result = asyncio.run(
        session.activate_publication(clients, descriptions, publication, staging="tensor", session_id="s")
    )
    assert len(result["receipts"]) == 4
    assert len(result["resumed_receipts"]) == 4
    assert [r["scheduler_timing"]["blocked_s"] for r in result["resumed_receipts"]] == [1.0, 1.0, 2.0, 2.0]
    assert sum(event == "resumed" for _, event in events) == 2
    assert clients[1].polls >= 2  # a pause ACK did not bypass its pending reader fence


@pytest.mark.parametrize("failure", ["prepare", "apply", "identity", "commit"])
def test_failure_never_resumes_or_blindly_replays(failure):
    clients, descriptions, publication, events = _setup(failure)
    with pytest.raises(RuntimeError):
        asyncio.run(session.activate_publication(clients, descriptions, publication, staging="full", session_id="s"))
    assert not any(event == "resumed" for _, event in events)
    if failure == "prepare":
        assert sum(event == "abort" for _, event in events) == 2
        assert not any(event == "pause_ack" for _, event in events)
    else:
        assert not any(event == "abort" for _, event in events)


def test_common_plan_deduplicates_replicas_but_rejects_conflicting_views():
    clients, descriptions, publication, _ = _setup()
    plan, identities, digest = session.merge_plans(descriptions)
    assert len(plan) == 1 and len(plan[0]["views"]) == 2 and len(identities) == 4
    assert digest == publication["plan_digest"]
    broken = copy.deepcopy(descriptions)
    broken[1]["participants"][0]["plan"]["tensors"][0]["views"][0]["slices"] = [[1, 2]]
    with pytest.raises(ValueError, match="conflict"):
        session.merge_plans(broken)


def test_bounded_wait_cancels_inflight_status_requests():
    clients, descriptions, publication, events = _setup()

    async def hanging_status(**kwargs):
        try:
            await asyncio.Future()
        finally:
            events.append((0, "status_cancelled"))

    for client in clients:
        client.get_weights_delta_status = hanging_status
    expected = [[p["identity"] for p in d["participants"]] for d in descriptions]
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            session._wait_state(
                clients,
                expected,
                state="PREPARED",
                pending="PREPARING",
                session_id="s",
                publication=publication,
                timeout=0.01,
            )
        )
    assert len(events) == 2 and all(event == "status_cancelled" for _, event in events)
