"""Independent engine phase ordering and failure boundaries without GPUs or HTTP."""

import asyncio
import copy

import pytest

from miles.backends.training_utils.weight_update import gpu_delta_session as session


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
                            "codec": "snappy-zstd",
                            "tensors": [
                                {"name": "w", "dtype": "U8", "shape": [4], "encoding": "xor_bytes", "views": [view]}
                            ]
                        },
                    }
                    for identity, view in zip(identities, views, strict=True)
                ],
            }
        )
        clients.append(_Engine(engine, identities, events, failure, failed_engine))
    plan, cohort, digest = session.merge_plans(descriptions)
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

    def _response(self, state):
        receipts = [
            {
                "identity": identity,
                "state": state,
                **{
                    key: self.args[key]
                    for key in ("session_id", "manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest")
                },
                "cohort_digest": "original-cohort",
            }
            for identity in self.identities
        ]
        return {"success": True, "participants": receipts}

    async def prepare_weights_from_delta(self, **kwargs):
        self.args = kwargs
        assert "staging" not in kwargs
        assert "expected_engines" not in kwargs
        assert list(kwargs["participants"]) == self.identities
        assert "cohort" not in kwargs
        assert kwargs["host_tensor_names"] == {f"engine-host-{self.index}": ["w"]}
        await asyncio.sleep(0.01 if self.index else 0)
        if self.failure == "prepare" and self.index == self.failed_engine:
            raise RuntimeError("prepare rejected")
        response = self._response("PREPARING")
        if self.index == self.failed_engine:
            if self.failure == "prepare_identity":
                response["participants"][0]["identity"] = self.identities[0] | {"rank_id": "replacement"}
            elif self.failure == "prepare_state":
                response["participants"][0]["state"] = "APPLIED"
        return response

    async def get_weights_delta_status(self, **kwargs):
        self.polls += 1
        if self.index == 1 and self.polls == 1:
            return self._response("PREPARING")
        if not self.prepared:
            self.events.append((self.index, "prepared"))
            self.prepared = True
        return self._response("PREPARED")

    async def update_weights_from_delta(self, **kwargs):
        assert (self.index, "prepared") in self.events
        assert kwargs == {"session_id": self.args["session_id"]}
        await asyncio.sleep(0.01 if self.index else 0)
        self.events.append((self.index, "applied"))
        if self.failure == "apply" and self.index == self.failed_engine:
            raise RuntimeError("apply failed")
        reply = self._response("APPLIED")
        if self.failure == "identity" and self.index == self.failed_engine:
            reply["participants"][0]["identity"] = self.identities[0] | {"rank_id": "replacement"}
        if self.failure == "plan" and self.index == self.failed_engine:
            reply["participants"][0]["plan_digest"] = "different-plan"
        for receipt in reply["participants"]:
            receipt["certificate"] = dict(receipt)
            receipt["result"] = {"large_nested_diagnostics": [1, 2, 3]}
        return reply

    async def resume_weights_from_delta(self, **kwargs):
        assert (self.index, "applied") in self.events
        assert len(kwargs["receipts"]) == 2 and all(r["state"] == "APPLIED" for r in kwargs["receipts"])
        assert [r["identity"] for r in kwargs["receipts"]] == self.identities
        assert all("result" not in r and "certificate" not in r for r in kwargs["receipts"])
        if self.failure == "resume" and self.index == self.failed_engine:
            raise RuntimeError("resume reply lost")
        self.events.append((self.index, "resumed"))
        reply = self._response("RESUMED")
        for receipt in reply["participants"]:
            receipt["scheduler_timing"] = {"blocked_s": 1.0 + self.index}
        return reply

    async def abort_weights_from_delta(self, **kwargs):
        self.events.append((self.index, "abort"))
        return {"success": True}


def test_fast_engine_resumes_while_other_engine_still_prepares(monkeypatch):
    clients, descriptions, publication, events = _setup()
    cohort = session.negotiate_cohort(descriptions)
    monkeypatch.setattr(session, "merge_plans", lambda *args, **kwargs: pytest.fail("Immutable plan renegotiated"))
    result = asyncio.run(session.activate_publication(clients, cohort, publication, session_id="s"))
    assert len(result["receipts"]) == 4
    assert all(r["state"] == "APPLIED" and "result" in r for r in result["receipts"])
    assert len(result["resumed_receipts"]) == 4
    assert [r["scheduler_timing"]["blocked_s"] for r in result["resumed_receipts"]] == [1.0, 1.0, 2.0, 2.0]
    assert sum(event == "resumed" for _, event in events) == 2
    assert clients[1].polls >= 2
    assert events.index((0, "resumed")) < events.index((1, "prepared"))
    assert {r["engine_id"] for r in result["engine_timings"]} == {"engine-0", "engine-1"}
    assert set(result["coordinator_timings"]) == {"activation_s"}


@pytest.mark.parametrize("failure", ["prepare", "prepare_identity", "prepare_state", "apply", "identity", "plan"])
def test_failure_is_scoped_to_its_engine_without_blind_replay(failure):
    clients, descriptions, publication, events = _setup(failure)
    with pytest.raises(RuntimeError):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(descriptions), publication, session_id="s"))
    assert (0, "resumed") in events and (1, "resumed") not in events
    if failure.startswith("prepare"):
        assert (1, "abort") in events and (0, "abort") not in events
        assert (1, "applied") not in events
    else:
        assert not any(event == "abort" for _, event in events)


def test_uncertain_resume_is_terminal_without_abort_or_replay():
    clients, descriptions, publication, events = _setup("resume")
    with pytest.raises(RuntimeError, match="resume reply lost"):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(descriptions), publication, session_id="s"))
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


def test_common_plan_deduplicates_replicas_but_rejects_conflicting_views():
    clients, descriptions, publication, _ = _setup()
    plan, identities, digest = session.merge_plans(descriptions)
    assert len(plan) == 1 and len(plan[0]["views"]) == 2 and len(identities) == 4
    assert digest == publication["plan_digest"]
    broken = copy.deepcopy(descriptions)
    broken[1]["participants"][0]["plan"]["tensors"][0]["views"][0]["slices"] = [[1, 2]]
    with pytest.raises(ValueError, match="conflict"):
        session.merge_plans(broken)


def test_codec_mismatch_rejected_before_any_engine_preparation():
    clients, descriptions, publication, events = _setup()
    descriptions[1]["participants"][0]["plan"]["codec"] = "zstd"
    with pytest.raises(ValueError, match="codecs differ"):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(descriptions), publication))
    assert not events
    assert all(client.args is None for client in clients)


def test_bounded_wait_cancels_inflight_status_requests():
    clients, descriptions, publication, events = _setup()

    async def hanging_status(**kwargs):
        try:
            await asyncio.Future()
        finally:
            events.append((0, "status_cancelled"))

    clients[0].get_weights_delta_status = hanging_status
    participants = [p["identity"] for p in descriptions[0]["participants"]]
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(session._wait_state(clients[0], participants, session_id="s", publication=publication, timeout=0.01))
    assert events == [(0, "status_cancelled")]


def test_cohort_only_decodes_each_hosts_union_and_requires_explicit_host_identity():
    _, descriptions, _, _ = _setup()
    for participant in descriptions[1]["participants"]:
        participant["identity"]["host_cache_id"] = "other-host"
        participant["plan"]["tensors"][0]["name"] = "other-experts"
    cohort = session.negotiate_cohort(descriptions)
    assert cohort.engine_host_tensor_names == ({"engine-host-0": ["w"]}, {"other-host": ["other-experts"]})
    del descriptions[1]["participants"][0]["identity"]["host_cache_id"]
    with pytest.raises(ValueError, match="engine-host cache identity"):
        session.negotiate_cohort(descriptions)


def test_shared_cache_across_engines_is_rejected_before_preparation():
    clients, descriptions, _, _ = _setup()
    for participant in descriptions[1]["participants"]:
        participant["identity"]["host_cache_id"] = "engine-host-0"
    with pytest.raises(ValueError, match="must not share a host arena"):
        session.negotiate_cohort(descriptions)
    assert all(client.args is None for client in clients)
