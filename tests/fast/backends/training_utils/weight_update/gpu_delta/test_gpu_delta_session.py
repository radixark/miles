"""Independent engine update requests and failure boundaries without GPUs or HTTP."""

import asyncio
from dataclasses import replace

import pytest
from tests.fast.backends.sglang_utils import test_sglang_api_client

from miles.backends.training_utils.weight_update.protocols.gpu_delta import session


def _setup(failure=None, failed_engine=1):
    events, clients, descriptions = [], [], []
    for engine in range(2):
        identities = [
            {
                "engine_id": f"engine-{engine}",
                "rank_id": f"rank-{engine}-{rank}",
            }
            for rank in range(2)
        ]
        descriptions.append(
            {
                "success": True,
                "participants": [{"identity": identity, "version": 0, "state": "IDLE"} for identity in identities],
            }
        )
        clients.append(_Engine(engine, identities, events, failure, failed_engine))
    publication = {
        "codec": "snappy-zstd",
        "manifest_path": "/shared/version/manifest.json",
        "manifest_sha256": "f" * 64,
        "stream_id": "stream",
        "base_version": 0,
        "target_version": 1,
        "plan_digest": "canonical-checkpoint-digest",
    }
    return clients, descriptions, publication, events


class _Engine:
    def __init__(self, index, identities, events, failure, failed_engine):
        self.index, self.identities, self.events, self.failure = index, identities, events, failure
        self.failed_engine = failed_engine
        self.args = None
        self.calls = 0
        self.timings = {"prepare_s": 10.0 + index, "apply_s": 2.0, "resume_s": 3.0, "activation_s": 15.0 + index}

    async def update_weights_from_gpu_delta(self, **kwargs):
        self.calls += 1
        assert self.calls == 1, "An uncertain XOR publication was replayed"
        self.args = kwargs
        assert list(kwargs["participants"]) == self.identities
        await asyncio.sleep(0.01 if self.index else 0)
        failure = self.failure if self.index == self.failed_engine else None
        if failure == "transport":
            raise RuntimeError("update request lost")
        self.events.append((self.index, "prepared"))
        if failure == "rejected":
            return {"success": False, "message": "update rejected"}
        provenance = {
            key: kwargs[key]
            for key in ("session_id", "stream_id", "manifest_sha256", "base_version", "target_version", "plan_digest")
        }
        applied = [
            {
                **provenance,
                "identity": identity,
                "state": "APPLIED",
                "result": {"applied": True, "target_version": kwargs["target_version"]},
            }
            for identity in self.identities
        ]
        self.events.append((self.index, "applied"))
        if failure == "applied_reply":
            raise RuntimeError("apply reply lost")
        self.events.append((self.index, "resumed"))
        if failure == "reply":
            raise RuntimeError("update reply lost")
        resumed = [
            {
                **provenance,
                "identity": identity,
                "state": "RESUMED",
                "scheduler_timing": {"blocked_s": 1.0 + self.index},
                "result": {"large_nested_diagnostics": [1, 2, 3]},
            }
            for identity in self.identities
        ]
        return {
            "success": True,
            "message": "",
            "participants": resumed,
            "applied_participants": applied,
            "timings": self.timings,
        }


@pytest.mark.parametrize("weight_version,skip_payload_hash", [(None, False), ("42", True)])
def test_fast_engine_resumes_while_other_engine_still_prepares(monkeypatch, weight_version, skip_payload_hash):
    if skip_payload_hash:
        monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "1")
    else:
        monkeypatch.delenv("GPU_DELTA_SKIP_PAYLOAD_HASH", raising=False)
    clients, descriptions, publication, events = _setup()
    pool = session.receiver_pool(descriptions)
    result = asyncio.run(
        session.activate_publication(clients, pool, publication, session_id="s", weight_version=weight_version)
    )
    for client in clients:
        assert client.calls == 1
        assert client.args == {
            **{key: value for key, value in publication.items() if key != "codec"},
            "session_id": "s",
            "participants": tuple(client.identities),
            "weight_version": weight_version,
            "validate_payload_hash": not skip_payload_hash,
        }
    assert len(result["receipts"]) == 4
    assert all(
        r["state"] == "APPLIED" and r["result"] == {"applied": True, "target_version": 1} for r in result["receipts"]
    )
    assert [r["identity"] for r in result["resumed_receipts"]] == [
        identity for client in clients for identity in client.identities
    ]
    assert all(r["result"] == {"large_nested_diagnostics": [1, 2, 3]} for r in result["resumed_receipts"])
    assert [r["scheduler_timing"]["blocked_s"] for r in result["resumed_receipts"]] == [1.0, 1.0, 2.0, 2.0]
    assert sum(event == "resumed" for _, event in events) == 2
    assert events.index((0, "resumed")) < events.index((1, "prepared"))
    assert result["engine_timings"] == [
        {"engine_id": f"engine-{client.index}", **client.timings} for client in clients
    ]
    assert set(result["coordinator_timings"]) == {"activation_s"}
    assert result["coordinator_timings"]["activation_s"] > 0


@pytest.mark.parametrize("failure", ["transport", "rejected", "reply"])
@pytest.mark.parametrize("failed_engine", [0, 1])
def test_update_failure_settles_other_engine_without_abort_or_replay(failure, failed_engine):
    clients, descriptions, publication, events = _setup(failure, failed_engine)
    with pytest.raises(RuntimeError, match="requires receiver recovery"):
        asyncio.run(
            session.activate_publication(clients, session.receiver_pool(descriptions), publication, session_id="s")
        )
    assert [client.calls for client in clients] == [1, 1]
    assert (1 - failed_engine, "resumed") in events
    # A lost reply may hide a completed update. Neither that ambiguity nor an
    # explicit failure authorizes another update request or a client abort.
    assert ((failed_engine, "resumed") in events) == (failure == "reply")


@pytest.mark.parametrize("failure", ["transport", "applied_reply", "reply", "rejected"])
def test_failed_or_lost_reply_quarantines_only_that_engine_without_status_or_replay(failure):
    clients, descriptions, publication, events = _setup(failure)
    result = asyncio.run(
        session.activate_publication(
            clients, session.receiver_pool(descriptions), publication, session_id="s", fault_tolerant=True
        )
    )
    assert set(result["failed_engines"]) == {"engine-1"}
    assert result["succeeded_engine_ids"] == ["engine-0"]
    assert [client.calls for client in clients] == [1, 1]
    assert len(result["resumed_receipts"]) == 2
    assert events.count((0, "applied")) == events.count((0, "resumed")) == 1
    assert events.count((1, "applied")) == int(failure in {"applied_reply", "reply"})
    assert events.count((1, "resumed")) == int(failure == "reply")


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", {"engine_id": "engine-1", "rank_id": "replacement"}),
        ("stream_id", "eval-or-other-job"),
        ("state", "APPLIED"),
    ],
)
def test_wrong_receipt_cannot_advance_a_failed_engine(field, value):
    clients, descriptions, publication, events = _setup()
    activate = clients[1].update_weights_from_gpu_delta

    async def wrong_receipt(**kwargs):
        result = await activate(**kwargs)
        result["participants"][0][field] = value
        return result

    clients[1].update_weights_from_gpu_delta = wrong_receipt
    result = asyncio.run(
        session.activate_publication(
            clients, session.receiver_pool(descriptions), publication, session_id="s", fault_tolerant=True
        )
    )
    assert set(result["failed_engines"]) == {"engine-1"}
    assert result["succeeded_engine_ids"] == ["engine-0"]
    assert len(result["resumed_receipts"]) == 2
    assert [client.calls for client in clients] == [1, 1]


def test_receiver_pool_contains_only_identity_and_rejects_duplicate_membership():
    clients, descriptions, publication, events = _setup()
    pool = session.receiver_pool(descriptions)
    assert pool.engine_ids == ("engine-0", "engine-1")
    assert pool.participants == tuple(tuple(client.identities) for client in clients)
    assert set(vars(pool)) == {"engine_ids", "participants"}
    for invalid in ([], descriptions + descriptions):
        with pytest.raises(ValueError, match="Missing or duplicate original receiver identities"):
            session.receiver_pool(invalid)
    for selected_clients, selected_pool in (
        (clients[:1], pool),
        (clients, replace(pool, engine_ids=pool.engine_ids[:1])),
        (clients, replace(pool, participants=pool.participants[:1])),
    ):
        with pytest.raises(ValueError, match="client and pool sizes differ"):
            asyncio.run(session.activate_publication(selected_clients, selected_pool, publication))
    assert not events
    assert all(client.args is None for client in clients)


client = test_sglang_api_client.client
recorder = test_sglang_api_client.recorder


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"release_state": True}, id="release-state"),
        pytest.param({"flush_cache": False}, id="no-flush"),
        pytest.param({"abort_all_requests": True}, id="abort-inflight"),
        pytest.param({"weight_version": "actor-42"}, id="model-version"),
        pytest.param({"validate_payload_hash": False}, id="skip-payload-hash"),
    ],
)
async def test_gpu_delta_request_identity_and_cache_policy(client, recorder, options):
    request = {
        "session_id": "publication-1",
        "manifest_path": "/checkpoint/gpu-delta/manifest.json",
        "manifest_sha256": "a" * 64,
        "stream_id": "stream-1",
        "base_version": 2,
        "target_version": 3,
        "plan_digest": "b" * 64,
        "participants": [{"engine_id": "engine", "rank_id": "rank"}],
    }
    await client.update_weights_from_gpu_delta(**request, **options)
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{test_sglang_api_client.SERVER_URL}/update_weights_from_gpu_delta")
    assert kwargs["json"] == {
        **request,
        "release_state": False,
        "flush_cache": True,
        "abort_all_requests": False,
        "validate_payload_hash": True,
        **options,
    }
    assert len(recorder.calls) == 1
