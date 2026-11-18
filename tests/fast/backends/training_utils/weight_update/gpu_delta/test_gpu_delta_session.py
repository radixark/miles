"""Independent engine update requests and failure boundaries without GPUs or HTTP."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from miles.backends.training_utils.weight_update.protocols.gpu_delta import session


def _setup(failure=None, failed_engine=1, payload_checksum_format="sha256"):
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
        "base_stream_id": None,
        "base_version": 0,
        "target_version": 1,
        "plan_digest": "canonical-checkpoint-digest",
        "payload_checksum_format": payload_checksum_format,
    }
    return clients, descriptions, publication, events


class _Engine:
    def __init__(self, index, identities, events, failure, failed_engine):
        self.index, self.identities, self.events, self.failure = index, identities, events, failure
        self.failed_engine = failed_engine
        self.args = None
        self.calls = 0
        self.completed = None
        self.info_queries = 0
        self.timings = {"prepare_s": 10.0 + index, "apply_s": 2.0, "resume_s": 3.0, "activation_s": 15.0 + index}

    async def _make_request(self, endpoint, payload):
        # The GPU-delta controls post through the engine client's request helper; ``args`` records the wire payload.
        return await getattr(self, endpoint)(**payload)

    async def get_gpu_delta_info(self, engine_id, session_id):
        self.info_queries += 1
        assert (engine_id, session_id) == (self.identities[0]["engine_id"], self.args["session_id"])
        state = "RESUMED" if self.completed else "IDLE"
        version = 1 if self.completed else 0
        if self.failure == "applied_reply":
            state = "APPLIED"
        rows = [
            {
                "identity": identity,
                "state": state,
                "stream_id": "stream" if version else None,
                "version": version,
                "plan_digest": self.args["plan_digest"],
            }
            for identity in self.identities
        ]
        return {"participants": rows, "activation": self.completed}

    async def update_weights_from_gpu_delta(self, **kwargs):
        self.calls += 1
        assert self.calls == 1 or (self.failure == "before" and self.calls == 2 and self.info_queries == 1)
        self.args = kwargs
        assert list(kwargs["participants"]) == self.identities
        await asyncio.sleep(0.01 if self.index else 0)
        failure = self.failure if self.index == self.failed_engine else None
        if failure == "before" and self.calls == 1:
            raise httpx.ConnectError("request did not arrive")
        if failure == "transport":
            error = RuntimeError("update request lost")
            error.add_note("response.text='TimeoutError: '")  # As SGLangApiClient attaches a receiver's 409 body.
            raise error
        self.events.append((self.index, "prepared"))
        provenance = {
            key: kwargs[key]
            for key in (
                "session_id",
                "stream_id",
                "base_stream_id",
                "manifest_sha256",
                "base_version",
                "target_version",
                "plan_digest",
            )
        }
        provenance.update(reverse=False, committed_stream_id=kwargs["stream_id"], version=kwargs["target_version"])
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
            raise httpx.ReadError("apply reply lost")
        self.events.append((self.index, "resumed"))
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
        self.completed = {
            "success": True,
            "message": "",
            "participants": resumed,
            "applied_participants": applied,
            "timings": self.timings,
        }
        if failure == "reply":
            raise httpx.ReadError("reply lost after resume")
        return self.completed


@pytest.mark.parametrize("weight_version,payload_checksum_format", [(None, "sha256"), ("42", "none")])
def test_fast_engine_resumes_while_other_engine_still_prepares(monkeypatch, weight_version, payload_checksum_format):
    # Requesters follow the writer's recorded hash policy, never their own environment.
    monkeypatch.setenv("GPU_DELTA_SKIP_PAYLOAD_HASH", "0" if payload_checksum_format == "none" else "1")
    clients, descriptions, publication, events = _setup(payload_checksum_format=payload_checksum_format)
    identities = session.engine_identities(descriptions)
    result = asyncio.run(
        session.activate_publication(clients, identities, publication, session_id="s", weight_version=weight_version)
    )
    for client in clients:
        assert client.calls == 1
        # An unset weight version stays off the wire, like every unset option.
        assert client.args == {
            **{key: value for key, value in publication.items() if key not in ("codec", "payload_checksum_format")},
            "session_id": "s",
            "participants": tuple(client.identities),
            **({} if weight_version is None else {"weight_version": weight_version}),
            "validate_payload_hash": payload_checksum_format == "sha256",
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


@pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
def test_rollout_sync_requests_its_pause_mode(pause_mode):
    # Eval pins and recovery omit the mode and retain the receiver's retracting default.
    clients, descriptions, publication, _ = _setup()
    identities = session.engine_identities(descriptions)
    asyncio.run(session.activate_publication(clients, identities, publication, session_id="s", pause_mode=pause_mode))
    for client in clients:
        assert client.args["pause_mode"] == pause_mode
        assert "flush_cache" not in client.args


@pytest.mark.parametrize("failure,requests", [("before", 2), ("reply", 1)])
def test_lost_reply_settles_only_the_affected_engine(failure, requests):
    clients, descriptions, publication, events = _setup(failure)
    result = asyncio.run(
        session.activate_publication(clients, session.engine_identities(descriptions), publication, session_id="s")
    )
    assert [client.calls for client in clients] == [1, requests]
    assert [client.info_queries for client in clients] == [0, 1]
    assert events.count((1, "applied")) == 1
    assert result["failed_engines"] == {}
    assert len(result["resumed_receipts"]) == 4
    assert result["engine_timings"][1]["prepare_s"] == clients[1].timings["prepare_s"]


async def test_late_first_request_finishes_before_the_guarded_resend():
    clients, descriptions, publication, _ = _setup()
    engine = clients[0]
    real_update = engine.update_weights_from_gpu_delta
    attempts = 0

    async def delayed(**request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            engine.args = request
            raise TimeoutError("initial request still in transit")
        # The first delivery won the race after the source-state read. The server
        # rejects the duplicate; its original completed session still has the proof.
        await real_update(**request)
        raise httpx.HTTPStatusError(
            "duplicate", request=httpx.Request("POST", "http://engine"), response=httpx.Response(409)
        )

    engine.update_weights_from_gpu_delta = delayed
    result = await session.activate_publication(
        clients, session.engine_identities(descriptions), publication, session_id="s"
    )
    assert result["failed_engines"] == {}
    assert (attempts, engine.info_queries, engine.calls) == (2, 2, 1)


@pytest.mark.parametrize("failure", ["transport", "applied_reply"])
@pytest.mark.parametrize("ft", [True, False])
def test_rejected_or_unsettled_update_is_not_replayed(failure, ft):
    clients, descriptions, publication, events = _setup(failure)
    activation = session.activate_publication(
        clients, session.engine_identities(descriptions), publication, session_id="s", fault_tolerant=ft
    )
    if ft:
        result = asyncio.run(activation)
        message = result["failed_engines"]["engine-1"]
        assert ("response.text=" if failure == "transport" else "unsettled weights") in message
        assert result["succeeded_engine_ids"] == ["engine-0"]
    else:
        with pytest.raises(RuntimeError, match="requires receiver recovery"):
            asyncio.run(activation)
    assert [client.calls for client in clients] == [1, 1]
    assert (0, "resumed") in events


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
            clients, session.engine_identities(descriptions), publication, session_id="s", fault_tolerant=True
        )
    )
    assert set(result["failed_engines"]) == {"engine-1"}
    assert result["succeeded_engine_ids"] == ["engine-0"]
    assert len(result["resumed_receipts"]) == 2
    assert [client.calls for client in clients] == [1, 1]


def test_engine_identities_contain_only_identity_and_reject_duplicate_membership():
    clients, descriptions, publication, events = _setup()
    identities = session.engine_identities(descriptions)
    assert identities.engine_ids == ("engine-0", "engine-1")
    assert identities.participants == tuple(tuple(client.identities) for client in clients)
    assert set(vars(identities)) == {"engine_ids", "participants"}
    for invalid in ([], descriptions + descriptions):
        with pytest.raises(ValueError, match="Missing or duplicate original receiver identities"):
            session.engine_identities(invalid)
    for selected_clients, selected in (
        (clients[:1], identities),
        (clients, replace(identities, engine_ids=identities.engine_ids[:1])),
        (clients, replace(identities, participants=identities.participants[:1])),
    ):
        with pytest.raises(ValueError, match="client and engine identity counts differ"):
            asyncio.run(session.activate_publication(selected_clients, selected, publication))
    assert not events
    assert all(client.args is None for client in clients)
