"""Independent engine update requests and failure boundaries without GPUs or HTTP."""

import asyncio
import copy
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
    _, digest = session.merge_plans(descriptions)
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
        self.calls = 0
        self.timings = {"prepare_s": 10.0 + index, "apply_s": 2.0, "resume_s": 3.0, "activation_s": 15.0 + index}

    async def update_weights_from_gpu_delta(self, **kwargs):
        self.calls += 1
        self.args = kwargs
        assert list(kwargs["participants"]) == self.identities
        await asyncio.sleep(0.01 if self.index else 0)
        failure = self.failure if self.index == self.failed_engine else None
        if failure == "transport":
            raise RuntimeError("update request lost")
        self.events.append((self.index, "prepared"))
        if failure == "rejected":
            return {"success": False, "message": "update rejected"}
        applied = [
            {
                "identity": identity,
                "state": "APPLIED",
                "result": {"applied": True, "target_version": kwargs["target_version"]},
            }
            for identity in self.identities
        ]
        self.events.append((self.index, "resumed"))
        if failure == "reply":
            raise RuntimeError("update reply lost")
        resumed = [
            {
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
    cohort = session.negotiate_cohort(descriptions)
    monkeypatch.setattr(session, "merge_plans", lambda *args, **kwargs: pytest.fail("Immutable plan renegotiated"))
    result = asyncio.run(
        session.activate_publication(clients, cohort, publication, session_id="s", weight_version=weight_version)
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
    with pytest.raises(RuntimeError, match="update"):
        asyncio.run(
            session.activate_publication(clients, session.negotiate_cohort(descriptions), publication, session_id="s")
        )
    assert [client.calls for client in clients] == [1, 1]
    assert (1 - failed_engine, "resumed") in events
    # A lost reply may hide a completed update. Neither that ambiguity nor an
    # explicit failure authorizes another update request or a client abort.
    assert ((failed_engine, "resumed") in events) == (failure == "reply")


def test_common_plan_deduplicates_replicas_and_rejects_invalid_cohorts_before_preparation():
    clients, descriptions, publication, events = _setup()
    plan, digest = session.merge_plans(descriptions)
    assert len(plan) == 1 and len(plan[0]["views"]) == 2
    assert digest == publication["plan_digest"]
    broken = copy.deepcopy(descriptions)
    broken[1]["participants"][0]["plan"]["tensors"][0]["views"][0]["slices"] = [[1, 2]]
    with pytest.raises(ValueError, match="conflict"):
        asyncio.run(session.activate_publication(clients, session.negotiate_cohort(broken), publication))
    for invalid in ([], descriptions + descriptions):
        with pytest.raises(ValueError, match="Missing or duplicate original receiver identities"):
            session.negotiate_cohort(invalid)
    cohort = session.negotiate_cohort(descriptions)
    for selected_clients, selected_cohort in (
        (clients[:1], cohort),
        (clients, replace(cohort, engine_ids=cohort.engine_ids[:1])),
        (clients, replace(cohort, participants=cohort.participants[:1])),
    ):
        with pytest.raises(ValueError, match="client and cohort sizes differ"):
            asyncio.run(session.activate_publication(selected_clients, selected_cohort, publication))
    assert not events
    assert all(client.args is None for client in clients)


@pytest.mark.parametrize("field,value", [("format", "fp8-block128-ue8m0"), ("checkpoint_dtype", "F32")])
def test_common_plan_binds_optional_canonical_representation(field, value):
    _, descriptions, _, _ = _setup()
    previous, previous_digest = session.merge_plans(descriptions)
    for description in descriptions:
        for participant in description["participants"]:
            participant["plan"]["tensors"][0][field] = value
    plan, digest = session.merge_plans(descriptions)
    assert field not in previous[0] and plan[0][field] == value and digest != previous_digest
    del descriptions[1]["participants"][0]["plan"]["tensors"][0][field]
    for ordered in (descriptions, descriptions[::-1]):
        with pytest.raises(ValueError, match="Receiver canonical plans differ"):
            session.merge_plans(ordered)


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
