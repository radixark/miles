"""The session server deletes the output store objects of a session that its client did not take.

Drives v1 and v2 cores in-process with a backend that answers like SGLang for an opted-in request:
a ref in meta_info, an abort answered 200 with finish_reason abort and its ref all the same.
"""

import json

import httpx
import orjson
import pytest
from fastapi import FastAPI
from starlette.responses import Response
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.utils.output_store.test_dropped import sample_owning
from tests.fast.utils.output_store.test_plan import R3, Ref, Spec, fake_store_refs

from miles.rollout.session import sessions
from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import OutputStoreContractError, UpstreamGenerationAbortedError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.store_refs import SessionStoreRefs, response_store_ref
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.output_store.plan import OUTPUT_STORE_REF_KEY, OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.processing_utils import load_tokenizer

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
LAYERS, TOPK = 2, 4


class _Store:
    def __init__(self) -> None:
        self.deleted: list[list[str]] = []

    def delete_many(self, refs) -> None:
        self.deleted.append(sorted(ref.key for ref in refs))


class _Backend:
    """Answers each chat with "ok" and a fresh ref; ``finish_reasons`` scripts the turns."""

    def __init__(self, tokenizer, finish_reasons: list[str]) -> None:
        self.output_ids = tokenizer.encode("ok", add_special_tokens=False) + [
            tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        self.finish_reasons = list(finish_reasons)
        self.num_calls = 0
        self.during_proxy = None
        # Answer like SGLang behind a router that does not forward the opt-in.
        self.drop_opt_in = False

    async def do_proxy(self, request, path, *, body, headers) -> dict:
        self.num_calls += 1
        payload = orjson.loads(body)
        meta_info = {
            "output_token_logprobs": [(-0.1, token_id) for token_id in self.output_ids],
            "completion_tokens": len(self.output_ids),
        }
        if payload.get(OUTPUT_STORE_REQUEST_KEY) and not self.drop_opt_in:
            rows = len(payload["input_ids"]) + len(self.output_ids) - 1
            meta_info[OUTPUT_STORE_REF_KEY] = Ref(
                f"key-{self.num_calls}", (Spec(R3, "int32", (rows, LAYERS, TOPK)),)
            ).to_builtins()
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": self.finish_reasons.pop(0),
            "meta_info": meta_info,
        }
        if self.during_proxy is not None:
            self.during_proxy()
        return {
            "request_body": body,
            "response_body": orjson.dumps({"id": "chat", "choices": [choice]}),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(HF_CHECKPOINT, chat_template_path=None, trust_remote_code=True)


def build_core(monkeypatch, tokenizer, *, v2: bool, finish_reasons: list[str]):
    fake_store_refs(monkeypatch)
    store = _Store()
    config = make_session_server_config(
        hf_checkpoint=HF_CHECKPOINT,
        apply_chat_template_kwargs={"enable_thinking": False},
        use_rollout_routing_replay=True,
        num_layers=LAYERS,
        moe_router_topk=TOPK,
        output_store_backend_config={"master_server_address": "unused"},
        use_session_server="v2" if v2 else None,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves" if v2 else None,
        session_sample_postprocessor_path=(
            "miles.rollout.session.v2.postprocessor_hub.default_postprocess" if v2 else None
        ),
    )
    tito_tokenizer = get_tito_tokenizer(tokenizer, tokenizer_type="default", chat_template_kwargs={})
    backend = _Backend(tokenizer, finish_reasons)
    if v2:
        core = SessionCoreV2(
            backend, SessionRegistryV2(tokenizer, tito_tokenizer=tito_tokenizer), config, output_store=store
        )
    else:
        core = SessionCore(
            backend, SessionRegistry(tokenizer, tito_tokenizer=tito_tokenizer), config, output_store=store
        )
    return core, backend, store


async def chat(core, session_id: str):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    return await core.chat_completions(session_id, method="POST", query="", headers={}, body=body)


async def new_session(core) -> str:
    return json.loads((await core.create_session()).body)["session_id"]


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
async def test_delete_removes_what_samples_did_not_hand_over(monkeypatch, tokenizer, v2):
    core, _, store = build_core(monkeypatch, tokenizer, v2=v2, finish_reasons=["abort", "stop"])
    session_id = await new_session(core)

    # The abort becomes a 503 and no record, but SGLang wrote its object.
    with pytest.raises(UpstreamGenerationAbortedError):
        await chat(core, session_id)
    assert (await chat(core, session_id)).status_code == 200
    assert (await core.collect_samples(session_id, max_seq_len=None)).status_code == 200
    assert store.deleted == []

    assert (await core.delete_session(session_id, samples_received=True)).status_code == 204
    assert store.deleted == [["key-1"]]


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
async def test_a_client_without_the_reply_has_everything_deleted(monkeypatch, tokenizer, v2):
    core, _, store = build_core(monkeypatch, tokenizer, v2=v2, finish_reasons=["abort", "stop"])
    session_id = await new_session(core)
    with pytest.raises(UpstreamGenerationAbortedError):
        await chat(core, session_id)
    await chat(core, session_id)
    # The reply was built, but the client timed out before it arrived.
    assert (await core.collect_samples(session_id, max_seq_len=None)).status_code == 200

    await core.delete_session(session_id, samples_received=False)

    assert store.deleted == [["key-1", "key-2"]]


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
async def test_a_later_reply_never_takes_back_an_earlier_one(monkeypatch, tokenizer, v2):
    core, _, store = build_core(monkeypatch, tokenizer, v2=v2, finish_reasons=["stop"])
    session_id = await new_session(core)
    await chat(core, session_id)
    assert (await core.collect_samples(session_id, max_seq_len=None)).status_code == 200
    # Truncated to nothing: an empty reply, while the client still holds the first one.
    assert (await core.collect_samples(session_id, max_seq_len=1)).status_code == 200

    await core.delete_session(session_id, samples_received=True)

    assert store.deleted == []


async def test_the_delete_route_passes_the_clients_report_on(monkeypatch, tokenizer):
    reports: list[bool] = []

    async def delete_session(self, session_id, *, samples_received):
        reports.append(samples_received)
        return Response(status_code=204)

    monkeypatch.setattr(SessionCore, "delete_session", delete_session)
    monkeypatch.setattr(sessions, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    app = FastAPI()
    config = make_session_server_config(hf_checkpoint=HF_CHECKPOINT, apply_chat_template_kwargs={})
    sessions.setup_session_routes(app, backend=None, config=config)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://session") as client:
        assert (await client.delete("/sessions/sid")).status_code == 204
        assert (await client.delete("/sessions/sid?samples_received=false")).status_code == 204

    assert reports == [True, False]


async def test_the_session_routes_hand_the_run_store_to_the_core(monkeypatch, tokenizer):
    store = _Store()
    connected, stores = [], []

    async def delete_session(self, session_id, *, samples_received):
        stores.append(self._output_store)
        return Response(status_code=204)

    monkeypatch.setattr(SessionCore, "delete_session", delete_session)
    monkeypatch.setattr(sessions, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(sessions, "connect_output_store", lambda config: connected.append(config) or store)
    app = FastAPI()
    store_config = {"master_server_address": "unused"}
    config = make_session_server_config(
        hf_checkpoint=HF_CHECKPOINT, apply_chat_template_kwargs={}, output_store_backend_config=store_config
    )
    sessions.setup_session_routes(app, backend=None, config=config)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://session") as client:
        assert (await client.delete("/sessions/sid")).status_code == 204

    assert connected == [store_config]
    assert stores == [store]


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
async def test_a_session_never_collected_deletes_everything(monkeypatch, tokenizer, v2):
    core, _, store = build_core(monkeypatch, tokenizer, v2=v2, finish_reasons=["stop", "stop"])
    session_id = await new_session(core)
    await chat(core, session_id)
    await chat(core, session_id)

    await core.delete_session(session_id, samples_received=True)

    assert store.deleted == [["key-1", "key-2"]]


async def test_a_response_arriving_after_delete_began_is_deleted_at_once(monkeypatch, tokenizer):
    core, backend, store = build_core(monkeypatch, tokenizer, v2=False, finish_reasons=["stop"])
    session_id = await new_session(core)
    session = core.registry.get_session(session_id)

    def delete_begins():
        session.closing = True

    backend.during_proxy = delete_begins
    await chat(core, session_id)

    assert store.deleted == [["key-1"]]
    assert session.store_refs.arrived == {}


def test_only_opted_in_successful_responses_carry_a_ref(monkeypatch):
    fake_store_refs(monkeypatch)
    choice = {"meta_info": {OUTPUT_STORE_REF_KEY: Ref("key", ()).to_builtins()}}
    body = orjson.dumps({"choices": [choice]})
    opted_in = {OUTPUT_STORE_REQUEST_KEY: True}

    assert response_store_ref(opted_in, {"status_code": 200, "response_body": body}).key == "key"
    assert response_store_ref({}, {"status_code": 200, "response_body": body}) is None
    assert response_store_ref(opted_in, {"status_code": 503, "response_body": body}) is None
    assert response_store_ref(opted_in, {"status_code": 200, "response_body": b"not json"}) is None


def test_an_opted_in_response_without_a_ref_breaks_the_store_contract(monkeypatch):
    fake_store_refs(monkeypatch)
    opted_in = {OUTPUT_STORE_REQUEST_KEY: True}

    def answered(finish_reason: str) -> dict:
        body = orjson.dumps({"choices": [{"finish_reason": finish_reason, "meta_info": {}}]})
        return {"status_code": 200, "response_body": body}

    with pytest.raises(StorePlanError, match="dropped the opt-in"):
        response_store_ref(opted_in, answered("stop"))
    # Aborted in SGLang's waiting queue: nothing was written, so there is no ref to take.
    assert response_store_ref(opted_in, answered("abort")) is None


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
async def test_a_dropped_opt_in_fails_the_turn_and_the_sessions_samples(monkeypatch, tokenizer, v2):
    core, backend, _ = build_core(monkeypatch, tokenizer, v2=v2, finish_reasons=["stop"])
    backend.drop_opt_in = True
    session_id = await new_session(core)

    with pytest.raises(OutputStoreContractError, match="dropped the opt-in"):
        await chat(core, session_id)
    # The agent may swallow the turn's error; /samples still fails with it although no record holds the turn.
    with pytest.raises(OutputStoreContractError, match="dropped the opt-in"):
        await core.collect_samples(session_id, max_seq_len=None)


def test_replies_hand_over_cumulatively():
    refs = SessionStoreRefs(arrived={key: Ref(key, ()) for key in ("a", "b", "c")})
    refs.hand_over([sample_owning("a")])
    refs.hand_over([sample_owning("b")])

    assert [ref.key for ref in refs.kept_by_session(samples_received=True)] == ["c"]
    assert [ref.key for ref in refs.kept_by_session(samples_received=False)] == ["a", "b", "c"]


async def test_a_failed_delete_still_completes_the_session_delete(monkeypatch, tokenizer, caplog):
    core, _, store = build_core(monkeypatch, tokenizer, v2=False, finish_reasons=["stop"])
    session_id = await new_session(core)
    await chat(core, session_id)

    def failing_delete_many(refs):
        raise RuntimeError("Mooncake remove failed")

    store.delete_many = failing_delete_many
    # A 500 here would send the client retrying into 404s for about a minute.
    assert (await core.delete_session(session_id, samples_received=True)).status_code == 204
    assert "Failed to delete 1 output store refs" in caplog.text
