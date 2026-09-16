"""The token trajectory collector over a real TinkerService (FakeBackend) with a character tokenizer.

Reused, not reimplemented: ``tests/fast/tinker/harness.py`` (``make_service``, ``FakeBackend.sample`` whose ``fail_on["sample"]``
doubles as the canned result) and the real checkpoint writer behind ``oai_fakes.write_sampler``.
"""

import pytest
from tests.fast.tinker.harness import make_service
from tests.fast.tinker.oai_fakes import (
    GEN,
    OTHER_TENANT,
    ROLE_IDS,
    SAMPLER,
    TENANT,
    THINK_ON,
    FakeTokenizer,
    write_sampler,
)

from miles.tinker.core.tinker_session_server import (
    SamplingBackendError,
    TrajectoryCollector,
    UnknownSessionError,
    to_sample_payload,
)
from miles.tinker.core.types import OwnershipError, UserInputError

BASE = "base"
HI = {"sequences": [{"tokens": [104, 105], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}]}
SKELETON = pytest.mark.skip(reason="skeleton: lands with the client plug-ins (action 3)")


@pytest.fixture
def collector(tmp_path):
    service = make_service(tmp_path, vocab_size=1000)
    write_sampler(tmp_path, BASE)
    now = [1000.0]
    collector = TrajectoryCollector(
        service, FakeTokenizer(), session_ttl_s=600.0, chat_template_kwargs=None, clock=lambda: now[0]
    )
    collector.now = now  # tests advance time through this handle
    return collector


async def _chat(collector, session_id, tenant=TENANT, **overrides):
    request = {"model": SAMPLER, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    request.update(overrides)
    return await collector.chat(request, session_id=session_id, tenant=tenant)


def _sample_calls(collector):
    return collector.service.backend.named("sample")


async def test_render_prompt_uses_generation_prompt(collector):
    """render_prompt applies the HF chat template with add_generation_prompt=True and the configured kwargs."""
    collector.bind("s1", TENANT, SAMPLER)
    await _chat(collector, "s1")
    assert _sample_calls(collector)[0]["payload"]["prompt_tokens"] == [ROLE_IDS["user"], ord("h"), ord("i"), GEN]

    collector.chat_template_kwargs = {"enable_thinking": True}
    await _chat(collector, "s1")
    assert _sample_calls(collector)[1]["payload"]["prompt_tokens"][-2:] == [THINK_ON, GEN]

    await _chat(collector, "s1", chat_template_kwargs={"enable_thinking": False})
    assert THINK_ON not in _sample_calls(collector)[2]["payload"]["prompt_tokens"]


def test_sample_payload_matches_tinker_sample():
    """to_sample_payload produces prompt_tokens, num_samples and sampling_params exactly as decode_sample_request does for Tinker sample; max_tokens is required; stop=[] is dropped."""
    payload = to_sample_payload(
        [1, 2, 3], {"max_tokens": 16, "temperature": 0.7, "top_p": 0.9, "stop": "END", "seed": 3}, SAMPLER
    )
    assert payload == {
        "model_path": SAMPLER,
        "num_samples": 1,
        "prompt_tokens": [1, 2, 3],
        "sampling_params": {"max_tokens": 16, "temperature": 0.7, "top_p": 0.9, "seed": 3, "stop": ["END"]},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }
    assert to_sample_payload([1], {"max_completion_tokens": 4}, None)["sampling_params"] == {
        "max_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert "stop" not in to_sample_payload([1], {"max_tokens": 4, "stop": []}, None)["sampling_params"]
    for bad in ({}, {"max_tokens": 0}, {"max_tokens": 4, "n": 2}, {"max_tokens": 4, "stream": True}):
        with pytest.raises(UserInputError):
            to_sample_payload([1], bad, None)


async def test_chat_samples_through_tinker_service(collector):
    """chat() goes through TinkerService.submit_sample (FakeBackend.sample sees lora_path=M@V) and records input_ids as rendered, output_ids/logprobs as returned."""
    collector.service.backend.fail_on["sample"] = HI
    collector.bind("s1", TENANT, SAMPLER)
    response = await _chat(collector, "s1", temperature=0.5)
    assert response["object"] == "chat.completion"
    assert response["choices"][0] == {
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop",
    }
    assert response["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}

    (call,) = _sample_calls(collector)
    assert call["lora_name"] == "m1@v0" and call["lora_path"].endswith("/m1/sampler_weights/v0")
    assert call["payload"]["sampling_params"] == {"max_tokens": 8, "temperature": 0.5, "top_p": 1.0}

    trajectory = collector.trajectory("s1", TENANT)
    assert trajectory["model_path"] == SAMPLER
    (turn,) = trajectory["turns"]
    assert turn["input_ids"] == call["payload"]["prompt_tokens"]
    assert turn["output_ids"] == [104, 105] and turn["logprobs"] == [-0.1, -0.2] and turn["finish_reason"] == "stop"


async def test_base_model_session_samples_without_an_adapter(collector):
    """No model (or the base model's name) binds to the frozen base: submit_sample runs with lora_name=None."""
    collector.bind("s1", TENANT, BASE)
    assert collector.sessions["s1"].model_path is None
    await _chat(collector, "s1", model=None)
    (call,) = _sample_calls(collector)
    assert call["lora_name"] is None and call["lora_path"] is None


async def test_auto_register_requires_bearer(collector):
    """A new session id auto-registers with a valid bearer and raises UnknownSessionError without one."""
    with pytest.raises(UnknownSessionError):
        await _chat(collector, "fresh", tenant=None)
    assert "fresh" not in collector.sessions

    await _chat(collector, "fresh", tenant=TENANT)
    assert collector.sessions["fresh"].tenant == TENANT
    assert collector.sessions["fresh"].model_path == SAMPLER


async def test_prebound_session_accepts_dummy_key(collector):
    """After bind(), chat requests without a bearer are served with the owner's tenant and recorded."""
    collector.bind("s1", TENANT, SAMPLER)
    await _chat(collector, "s1", tenant=None, model="model")
    await _chat(collector, "s1", tenant="dummy", model="openai/model")
    assert len(collector.trajectory("s1", TENANT)["turns"]) == 2
    assert all(call["lora_name"] == "m1@v0" for call in _sample_calls(collector))


async def test_bound_session_rejects_other_model(collector):
    """A request naming a different tinker:// path than the bound one gets a UserInputError (400)."""
    collector.bind("s1", TENANT, SAMPLER)
    with pytest.raises(UserInputError):
        await _chat(collector, "s1", model="tinker://m1/sampler_weights/v9")
    with pytest.raises(UserInputError):
        collector.bind("s1", TENANT, "tinker://m1/sampler_weights/v9")
    assert collector.trajectory("s1", TENANT)["turns"] == []


def test_get_and_delete_check_ownership(collector):
    """trajectory()/delete() raise OwnershipError for another tenant's key and UnknownSessionError afterwards."""
    collector.bind("s1", TENANT, SAMPLER)
    with pytest.raises(OwnershipError):
        collector.trajectory("s1", OTHER_TENANT)
    with pytest.raises(OwnershipError):
        collector.delete("s1", OTHER_TENANT)
    with pytest.raises(OwnershipError):
        collector.bind("s1", OTHER_TENANT, SAMPLER)
    collector.delete("s1", TENANT)
    with pytest.raises(UnknownSessionError):
        collector.trajectory("s1", TENANT)


def test_unknown_sampler_path_is_user_error(collector):
    """bind() with a tinker:// path lacking META.json is a UserInputError (resolve_sampler_checkpoint), another tenant's path an OwnershipError."""
    with pytest.raises(UserInputError):
        collector.bind("s1", TENANT, "tinker://m1/sampler_weights/missing")
    with pytest.raises(OwnershipError):
        collector.bind("s2", OTHER_TENANT, SAMPLER)
    with pytest.raises(UserInputError):
        collector.bind("s3", TENANT, "some-other-model")
    with pytest.raises(UserInputError):
        collector.bind("s4", "", SAMPLER)
    assert collector.sessions == {}


async def test_sampling_session_id_binds_to_its_sampler(collector):
    """bind(sampling_session_id=...) resolves the sampler path through service.get_sampler, ownership included."""
    service = collector.service
    tinker_session = service.create_session(TENANT)
    sampling_session_id = service.create_sampling_session(
        TENANT, {"session_id": tinker_session, "sampling_session_seq_id": 0, "model_path": SAMPLER}
    )
    assert collector.bind("s1", TENANT, sampling_session_id=sampling_session_id).model_path == SAMPLER
    with pytest.raises(OwnershipError):
        collector.bind("s2", OTHER_TENANT, sampling_session_id=sampling_session_id)


async def test_backend_failure_records_no_turn(collector):
    """A failed sampling future surfaces as an error and leaves the session without a half turn."""
    collector.bind("s1", TENANT, SAMPLER)
    collector.service.backend.fail_on["sample"] = {"error": "engine aborted"}
    with pytest.raises(SamplingBackendError):
        await _chat(collector, "s1")
    assert collector.trajectory("s1", TENANT)["turns"] == []

    with pytest.raises(UserInputError):  # token ids outside the vocab are rejected by validate_sample_payload
        await _chat(collector, "s1", messages=[{"role": "user", "content": "中"}])
    assert collector.trajectory("s1", TENANT)["turns"] == []


def test_sweep_expires_idle_sessions(collector):
    """sweep() drops sessions idle longer than session_ttl_s and keeps the others."""
    collector.bind("old", TENANT, SAMPLER)
    collector.now[0] += collector.session_ttl_s - 1
    collector.bind("young", TENANT, SAMPLER)
    collector.now[0] += 1
    assert collector.sweep() == 1
    assert set(collector.sessions) == {"young"}


@SKELETON
def test_turns_round_trip_through_cookbook():
    """Two chained turns exported by GET /oai/sessions/{sid} become one Datum through turns_to_trajectory + trajectory_to_data; a broken prefix becomes two."""
