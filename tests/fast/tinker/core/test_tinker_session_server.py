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
    SessionLimitError,
    TrajectoryCollector,
    UnknownSessionError,
    to_sample_payload,
)
from miles.tinker.core.types import OwnershipError, UserInputError

BASE = "base"
HI = {"sequences": [{"tokens": [104, 105], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}]}


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


async def test_exported_turns_round_trip_through_cookbook(collector):
    """GET /oai/sessions/{sid} turns feed turns_to_trajectory + trajectory_to_data unchanged (the client-side test file covers merge vs split)."""
    pytest.importorskip("tinker_cookbook")
    from examples.multi_lora.harbor_tinker.harbor_env import turns_to_trajectory
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    collector.bind("s1", TENANT, SAMPLER)
    await _chat(collector, "s1")
    await _chat(collector, "s1", messages=[{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}])
    trajectory = turns_to_trajectory(collector.trajectory("s1", TENANT)["turns"])
    assert [transition.ac.tokens for transition in trajectory.transitions] == [[1, 2], [1, 2]]  # FakeBackend's output
    datums = trajectory_to_data(trajectory, traj_advantage=1.0)
    assert len(datums) == 2  # the character template re-renders history without the previous output: split per turn
    assert datums[1].loss_fn_inputs["target_tokens"].tolist()[-2:] == [1, 2]  # the engine's ids become the targets


# --- multi-tenant hardening ------------------------------------------------------


async def test_foreign_key_on_a_bound_session_is_forbidden(collector):
    """Another tenant's real key cannot sample on (or record into) a session it does not own; nothing is recorded."""
    collector.bind("s1", TENANT, SAMPLER)
    with pytest.raises(OwnershipError):
        await _chat(collector, "s1", tenant=OTHER_TENANT)
    assert collector.trajectory("s1", TENANT)["turns"] == [] and _sample_calls(collector) == []
    await _chat(collector, "s1", tenant=TENANT)  # the owner's own key is fine
    assert len(collector.trajectory("s1", TENANT)["turns"]) == 1


async def test_placeholder_key_cannot_register_or_bind(collector):
    """The harness's dummy key is treated like no key: it serves bound sessions but creates nothing."""
    with pytest.raises(UnknownSessionError):
        await _chat(collector, "fresh", tenant="dummy")
    with pytest.raises(UserInputError):
        collector.bind("fresh", "dummy", SAMPLER)
    assert collector.sessions == {}


def test_session_id_is_validated(collector):
    """Session ids are URL path segments: 1-128 chars of [A-Za-z0-9._:-], alphanumeric first."""
    for bad in ("", "-leading", "a/b", "x" * 129, "sp ace"):
        with pytest.raises(UserInputError):
            collector.bind(bad, TENANT, SAMPLER)
    collector.bind("harbor-" + "a" * 32 + ":1.0", TENANT, SAMPLER)


async def test_session_and_turn_caps(collector):
    """A tenant's open sessions and a session's turns are capped (429); DELETE frees capacity; other tenants are unaffected."""
    collector.max_sessions_per_tenant = 2
    collector.max_turns_per_session = 1
    collector.bind("a", TENANT, SAMPLER)
    collector.bind("b", TENANT, SAMPLER)
    with pytest.raises(SessionLimitError):
        collector.bind("c", TENANT, SAMPLER)
    collector.bind("a", TENANT, SAMPLER)  # re-binding an open session is not a new one
    collector.bind("other", OTHER_TENANT, BASE)  # the cap is per tenant
    collector.delete("b", TENANT)
    collector.bind("c", TENANT, SAMPLER)

    await _chat(collector, "a")
    with pytest.raises(SessionLimitError):
        await _chat(collector, "a")
    assert len(collector.trajectory("a", TENANT)["turns"]) == 1 and len(_sample_calls(collector)) == 1


def test_sweep_leaves_in_flight_sessions_alone(collector):
    """A session with a sample running is not swept even when idle past the TTL; it is swept once the sample settled."""
    collector.bind("busy", TENANT, SAMPLER)
    collector.sessions["busy"].in_flight = 1
    collector.now[0] += collector.session_ttl_s + 1
    assert collector.sweep() == 0 and "busy" in collector.sessions
    collector.sessions["busy"].in_flight = 0
    assert collector.sweep() == 1 and collector.sessions == {}


async def test_turns_are_stored_compactly_and_exported_as_lists(collector):
    """Token ids and logprobs live in arrays (4 or 8 bytes each) but the export is plain JSON lists."""
    collector.bind("s1", TENANT, SAMPLER)
    await _chat(collector, "s1")
    (turn,) = collector.sessions["s1"].turns
    assert turn.input_ids.typecode == "i" and turn.output_ids.typecode == "i" and turn.logprobs.typecode == "d"
    (exported,) = collector.trajectory("s1", TENANT)["turns"]
    assert (
        isinstance(exported["input_ids"], list)
        and exported["output_ids"] == [1, 2]
        and exported["logprobs"] == [0.0, 0.0]
    )
