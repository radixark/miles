from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_contract import SessionRequestContract
from miles.utils.chat_template_utils.tito_tokenizer import FixedTemplate, TITOTokenizer
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled

_ABSENT = object()


def _contract(
    *,
    tito_tokenizer: TITOTokenizer | None = None,
    launch_kwargs: dict | None = None,
    **launch_args,
) -> SessionRequestContract:
    tito_tokenizer = tito_tokenizer or TITOTokenizer(None, chat_template_kwargs=launch_kwargs)
    args = SimpleNamespace(**launch_args)
    return SessionRequestContract.from_settings(
        tito_tokenizer,
        force_return_routed_experts=getattr(args, "use_rollout_routing_replay", False),
        force_return_indexer_topk=getattr(args, "use_rollout_indexer_replay", False),
        lora_path=LORA_ADAPTER_NAME if is_lora_enabled(args) else None,
    )


def _body(**values) -> bytes:
    return json.dumps(values).encode()


def test_prepare_forces_miles_fields_and_preserves_unknown_fields():
    prepared = _contract(
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        lora_rank=8,
    ).prepare(
        _body(
            messages=[],
            custom_backend_field="keep",
            logprobs=False,
            return_meta_info=False,
            no_stop_trim=True,
            return_routed_experts=False,
            return_indexer_topk=False,
            lora_path="client-adapter",
        )
    )

    assert prepared.body == {
        "messages": [],
        "custom_backend_field": "keep",
        "logprobs": True,
        "return_meta_info": True,
        "no_stop_trim": False,
        "return_routed_experts": True,
        "return_indexer_topk": True,
        "lora_path": "miles_lora",
    }


def test_disabled_launch_features_preserve_client_fields():
    prepared = _contract().prepare(
        _body(
            messages=[],
            return_routed_experts=False,
            return_indexer_topk="client-indexer-value",
            lora_path="client-adapter",
        )
    )

    assert prepared.body["return_routed_experts"] is False
    assert prepared.body["return_indexer_topk"] == "client-indexer-value"
    assert prepared.body["lora_path"] == "client-adapter"


def test_finalize_silently_overwrites_client_input_ids():
    contract = _contract()
    prepared = contract.prepare(_body(messages=[], input_ids=[1, 2, 3]))

    outbound = contract.finalize(prepared, input_ids=[10, 11])

    assert prepared.body["input_ids"] == [1, 2, 3]
    assert outbound["input_ids"] == [10, 11]


def test_request_chat_template_kwargs_override_launch_for_renderer_and_backend():
    prepared = _contract(launch_kwargs={"enable_thinking": False}).prepare(
        _body(messages=[], chat_template_kwargs={"enable_thinking": True})
    )

    assert prepared.tito_tokenizer.chat_template_kwargs == {"enable_thinking": True}
    assert prepared.body["chat_template_kwargs"] == prepared.tito_tokenizer.chat_template_kwargs


def test_null_chat_template_kwargs_preserve_launch_defaults():
    prepared = _contract(launch_kwargs={"enable_thinking": False}).prepare(
        _body(messages=[], chat_template_kwargs=None)
    )

    assert prepared.tito_tokenizer.chat_template_kwargs == {"enable_thinking": False}
    assert prepared.body["chat_template_kwargs"] == {"enable_thinking": False}


def test_non_object_chat_template_kwargs_are_rejected():
    with pytest.raises(MessageValidationError, match="chat_template_kwargs must be an object"):
        _contract().prepare(_body(messages=[], chat_template_kwargs=[]))


class _FixedTokenizer(TITOTokenizer):
    FIXED_TEMPLATE = FixedTemplate(extra_kwargs={"preserve_thinking": True})


def test_matching_fixed_template_kwarg_is_accepted():
    prepared = _contract(tito_tokenizer=_FixedTokenizer(None)).prepare(
        _body(messages=[], chat_template_kwargs={"preserve_thinking": True})
    )

    assert prepared.body["chat_template_kwargs"] == {"preserve_thinking": True}
    assert prepared.tito_tokenizer.chat_template_kwargs == {"preserve_thinking": True}


def test_conflicting_fixed_template_kwarg_is_rejected():
    with pytest.raises(MessageValidationError, match="conflicts with the value registered"):
        _contract(tito_tokenizer=_FixedTokenizer(None)).prepare(
            _body(messages=[], chat_template_kwargs={"preserve_thinking": False})
        )


class _AliasTokenizer(TITOTokenizer):
    chat_template_kwarg_aliases = frozenset({"thinking", "enable_thinking"})


def test_existing_chat_template_alias_group_merge_is_preserved():
    tokenizer = _AliasTokenizer(None, chat_template_kwargs={"thinking": False})
    prepared = _contract(tito_tokenizer=tokenizer).prepare(
        _body(messages=[], chat_template_kwargs={"enable_thinking": True})
    )

    assert prepared.tito_tokenizer.chat_template_kwargs == {"enable_thinking": True}
    assert prepared.body["chat_template_kwargs"] == {"enable_thinking": True}


def test_stream_intent_is_client_only():
    prepared = _contract().prepare(_body(messages=[], stream=True, stream_options={"include_usage": True}))

    assert prepared.client_stream is True
    assert "stream" not in prepared.body
    assert "stream_options" not in prepared.body


def test_empty_body_preserves_existing_empty_object_behavior():
    assert _contract().prepare(b"").body == {
        "logprobs": True,
        "return_meta_info": True,
        "no_stop_trim": False,
    }


def test_malformed_json_is_a_message_validation_error():
    with pytest.raises(MessageValidationError, match="invalid JSON body"):
        _contract().prepare(b"{not json")


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        (b"[]", TypeError),
        (b"null", AttributeError),
        (b"1", AttributeError),
        (b'"text"', AttributeError),
        (b"\xff", UnicodeDecodeError),
    ],
)
def test_non_object_and_non_utf8_json_keep_existing_error_boundary(body, error_type):
    with pytest.raises(error_type):
        _contract().prepare(body)


@pytest.mark.parametrize(
    (
        "launch_replay",
        "use_addition_r3",
        "request_fields",
        "expected_return_routed_experts",
        "expected_start_len",
    ),
    [
        (False, False, {}, _ABSENT, _ABSENT),
        (False, False, {"routed_experts_start_len": 999}, _ABSENT, 999),
        (False, True, {"return_routed_experts": True, "routed_experts_start_len": 999}, True, 2),
        (True, True, {"return_routed_experts": False, "routed_experts_start_len": 999}, True, 2),
        (False, True, {"return_routed_experts": False, "routed_experts_start_len": 999}, False, 999),
        (False, False, {"return_routed_experts": True, "routed_experts_start_len": 999}, True, 999),
    ],
)
def test_routed_experts_start_len_preserves_existing_conditional_precedence(
    launch_replay,
    use_addition_r3,
    request_fields,
    expected_return_routed_experts,
    expected_start_len,
):
    contract = _contract(use_rollout_routing_replay=launch_replay)
    prepared = contract.prepare(_body(messages=[], **request_fields))
    core = object.__new__(SessionCore)
    core.request_contract = contract
    core.use_addition_r3 = use_addition_r3

    outbound = core._finalize_chat_request(
        prepared,
        checkpoint_token_ids=[10, 11, 12],
        prompt_token_ids=[10, 11, 12, 13],
    )

    if expected_return_routed_experts is _ABSENT:
        assert "return_routed_experts" not in outbound
    else:
        assert outbound["return_routed_experts"] == expected_return_routed_experts
    if expected_start_len is _ABSENT:
        assert "routed_experts_start_len" not in outbound
    else:
        assert outbound["routed_experts_start_len"] == expected_start_len
