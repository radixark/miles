from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_contract import (
    MISSING,
    RequestFieldContract,
    RequestValuePolicy,
    SessionRequestContract,
)
from miles.utils.chat_template_utils.request_profile import ModelRequestProfile, RequestIntent
from miles.utils.chat_template_utils.tito_tokenizer import FixedTemplate, TITOTokenizer


def _contract(
    *,
    tito_tokenizer: TITOTokenizer | None = None,
    launch_kwargs: dict | None = None,
    **launch_args,
) -> SessionRequestContract:
    tito_tokenizer = tito_tokenizer or TITOTokenizer(None, chat_template_kwargs=launch_kwargs)
    return SessionRequestContract.from_launch_args(SimpleNamespace(**launch_args), tito_tokenizer)


def _body(**values) -> bytes:
    return json.dumps(values).encode()


def test_request_or_default_distinguishes_missing_from_explicit_none():
    field = RequestFieldContract(
        "field",
        RequestValuePolicy.REQUEST_OR_DEFAULT,
        default="launch",
    )

    assert field.resolve(MISSING) == "launch"
    assert field.resolve(None) is None


def test_miles_owned_fields_override_client_values_and_unknown_fields_pass_through():
    resolved = _contract(
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=True,
        lora_rank=8,
    ).resolve(
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

    assert resolved.outbound_body == {
        "messages": [],
        "custom_backend_field": "keep",
        "logprobs": True,
        "return_meta_info": True,
        "no_stop_trim": False,
        "return_routed_experts": True,
        "return_indexer_topk": True,
        "lora_path": "miles_lora",
    }


def test_client_input_ids_are_rejected_before_derived_ids_are_added():
    with pytest.raises(MessageValidationError, match="input_ids is owned by Miles"):
        _contract().resolve(_body(messages=[], input_ids=None))


def test_request_chat_template_kwargs_override_launch_for_renderer_and_backend():
    resolved = _contract(launch_kwargs={"enable_thinking": False}).resolve(
        _body(messages=[], chat_template_kwargs={"enable_thinking": True})
    )

    assert resolved.render_kwargs == {"enable_thinking": True}
    assert resolved.tito_tokenizer.chat_template_kwargs == resolved.render_kwargs
    assert resolved.outbound_body["chat_template_kwargs"] == resolved.render_kwargs


def test_null_chat_template_kwargs_preserve_launch_defaults():
    resolved = _contract(launch_kwargs={"enable_thinking": False}).resolve(
        _body(messages=[], chat_template_kwargs=None)
    )

    assert resolved.render_kwargs == {"enable_thinking": False}
    assert resolved.outbound_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_non_object_chat_template_kwargs_are_rejected():
    with pytest.raises(MessageValidationError, match="chat_template_kwargs must be an object"):
        _contract().resolve(_body(messages=[], chat_template_kwargs=[]))


class _FixedTokenizer(TITOTokenizer):
    FIXED_TEMPLATE = FixedTemplate(extra_kwargs={"preserve_thinking": True})


def test_per_request_fixed_template_keys_are_rejected_even_when_equal():
    contract = _contract(tito_tokenizer=_FixedTokenizer(None))

    with pytest.raises(MessageValidationError, match="chat_template_kwargs.preserve_thinking is owned by Miles"):
        contract.resolve(_body(messages=[], chat_template_kwargs={"preserve_thinking": True}))


class _AliasTokenizer(TITOTokenizer):
    chat_template_kwarg_aliases = frozenset({"thinking", "enable_thinking"})


def test_existing_chat_template_alias_group_merge_is_preserved():
    tokenizer = _AliasTokenizer(None, chat_template_kwargs={"thinking": False})
    resolved = _contract(tito_tokenizer=tokenizer).resolve(
        _body(messages=[], chat_template_kwargs={"enable_thinking": True})
    )

    assert resolved.render_kwargs == {"enable_thinking": True}


class _TopLevelModeProfile(ModelRequestProfile):
    def extract(self, request_body):
        nested = super().extract(request_body)
        mode = request_body.get("mode", MISSING)
        request_kwargs = dict(nested.chat_template_kwargs)
        if mode is not MISSING:
            request_kwargs.setdefault("mode", mode)
        return RequestIntent(
            chat_template_kwargs=request_kwargs,
            chat_template_kwargs_present=nested.chat_template_kwargs_present or mode is not MISSING,
            consumed_fields=nested.consumed_fields | {"mode"},
        )

    def render_fingerprint(self, render_kwargs):
        return ("mode", render_kwargs.get("mode", "default"))


class _ProfiledTokenizer(TITOTokenizer):
    request_profile = _TopLevelModeProfile()


def test_profile_consumes_alias_and_drives_one_canonical_render_result():
    resolved = _contract(tito_tokenizer=_ProfiledTokenizer(None)).resolve(
        _body(messages=[], mode="low", chat_template_kwargs={"mode": "medium"})
    )

    assert "mode" not in resolved.outbound_body
    assert resolved.render_kwargs == {"mode": "medium"}
    assert resolved.outbound_body["chat_template_kwargs"] == resolved.render_kwargs
    assert resolved.render_fingerprint == ("mode", "medium")


def test_stream_intent_is_client_only():
    resolved = _contract().resolve(_body(messages=[], stream=True, stream_options={"include_usage": True}))

    assert resolved.client_stream is True
    assert "stream" not in resolved.outbound_body
    assert "stream_options" not in resolved.outbound_body
