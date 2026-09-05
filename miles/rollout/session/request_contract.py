from __future__ import annotations

import json
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from miles.rollout.session.errors import MessageValidationError
from miles.utils.arg_resolution import (
    MISSING,
    ArgBatch,
    ArgResolutionContract,
    ArgResolutionError,
    Binding,
    PrimaryField,
    PrimarySchema,
    SourceSpec,
    UnknownInputPolicy,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

_MISSING = object()
_REQUEST_FIELDS = (
    "logprobs",
    "return_meta_info",
    "return_routed_experts",
    "return_indexer_topk",
    "no_stop_trim",
    "lora_path",
    "input_ids",
    "routed_experts_start_len",
)
_REQUEST_RESOLVER = ArgResolutionContract(
    PrimarySchema(tuple(PrimaryField(name) for name in _REQUEST_FIELDS)),
    (
        SourceSpec(
            "request",
            0,
            tuple(Binding(name, name) for name in _REQUEST_FIELDS),
            unknown_inputs=UnknownInputPolicy.IGNORE,
        ),
        SourceSpec("configured", 10, tuple(Binding(name, name) for name in _REQUEST_FIELDS[:6])),
        SourceSpec("session", 20, tuple(Binding(name, name) for name in _REQUEST_FIELDS[6:])),
    ),
)


def _resolve_request_fields(body: Mapping[str, Any], overrides: ArgBatch) -> Mapping[str, Any]:
    try:
        resolved = _REQUEST_RESOLVER.resolve((ArgBatch("request", body), overrides))
    except ArgResolutionError as exc:
        raise MessageValidationError(str(exc)) from exc
    return {name: resolved.values[name] for name in _REQUEST_FIELDS if name in resolved.values}


@dataclass(frozen=True)
class PreparedChatRequest:
    body: Mapping[str, Any]
    client_stream: bool
    tito_tokenizer: TITOTokenizer


@dataclass(frozen=True)
class SessionRequestContract:
    """Resolve client, launch, and Miles-owned values into one backend request."""

    startup_tito_tokenizer: TITOTokenizer
    force_return_routed_experts: bool
    force_return_indexer_topk: bool
    lora_path: str | None

    @classmethod
    def from_settings(
        cls,
        tito_tokenizer: TITOTokenizer,
        *,
        force_return_routed_experts: bool,
        force_return_indexer_topk: bool,
        lora_path: str | None,
    ) -> SessionRequestContract:
        return cls(
            startup_tito_tokenizer=tito_tokenizer,
            force_return_routed_experts=force_return_routed_experts,
            force_return_indexer_topk=force_return_indexer_topk,
            lora_path=lora_path,
        )

    def prepare(self, body: bytes) -> PreparedChatRequest:
        """Resolve request and launch values needed before local TITO rendering."""
        try:
            request_body = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise MessageValidationError(f"invalid JSON body: {exc}") from exc

        # The backend must stay non-streaming because TITO consumes the complete
        # response and meta_info. Client streaming is synthesized afterward.
        # Non-object JSON intentionally keeps the existing native failure here.
        client_stream = bool(request_body.pop("stream", False))
        request_body.pop("stream_options", None)

        # TITO needs Miles-owned prompt IDs plus SGLang's token metadata. These
        # values are forced so a client override cannot break token tracking.
        request_body.update(
            _resolve_request_fields(
                request_body,
                ArgBatch(
                    "configured",
                    {
                        "logprobs": True,
                        "return_meta_info": True,
                        "no_stop_trim": False,
                        "return_routed_experts": True if self.force_return_routed_experts else MISSING,
                        "return_indexer_topk": True if self.force_return_indexer_topk else MISSING,
                        "lora_path": self.lora_path if self.lora_path is not None else MISSING,
                    },
                ),
            )
        )

        request_kwargs = request_body.get("chat_template_kwargs")
        if request_kwargs is not None and not isinstance(request_kwargs, dict):
            raise MessageValidationError("chat_template_kwargs must be an object")
        request_tito_tokenizer = self.startup_tito_tokenizer
        reasoning_config = request_tito_tokenizer.reasoning_template_config
        if reasoning_config is not None:
            try:
                request_kwargs = reasoning_config.resolve(request_tito_tokenizer.chat_template_kwargs, request_body)
            except ArgResolutionError as exc:
                raise MessageValidationError(str(exc)) from exc
        if request_kwargs:
            try:
                request_tito_tokenizer = request_tito_tokenizer.clone_with_chat_template_kwargs(request_kwargs)
            except ValueError as exc:
                raise MessageValidationError(str(exc)) from exc

        renderer = copy(self.startup_tito_tokenizer)
        renderer.chat_template_kwargs = deepcopy(request_tito_tokenizer.chat_template_kwargs)
        if renderer.chat_template_kwargs:
            request_body["chat_template_kwargs"] = deepcopy(renderer.chat_template_kwargs)
        else:
            request_body.pop("chat_template_kwargs", None)

        return PreparedChatRequest(
            body=MappingProxyType(request_body),
            client_stream=client_stream,
            tito_tokenizer=renderer,
        )

    def finalize(
        self,
        prepared: PreparedChatRequest,
        *,
        input_ids: list[int],
        routed_experts_start_len: Any = _MISSING,
    ) -> dict[str, Any]:
        """Apply session-derived values and return the body sent to SGLang."""
        outbound_body = dict(prepared.body)
        outbound_body.update(
            _resolve_request_fields(
                prepared.body,
                ArgBatch(
                    "session",
                    {
                        "input_ids": input_ids,
                        "routed_experts_start_len": (
                            MISSING if routed_experts_start_len is _MISSING else routed_experts_start_len
                        ),
                    },
                ),
            )
        )
        if "chat_template_kwargs" in outbound_body:
            outbound_body["chat_template_kwargs"] = deepcopy(outbound_body["chat_template_kwargs"])
        return outbound_body
