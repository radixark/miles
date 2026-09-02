from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from miles.rollout.session.errors import MessageValidationError
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

_MISSING = object()


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
        request_body["logprobs"] = True
        request_body["return_meta_info"] = True
        if self.force_return_routed_experts:
            request_body["return_routed_experts"] = True
        if self.force_return_indexer_topk:
            request_body["return_indexer_topk"] = True
        # Stop-token text is trimmed from assistant content; token IDs still
        # come from the logprobs metadata above.
        request_body["no_stop_trim"] = False
        if self.lora_path is not None:
            request_body["lora_path"] = self.lora_path

        # FIXME(session): Only nested `chat_template_kwargs` reach the local
        # renderer. Top-level `reasoning` / `reasoning_effort` remain passthrough.
        request_kwargs = request_body.get("chat_template_kwargs")
        if request_kwargs is not None and not isinstance(request_kwargs, dict):
            raise MessageValidationError("chat_template_kwargs must be an object")
        request_tito_tokenizer = self.startup_tito_tokenizer
        if request_kwargs:
            try:
                request_tito_tokenizer = request_tito_tokenizer.clone_with_chat_template_kwargs(request_kwargs)
            except ValueError as exc:
                raise MessageValidationError(str(exc)) from exc

        if request_tito_tokenizer.chat_template_kwargs:
            request_body["chat_template_kwargs"] = dict(request_tito_tokenizer.chat_template_kwargs)
        else:
            request_body.pop("chat_template_kwargs", None)

        return PreparedChatRequest(
            body=MappingProxyType(request_body),
            client_stream=client_stream,
            tito_tokenizer=request_tito_tokenizer,
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
        outbound_body["input_ids"] = input_ids
        if routed_experts_start_len is not _MISSING:
            outbound_body["routed_experts_start_len"] = routed_experts_start_len
        return outbound_body
