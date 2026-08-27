from __future__ import annotations

import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Any

from miles.rollout.session.errors import MessageValidationError
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled

MISSING = object()


class RequestValuePolicy(Enum):
    REQUEST_OR_DEFAULT = auto()
    REJECT_IF_PRESENT = auto()
    FORCE_SERVER_VALUE = auto()


@dataclass(frozen=True)
class RequestFieldContract:
    name: str
    policy: RequestValuePolicy
    default: Any = MISSING
    server_value: Any = MISSING

    def resolve(self, request_value: Any = MISSING) -> Any:
        if self.policy is RequestValuePolicy.REQUEST_OR_DEFAULT:
            return request_value if request_value is not MISSING else self.default
        if self.policy is RequestValuePolicy.REJECT_IF_PRESENT:
            if request_value is not MISSING:
                raise MessageValidationError(f"{self.name} is owned by Miles and cannot be provided by the client")
            return self.server_value
        if self.server_value is MISSING:
            raise AssertionError(f"{self.name}: FORCE_SERVER_VALUE requires server_value")
        return self.server_value


@dataclass(frozen=True)
class ResolvedChatRequest:
    outbound_body: dict[str, Any]
    client_stream: bool
    tito_tokenizer: TITOTokenizer
    render_kwargs: Mapping[str, Any]
    render_fingerprint: Hashable | None


@dataclass(frozen=True)
class SessionRequestContract:
    """Resolve client, launch, and Miles-owned values into one backend request."""

    startup_tito_tokenizer: TITOTokenizer
    field_contracts: tuple[RequestFieldContract, ...]

    @classmethod
    def from_launch_args(cls, args, tito_tokenizer: TITOTokenizer) -> SessionRequestContract:
        fields = [
            RequestFieldContract("input_ids", RequestValuePolicy.REJECT_IF_PRESENT),
            RequestFieldContract("logprobs", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=True),
            RequestFieldContract("return_meta_info", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=True),
            RequestFieldContract("no_stop_trim", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=False),
        ]
        if getattr(args, "use_rollout_routing_replay", False):
            fields.append(
                RequestFieldContract("return_routed_experts", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=True)
            )
        if getattr(args, "use_rollout_indexer_replay", False):
            fields.append(
                RequestFieldContract("return_indexer_topk", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=True)
            )
        if is_lora_enabled(args):
            fields.append(
                RequestFieldContract(
                    "lora_path", RequestValuePolicy.FORCE_SERVER_VALUE, server_value=LORA_ADAPTER_NAME
                )
            )
        return cls(startup_tito_tokenizer=tito_tokenizer, field_contracts=tuple(fields))

    def resolve(self, body: bytes) -> ResolvedChatRequest:
        try:
            raw_body = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise MessageValidationError(f"invalid JSON body: {exc}") from exc

        profile = self.startup_tito_tokenizer.request_profile
        try:
            intent = profile.extract(raw_body)
        except ValueError as exc:
            raise MessageValidationError(str(exc)) from exc

        outbound_body = dict(raw_body)
        client_stream = bool(outbound_body.pop("stream", False))
        outbound_body.pop("stream_options", None)
        for field_name in intent.consumed_fields:
            outbound_body.pop(field_name, None)

        for field_contract in self.field_contracts:
            request_value = raw_body.get(field_contract.name, MISSING)
            resolved_value = field_contract.resolve(request_value)
            if resolved_value is not MISSING:
                outbound_body[field_contract.name] = resolved_value

        request_kwargs = RequestFieldContract(
            "chat_template_kwargs",
            RequestValuePolicy.REQUEST_OR_DEFAULT,
            default=MISSING,
        ).resolve(intent.chat_template_kwargs if intent.chat_template_kwargs_present else MISSING)
        if request_kwargs is not MISSING:
            for key in self.startup_tito_tokenizer.FIXED_TEMPLATE.extra_kwargs:
                RequestFieldContract(
                    f"chat_template_kwargs.{key}",
                    RequestValuePolicy.REJECT_IF_PRESENT,
                ).resolve(request_kwargs.get(key, MISSING))

        request_tito_tokenizer = self.startup_tito_tokenizer
        if request_kwargs is not MISSING and request_kwargs:
            try:
                request_tito_tokenizer = request_tito_tokenizer.clone_with_chat_template_kwargs(dict(request_kwargs))
            except ValueError as exc:
                raise MessageValidationError(str(exc)) from exc

        render_kwargs = dict(request_tito_tokenizer.chat_template_kwargs)
        if render_kwargs:
            outbound_body["chat_template_kwargs"] = dict(render_kwargs)
        else:
            outbound_body.pop("chat_template_kwargs", None)

        return ResolvedChatRequest(
            outbound_body=outbound_body,
            client_stream=client_stream,
            tito_tokenizer=request_tito_tokenizer,
            render_kwargs=MappingProxyType(render_kwargs),
            render_fingerprint=profile.render_fingerprint(render_kwargs),
        )


def validate_session_render_fingerprint(
    established: Hashable | None,
    requested: Hashable | None,
) -> None:
    """Reject a renderer identity that cannot reuse the session's token prefix."""
    if established is not None and requested != established:
        raise MessageValidationError(
            "render configuration cannot change within a session: "
            f"established={established!r}, requested={requested!r}"
        )
