"""How one session-server chat request becomes the request sent to SGLang.

The whole decision lives in this module, in the order it runs:

1. ``parse_chat_request``: bytes to the client's dict; the fake-streaming
   flag is popped here and honored when the reply is rendered.
2. ``decide_chat_request_args``: the body, field by field.  Every field named
   there is the server's: ``server_first`` replaces a differing client value
   and logs why, ``server_strict`` refuses it with HTTP 400.  A field named
   nowhere is the client's and is forwarded as sent (sampling parameters,
   ``model``, ``messages``, ``tools``, unknown keys).
3. ``TITOTokenizer.template_args_for_request``: the template args, one dict
   holding every keyword the chat template takes besides the messages,
   ``tools`` included, chosen by the tokenizer family against the
   ``turn_args`` recorded by the turn this request continues (``None`` for a
   new root).  That dict is what the prompt is rendered with, what goes on
   the wire as ``tools`` and ``chat_template_kwargs``, and what the committed
   turn records.

``prepare_chat_request`` runs steps 2 and 3.  The session's
``prepare_token_ids_and_request_args`` (v1 ``LinearTrajectory`` method, v2
``session_state`` function) calls it after rolling back / positioning on the
request's messages and before rendering ``input_ids`` with the template args
it returned, because the args change the token ids.

Why this order: messages are the highest-priority source of truth.  Some
models break the KV cache when certain args change between turns, such as
tools or reasoning effort; that is incorrect behavior.  To prevent it, the
session stores these args on the checkpoint (``turn_args``, the template args
it was rendered with) and the tokenizer family applies a customizable check
against them (``TITOTokenizer.template_args_for_request``) before the prompt
is rendered.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled

logger = logging.getLogger(__name__)


def parse_chat_request(body: bytes) -> tuple[dict[str, Any], bool]:
    """Parse a chat request body into the client's dict and its streaming intent.

    Fake streaming: the backend must stay non-streaming (TITO needs the
    complete message + meta_info, and sglang rejects return_meta_info with
    stream=true), so the client's intent is popped here and honored when
    rendering the client response.
    """
    try:
        client = json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise MessageValidationError(f"invalid JSON body: {e}") from e
    client_stream = bool(client.pop("stream", False))
    client.pop("stream_options", None)
    return client, client_stream


@dataclass
class PreparedChatRequest:
    """``prepare_chat_request`` output: the outbound body before the session adds
    its rendered ``input_ids``, and the template args to render them with (the
    dict the committed turn then records as ``turn_args``)."""

    body: dict[str, Any]
    template_args: dict[str, Any]


def prepare_chat_request(
    client: dict[str, Any],
    tito_tokenizer: TITOTokenizer,
    *,
    config: SessionServerConfig,
    turn_args: dict[str, Any] | None,
) -> PreparedChatRequest:
    """Resolve the outbound request and template arguments for v1 and v2 sessions.

    `turn_args` contains the continued turn's recorded template arguments;
    `None` starts a new root. Raise `MessageValidationError` (HTTP 400) on rejection.
    """
    wire = decide_chat_request_args(client, config)
    try:
        template_args = tito_tokenizer.template_args_for_request(client, turn_args=turn_args)
    except ValueError as e:
        raise MessageValidationError(str(e)) from e
    # Send SGLang the same tools and template kwargs used for local rendering.
    # Its HTTP API keeps `tools` separate from `chat_template_kwargs`.
    tools = template_args.get("tools")
    kwargs = {key: value for key, value in template_args.items() if key != "tools"}
    if tools:
        wire["tools"] = tools
    else:
        wire.pop("tools", None)
    if kwargs:
        wire["chat_template_kwargs"] = kwargs
    else:
        wire.pop("chat_template_kwargs", None)
    return PreparedChatRequest(body=wire, template_args=template_args)


def decide_chat_request_args(client: dict[str, Any], config: SessionServerConfig) -> dict[str, Any]:
    """The outbound ``/v1/chat/completions`` body: the client's fields as sent,
    then every field the session server decides.  Client key order is kept;
    fields the client did not send follow."""
    wire = dict(client)

    # TITO needs these on every request: agent-side overrides would break token accumulation.
    server_first(wire, "logprobs", True, why="TITO reads meta_info.output_token_logprobs")
    server_first(wire, "return_meta_info", True, why="wraps output_token_logprobs in choice.meta_info")
    server_first(wire, "no_stop_trim", False, why="stop-token text is trimmed; token ids come from logprobs")

    # R3 replay follows the launch flags, on or off.
    server_first(
        wire,
        "return_routed_experts",
        bool(config.use_rollout_routing_replay),
        why="follows --use-rollout-routing-replay",
    )
    server_first(
        wire,
        "return_indexer_topk",
        bool(config.use_rollout_indexer_replay),
        why="follows --use-rollout-indexer-replay",
    )

    # The served adapter is selected by training; SGLang lets a ``base:adapter``
    # model parameter beat ``lora_path``, so that spelling is refused too.
    lora_path = LORA_ADAPTER_NAME if lora_rollout_enabled(config) else None
    server_strict(wire, "lora_path", lora_path, why="the served adapter is selected by training")
    if lora_path is not None and ":" in str(wire.get("model") or ""):
        raise MessageValidationError(
            "model must not name a LoRA adapter; the session server serves the trained adapter"
        )

    # A client setting these has passed out-of-scope information: fail loud.
    server_strict(wire, "input_ids", None, why="TITO token ids are rendered by the session server")
    server_strict(wire, "routed_experts_start_len", None, why="R3 offsets are computed by the session server")
    server_strict(wire, "logprob_start_len", None, why="not supported on the session chat path")
    return wire


def server_first(wire: dict[str, Any], name: str, value: Any, *, why: str) -> None:
    """Set ``wire[name]`` to the server's ``value``; a differing client value is
    replaced and logged with ``why``.  ``value=None`` takes the field off the wire.
    A client value of ``None`` counts as not sent."""
    sent = wire.get(name)
    if sent is not None and sent != value:
        logger.warning("%s=%r from the client replaced by %r: %s", name, sent, value, why)
    if value is None:
        wire.pop(name, None)
    else:
        wire[name] = value


def server_strict(wire: dict[str, Any], name: str, value: Any, *, why: str) -> None:
    """Set ``wire[name]`` to the server's ``value``; a differing client value is
    rejected with HTTP 400 quoting ``why``.  ``value=None`` means the client may
    not send the field at all.  A client value of ``None`` counts as not sent."""
    sent = wire.get(name)
    if sent is not None and sent != value:
        raise MessageValidationError(f"{name}={sent!r} is not accepted: {why}")
    if value is None:
        wire.pop(name, None)
    else:
        wire[name] = value
