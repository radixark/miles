"""Apply session server constraints to outgoing chat requests."""

import json
import logging
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled

logger = logging.getLogger(__name__)


def parse_chat_request(body: bytes) -> dict[str, Any]:
    """Decode the JSON request body, treating an empty body as an empty dict."""
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise MessageValidationError(f"invalid JSON body: {e}") from e


def resolve_request_args_by_config(
    request_args: dict[str, Any], config: SessionServerConfig
) -> tuple[dict[str, Any], bool]:
    """Apply server constraints in place and return the same request and stream intent.

    The caller owns ``request_args``; this function does not copy it.
    """
    # TITO needs these on every request: agent-side overrides would break token accumulation.
    server_first(request_args, "logprobs", True, why="TITO reads meta_info.output_token_logprobs")
    server_first(request_args, "return_meta_info", True, why="wraps output_token_logprobs in choice.meta_info")
    server_first(request_args, "no_stop_trim", False, why="stop-token text is trimmed; token ids come from logprobs")

    # R3 replay follows the launch flags, on or off.
    server_first(
        request_args,
        "return_routed_experts",
        bool(config.use_rollout_routing_replay),
        why="follows --use-rollout-routing-replay",
    )
    server_first(
        request_args,
        "return_indexer_topk",
        bool(config.use_rollout_indexer_replay),
        why="follows --use-rollout-indexer-replay",
    )

    # The served adapter is selected by training; SGLang lets a ``base:adapter``
    # model parameter beat ``lora_path``, so that spelling is refused too.
    lora_path = LORA_ADAPTER_NAME if lora_rollout_enabled(config) else None
    server_strict(request_args, "lora_path", lora_path, why="the served adapter is selected by training")
    if lora_path is not None and ":" in str(request_args.get("model") or ""):
        raise MessageValidationError(
            "model must not name a LoRA adapter; the session server serves the trained adapter"
        )

    # A client setting these has passed out-of-scope information: fail loud.
    server_strict(request_args, "input_ids", None, why="TITO token ids are rendered by the session server")
    server_strict(request_args, "routed_experts_start_len", None, why="R3 offsets are computed by the session server")
    server_strict(request_args, "logprob_start_len", None, why="not supported on the session chat path")
    # TITO needs a complete backend response with meta_info. Preserve the client's
    # streaming intent for the response, but keep the backend request non-streaming.
    client_stream = bool(request_args.pop("stream", False))
    request_args.pop("stream_options", None)
    kwargs = request_args.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise MessageValidationError("chat_template_kwargs must be an object")
    if kwargs is not None and "tools" in kwargs:
        raise MessageValidationError("tools belongs at the top level of the request, not in chat_template_kwargs")
    return request_args, client_stream


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
