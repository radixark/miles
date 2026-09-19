"""Resolve chat arguments before rendering or forwarding a request.

``prepare_chat_request`` owns a copy of the client's input. Config rules apply
Session server constraints and retain the client's streaming preference. The
TITO tokenizer then applies model rules to the same full request.

After rendering and a successful generation, the session records the complete
resolved request as ``turn_args``. A continuation supplies that history to the
model resolver, which decides which fields inherit or must stay compatible.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, extract_template_args
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled

DEFAULT_TURN_ARGS_DROP_KEYS = ("input_ids", "messages")


def filter_turn_args(
    turn_args: dict[str, Any], *, drop_keys: tuple[str, ...] = DEFAULT_TURN_ARGS_DROP_KEYS
) -> dict[str, Any]:
    """Copy turn args for metadata, omitting payloads before copying their values."""
    return deepcopy({key: value for key, value in turn_args.items() if key not in drop_keys})


def parse_chat_request(body: bytes) -> dict[str, Any]:
    """Decode the JSON request body, treating an empty body as an empty dict."""
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise MessageValidationError(f"invalid JSON body: {e}") from e


@dataclass
class PreparedChatRequest:
    """Outbound body before adding ``input_ids``, template args to render them,
    and the client's streaming preference for the response."""

    body: dict[str, Any]
    template_args: dict[str, Any]
    client_stream: bool


def prepare_chat_request(
    client_args: dict[str, Any],
    tito_tokenizer: TITOTokenizer,
    *,
    config: SessionServerConfig,
    turn_args: dict[str, Any] | None,
    evaluation: bool = False,
) -> PreparedChatRequest:
    """Resolve an owned request using server rules, model rules, and prior turn args.

    ``turn_args`` is the continued turn's full request; ``None`` starts a root.
    Client input and recorded history remain unchanged.
    """
    request_args, client_stream = resolve_request_args_by_config(deepcopy(client_args), config, evaluation=evaluation)
    try:
        request_args = tito_tokenizer.resolve_request_args(request_args, turn_args=turn_args)
    except ValueError as e:
        raise MessageValidationError(str(e)) from e
    if evaluation:
        # Model rules must not re-enable training replay outputs for evaluation.
        request_args.update(return_sampling_mask=False, return_routed_experts=False, return_indexer_topk=False)
        request_args.pop("routed_experts_start_len", None)
    return PreparedChatRequest(
        body=request_args, template_args=extract_template_args(request_args), client_stream=client_stream
    )


def resolve_request_args_by_config(
    request_args: dict[str, Any], config: SessionServerConfig, *, evaluation: bool = False
) -> tuple[dict[str, Any], bool]:
    """Apply server constraints in place and return the same request and stream intent.

    The caller owns ``request_args``; this function does not copy it.
    """
    # TITO needs these on every request: agent-side overrides would break token accumulation.
    request_args["logprobs"] = True
    request_args["return_meta_info"] = True
    # Must be False so stop-token text is trimmed from assistant content;
    # token IDs still come from logprobs below.
    request_args["no_stop_trim"] = False
    # Training replay follows the launch flags; eval never requests replay outputs.
    request_args["return_routed_experts"] = not evaluation and bool(config.use_rollout_routing_replay)
    request_args["return_indexer_topk"] = not evaluation and bool(config.use_rollout_indexer_replay)

    # The served adapter is selected by training; SGLang lets a ``base:adapter``
    # model parameter beat ``lora_path``, so that spelling is refused too.
    lora_path = LORA_ADAPTER_NAME if lora_rollout_enabled(config) else None
    if (value := request_args.get("lora_path")) is not None and value != lora_path:
        raise MessageValidationError(
            f"lora_path={value!r} is not accepted: the served adapter is selected by training"
        )
    if lora_path is None:
        request_args.pop("lora_path", None)
    else:
        request_args["lora_path"] = lora_path
    if lora_path is not None and ":" in str(request_args.get("model") or ""):
        raise MessageValidationError(
            "model must not name a LoRA adapter; the session server serves the trained adapter"
        )

    # A client setting these has passed out-of-scope information: fail loud.
    if (value := request_args.get("input_ids")) is not None:
        raise MessageValidationError(
            f"input_ids={value!r} is not accepted: TITO token ids are rendered by the session server"
        )
    request_args.pop("input_ids", None)
    if not evaluation and (value := request_args.get("routed_experts_start_len")) is not None:
        raise MessageValidationError(
            f"routed_experts_start_len={value!r} is not accepted: R3 offsets are computed by the session server"
        )
    request_args.pop("routed_experts_start_len", None)
    if (value := request_args.get("logprob_start_len")) is not None:
        raise MessageValidationError(
            f"logprob_start_len={value!r} is not accepted: not supported on the session chat path"
        )
    request_args.pop("logprob_start_len", None)
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
