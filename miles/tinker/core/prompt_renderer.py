"""Prompt rendering for recorded sessions: a full render of the history through an injected miles TITOTokenizer."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.tinker.core.types import UserInputError

if TYPE_CHECKING:
    from miles.tinker.core.tinker_session_server import TrajectorySession, Turn

MessageMatcher = Callable[[dict[str, Any], dict[str, Any]], bool]  # (stored message, request message) -> same?


def _same_role_and_content(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """The fallback matcher, role and content only; serve_tinker injects the miles strict matcher instead."""
    return stored.get("role") == new.get("role") and stored.get("content") == new.get("content")


def _named_parameters(function) -> frozenset[str]:
    """The keyword-passable parameter names of a callable (none when absent): apply_chat_template's own arguments."""
    if function is None:
        return frozenset()
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return frozenset(name for name, param in inspect.signature(function).parameters.items() if param.kind in kinds)


@dataclass(frozen=True)
class Rendered:
    """One turn's prompt: its ids and the turn it continues (None: a new root)."""

    prompt_token_ids: list[int]
    parent: int | None


def _token_list(rendered) -> list[int]:
    """Flatten apply_chat_template(tokenize=True) output (list, BatchEncoding, or batch of one) into ids."""
    if hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def _validate_messages(request_messages: Any) -> None:
    """A non-empty list of objects with a role, else UserInputError (400); runs before any render."""
    if not isinstance(request_messages, list) or not request_messages:
        raise UserInputError("messages must be a non-empty list")
    for index, message in enumerate(request_messages):
        if not isinstance(message, dict) or "role" not in message:
            raise UserInputError(f"messages[{index}] must be an object with a role")
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not (isinstance(tool_calls, list) and all(type(c) is dict for c in tool_calls)):
            raise UserInputError(f"messages[{index}].tool_calls must be a list of objects")


def _rendered_ids(render: Any) -> list[int]:
    """Run a chat-template render, mapping template errors to UserInputError and refusing an empty prompt."""
    try:
        rendered = render()
    except Exception as error:  # the template is fixed: a failed render, jinja's too, is the request's fault
        raise UserInputError(f"cannot render messages with the chat template: {error}") from error
    ids = _token_list(rendered)
    if not ids:
        raise UserInputError("the chat template rendered an empty prompt")
    return ids


def _template_args(request_args: dict[str, Any]) -> dict[str, Any]:
    """Renderer kwargs of a resolved request: its chat_template_kwargs plus tools (as chat_template_utils extracts)."""
    args = dict(request_args.get("chat_template_kwargs") or {})
    if request_args.get("tools"):
        args["tools"] = request_args["tools"]
    return args


def _is_prefix(history: list[dict[str, Any]], request_messages: list[dict[str, Any]], matcher: MessageMatcher) -> bool:
    """True when the history matches the leading messages of a longer request, message by message."""
    return len(history) < len(request_messages) and all(
        matcher(stored, new) for stored, new in zip(history, request_messages, strict=False)
    )


def _attach_point(turns: list[Turn], request_messages: list[dict[str, Any]], matcher: MessageMatcher) -> int | None:
    """The turn this request continues: the longest recorded history prefixing it (latest on ties), else None."""
    best: int | None = None
    for index, turn in enumerate(turns):
        if turn.messages is None or not _is_prefix(turn.messages, request_messages, matcher):
            continue
        if best is None or len(turn.messages) >= len(turns[best].messages):
            best = index
    return best


class PromptRenderer:
    """A session's history as prompt ids: the turn a request continues, and a full render of its messages."""

    def __init__(self, tokenizer, tito_tokenizer, *, message_matcher: MessageMatcher | None = None) -> None:
        """Keep the HF tokenizer (decode), the TITOTokenizer that renders, and the message matcher."""
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.message_matcher = message_matcher or _same_role_and_content
        self._render_arguments = _named_parameters(getattr(tokenizer, "apply_chat_template", None))

    def prepare_pretokenized(
        self,
        session: TrajectorySession,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        override: dict[str, Any] | None,
    ) -> Rendered:
        """Find the turn the request continues (the tree edge), then render the whole history."""
        _validate_messages(request_messages)
        self._check_override(override)
        parent = _attach_point(session.turns, request_messages, self.message_matcher)
        request = {"chat_template_kwargs": dict(override or {}), "tools": tools}
        template_args = _template_args(self.tito_tokenizer.resolve_request_args(request, turn_args=None))
        tito = self.tito_tokenizer
        ids = _rendered_ids(
            lambda: tito.apply_chat_template(
                request_messages, add_generation_prompt=True, tokenize=True, template_args=template_args
            )
        )
        return Rendered(ids, parent=parent)

    def _check_override(self, override: dict[str, Any] | None) -> None:
        """A turn's chat_template_kwargs are template variables; apply_chat_template's own arguments are refused."""
        if override is None:
            return
        if not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        if refused := sorted(self._render_arguments.intersection(override)):
            raise UserInputError(f"chat_template_kwargs cannot set {refused}: apply_chat_template's own arguments")

    def decode(self, ids) -> str:
        """The reply text for the wire response, special tokens dropped."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def assistant_message(self, turn: Turn, stop: list[str] | None = None) -> dict[str, Any]:
        """The unified assistant message, without a stop string the reply ended on (as OpenAI)."""
        content = self.decode(turn.output_ids)
        if turn.finish_reason == "stop":
            for suffix in stop or ():
                if suffix and content.endswith(suffix):
                    return {"role": "assistant", "content": content[: -len(suffix)]}
        return {"role": "assistant", "content": content}
