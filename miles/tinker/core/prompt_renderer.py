"""Prompt rendering for recorded sessions: full chat-template render, or TITO prefix inheritance when injected."""

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
    """One turn's prompt: its ids, whether they inherit the parent's, why not, the resolved args, the parent turn."""

    prompt_token_ids: list[int]
    inherits: bool
    reset_reason: str | None
    request_args: dict[str, Any] | None
    parent: int | None


def _token_list(rendered) -> list[int]:
    """Flatten apply_chat_template(tokenize=True) output (list, BatchEncoding, or batch of one) into ids."""
    if hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def validate_messages(request_messages: Any) -> None:
    """A non-empty list of objects with a role, else UserInputError (400); runs before any render or TITO merge."""
    if not isinstance(request_messages, list) or not request_messages:
        raise UserInputError("messages must be a non-empty list")
    for index, message in enumerate(request_messages):
        if not isinstance(message, dict) or "role" not in message:
            raise UserInputError(f"messages[{index}] must be an object with a role")


def _rendered_ids(render: Any) -> list[int]:
    """Run a chat-template render, mapping template errors to UserInputError and refusing an empty prompt."""
    try:
        rendered = render()
    except (TypeError, ValueError, KeyError) as error:
        raise UserInputError(f"cannot render messages with the chat template: {error}") from error
    ids = _token_list(rendered)
    if not ids:
        raise UserInputError("the chat template rendered an empty prompt")
    return ids


def render_prompt(
    request_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int]:
    """Validate the messages and render them with apply_chat_template(add_generation_prompt=True, tokenize=True)."""
    validate_messages(request_messages)
    kwargs = dict(chat_template_kwargs)
    if tools:
        kwargs["tools"] = tools
    return _rendered_ids(
        lambda: tokenizer.apply_chat_template(request_messages, add_generation_prompt=True, tokenize=True, **kwargs)
    )


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


def attach_point(
    turns: list[Turn], request_messages: list[dict[str, Any]], matcher: MessageMatcher
) -> tuple[int | None, str | None]:
    """The turn this request continues (longest history prefixing it, latest on ties), or (None, why not)."""
    best: int | None = None
    for index, turn in enumerate(turns):
        if turn.messages is None or not _is_prefix(turn.messages, request_messages, matcher):
            continue
        if best is None or len(turn.messages) >= len(turns[best].messages):
            best = index
    if best is not None:
        return best, None
    if not turns:
        return None, "first"
    for turn in turns:  # the request repeats a recorded turn's own request: a retry, not an edited history
        history = turn.messages or []
        if len(history) - 1 == len(request_messages) and all(
            matcher(stored, new) for stored, new in zip(history, request_messages, strict=False)
        ):
            return None, "retry"
    return None, "rewrite"


def _try_merge_tokens(
    parent: Turn,
    request_messages: list[dict[str, Any]],
    tito_tokenizer,
    template_args: dict[str, Any],
    *,
    max_new_tokens: int,
    budget: int | None,
) -> tuple[list[int] | None, str | None]:
    """TITO: the parent's ids + tokens of the appended messages (merge_tokens), or (None, why a full render)."""
    prefix_ids = [*parent.input_ids, *parent.output_ids]
    try:
        prompt = tito_tokenizer.merge_tokens(
            old_messages=parent.messages,
            new_messages=request_messages,
            pretokenized_token_ids=prefix_ids,
            template_args=template_args,
        )
    except ValueError:  # the appended messages cannot extend this family's template (e.g. a disallowed role)
        return None, "rewrite"
    prompt_token_ids = [int(token) for token in prompt]
    kept = len(prefix_ids) - getattr(tito_tokenizer, "max_trim_tokens", 0)
    if kept > 0 and prompt_token_ids[:kept] != prefix_ids[:kept]:
        return None, "mismatch"  # the merge did not extend the recorded prefix: never sample it, re-render instead
    if budget is not None and len(prompt_token_ids) + max_new_tokens > budget:
        return None, "budget"
    return prompt_token_ids, None


class PromptRenderer:
    """A session's history as prompt ids: a TITO merge from the turn it continues, else a full render."""

    def __init__(
        self,
        tokenizer,
        chat_template_kwargs: dict[str, Any] | None,
        tito_tokenizer=None,
        message_matcher: MessageMatcher | None = None,
    ) -> None:
        """Keep the HF tokenizer, the gateway's chat_template_kwargs, the optional TITOTokenizer and matcher."""
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tito_tokenizer = tito_tokenizer
        self.message_matcher = message_matcher or _same_role_and_content
        self._render_arguments = _named_parameters(getattr(tokenizer, "apply_chat_template", None))

    def prepare_pretokenized(
        self,
        session: TrajectorySession,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        override: dict[str, Any] | None,
        *,
        max_new_tokens: int,
        budget: int | None,
    ) -> Rendered:
        """Find the turn the request continues, then a TITO merge from it when it applies, else a full render."""
        validate_messages(request_messages)
        self._check_override(override)
        parent, reason = attach_point(session.turns, request_messages, self.message_matcher)
        if self.tito_tokenizer is None:
            ids = render_prompt(request_messages, tools, self.template_kwargs(override), self.tokenizer)
            return Rendered(ids, False, "no_tito", None, parent)
        parent_turn = session.turns[parent] if parent is not None else None
        request_args, continued = self._resolve_request_args(parent_turn, tools, override)
        template_args = _template_args(request_args)
        if parent_turn is not None:
            reason = "rewrite"  # the parent's prefix cannot be reused: its tools changed, or the merge refused
            if continued:
                ids, reason = _try_merge_tokens(
                    parent_turn,
                    request_messages,
                    self.tito_tokenizer,
                    template_args,
                    max_new_tokens=max_new_tokens,
                    budget=budget,
                )
                if ids is not None:
                    return Rendered(ids, True, None, request_args, parent)
        tito = self.tito_tokenizer
        ids = _rendered_ids(
            lambda: tito.apply_chat_template(
                request_messages, add_generation_prompt=True, tokenize=True, template_args=template_args
            )
        )
        return Rendered(ids, False, reason, request_args, parent)

    def _check_override(self, override: dict[str, Any] | None) -> None:
        """A turn's chat_template_kwargs are template variables; apply_chat_template's own arguments are refused."""
        if override is None:
            return
        if not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        if refused := sorted(self._render_arguments.intersection(override)):
            raise UserInputError(f"chat_template_kwargs cannot set {refused}: apply_chat_template's own arguments")

    def _resolve_request_args(
        self, parent: Turn | None, tools: list[dict[str, Any]] | None, override: dict[str, Any] | None
    ) -> tuple[dict[str, Any], bool]:
        """The TITO family's resolved (chat_template_kwargs, tools) for this turn; False when it cannot continue."""
        if override is not None and not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        request = {"chat_template_kwargs": dict(override or {}), "tools": tools}
        if parent is not None and parent.request_args is not None:
            try:  # omitted fields inherit the parent turn's; tools that changed cannot reuse its prefix
                return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=parent.request_args), True
            except ValueError:
                return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=None), False
        return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=None), True

    @property
    def max_trim_tokens(self) -> int:
        """Trailing tokens the TITO family may drop when it extends a prefix (GLM: 1); 0 without TITO."""
        return getattr(self.tito_tokenizer, "max_trim_tokens", 0)

    def decode(self, ids) -> str:
        """The reply text for the wire response, special tokens dropped."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def assistant_message(self, turn: Turn) -> dict[str, Any]:
        """The unified assistant message for a reply: text today; per-family tool_call parsing would plug in here."""
        return {"role": "assistant", "content": self.decode(turn.output_ids)}

    def template_kwargs(self, override: dict[str, Any] | None) -> dict[str, Any]:
        """The gateway's chat_template_kwargs, overridden by the turn's chat_template_kwargs object (no TITO)."""
        kwargs = dict(self.chat_template_kwargs)
        if override is not None:
            if not isinstance(override, dict):
                raise UserInputError("chat_template_kwargs must be an object")
            kwargs.update(override)
        return kwargs
