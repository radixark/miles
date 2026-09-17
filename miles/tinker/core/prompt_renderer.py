"""Prompt rendering for recorded sessions: full chat-template render, or TITO prefix inheritance when injected."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from miles.tinker.core.types import UserInputError

if TYPE_CHECKING:
    from miles.tinker.core.tinker_session_server import TrajectorySession, Turn


def _token_list(rendered) -> list[int]:
    """Flatten apply_chat_template(tokenize=True) output (list, BatchEncoding, or batch of one) into ids."""
    if hasattr(rendered, "input_ids"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token) for token in rendered]


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    tokenizer,
) -> list[int]:
    """Validate the messages and render them with apply_chat_template(add_generation_prompt=True, tokenize=True)."""
    if not isinstance(messages, list) or not messages:
        raise UserInputError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise UserInputError(f"messages[{index}] must be an object with a role")
    kwargs = dict(chat_template_kwargs)
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    except (TypeError, ValueError, KeyError) as error:
        raise UserInputError(f"cannot render messages with the chat template: {error}") from error
    ids = _token_list(rendered)
    if not ids:
        raise UserInputError("the chat template rendered an empty prompt")
    return ids


def tito_render_prompt(
    session: TrajectorySession,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tito_tokenizer,
    *,
    max_new_tokens: int,
    budget: int | None,
) -> list[int] | None:
    """TITO: previous turn's ids + tokens of the appended messages (merge_tokens); None = full render, new segment."""
    if session.tito_messages is None or session.tito_token_ids is None:
        return None
    if not isinstance(messages, list) or len(messages) <= len(session.tito_messages):
        return None
    try:
        prompt = tito_tokenizer.merge_tokens(
            old_messages=session.tito_messages,
            new_messages=messages,
            pretokenized_token_ids=session.tito_token_ids,
            tools=tools,
        )
    except ValueError:  # the harness edited, reordered or summarized the history; a fresh render is still exact
        return None
    prompt_ids = [int(token) for token in prompt]
    if budget is not None and len(prompt_ids) + max_new_tokens > budget:
        return None
    return prompt_ids


def on_turn_committed(
    session: TrajectorySession, turn: Turn, messages: list[dict[str, Any]], reply: dict[str, Any]
) -> None:
    """TITO: remember the answered history + reply and this turn's ids as the prefix the next turn inherits."""
    session.tito_messages = [*messages, reply]
    session.tito_token_ids = array("i", [*turn.input_ids, *turn.output_ids])


class PromptRenderer:
    """A session's history as prompt ids: a TITO merge when the history extends the last turn, else a full render."""

    def __init__(self, tokenizer, chat_template_kwargs: dict[str, Any] | None, tito_tokenizer=None) -> None:
        """Keep the HF tokenizer, the gateway's chat_template_kwargs and the optional injected TITOTokenizer."""
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tito_tokenizer = tito_tokenizer

    def render(
        self,
        session: TrajectorySession,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        override: dict[str, Any] | None,
        *,
        max_new_tokens: int,
        budget: int | None,
    ) -> tuple[list[int], bool]:
        """(prompt_ids, inherits): the TITO prefix when it applies, else apply_chat_template over the whole history."""
        template_kwargs = self.template_kwargs(override)
        if self.tito_tokenizer is not None:
            tito_tokenizer = self.tito_tokenizer_for(override)
            prompt_ids = tito_render_prompt(
                session, messages, tools, tito_tokenizer, max_new_tokens=max_new_tokens, budget=budget
            )
            if prompt_ids is not None:
                return prompt_ids, True
        return render_prompt(messages, tools, template_kwargs, self.tokenizer), False

    def committed(self, session: TrajectorySession, turn: Turn, messages: list[dict[str, Any]], text: str) -> None:
        """Remember the answered history for the next TITO merge; a no-op without a TITOTokenizer."""
        if self.tito_tokenizer is not None:
            on_turn_committed(session, turn, messages, {"role": "assistant", "content": text})

    def decode(self, ids) -> str:
        """The reply text for the wire response, special tokens dropped."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def template_kwargs(self, override: dict[str, Any] | None) -> dict[str, Any]:
        """The gateway's chat_template_kwargs, overridden by the turn's chat_template_kwargs object."""
        kwargs = dict(self.chat_template_kwargs)
        if override is not None:
            if not isinstance(override, dict):
                raise UserInputError("chat_template_kwargs must be an object")
            kwargs.update(override)
        return kwargs

    def tito_tokenizer_for(self, override: dict[str, Any] | None):
        """The injected TITOTokenizer, re-scoped with the turn's chat_template_kwargs override when present."""
        if not override:
            return self.tito_tokenizer
        if not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        try:
            return self.tito_tokenizer.clone_with_chat_template_kwargs(override)
        except ValueError as error:
            raise UserInputError(f"chat_template_kwargs conflict with the TITO template: {error}") from error
