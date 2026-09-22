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


def _resends_history(request_messages: Any, stored: list[dict[str, Any]]) -> bool:
    """True when request_messages are a leading slice of the stored history by (role, content): a re-sent request."""
    if not isinstance(request_messages, list) or not request_messages:
        return False
    pairs = zip(request_messages, stored[: len(request_messages)], strict=False)
    return all(a.get("role") == b.get("role") and a.get("content") == b.get("content") for a, b in pairs)


def _try_merge_tokens(
    session: TrajectorySession,
    request_messages: list[dict[str, Any]],
    tito_tokenizer,
    template_args: dict[str, Any],
    *,
    max_new_tokens: int,
    budget: int | None,
) -> tuple[list[int] | None, str | None]:
    """TITO: previous turn's ids + tokens of the appended messages (merge_tokens), or (None, why a full render)."""
    if session.messages is None or session.token_ids is None:
        return None, "first"
    if not isinstance(request_messages, list) or len(request_messages) <= len(session.messages):
        # a re-sent earlier state is a retry (rollback); a shorter history with new content is a rewrite (compaction)
        return None, "retry" if _resends_history(request_messages, session.messages) else "rewrite"
    try:
        prompt = tito_tokenizer.merge_tokens(
            old_messages=session.messages,
            new_messages=request_messages,
            pretokenized_token_ids=session.token_ids,
            template_args=template_args,
        )
    except ValueError:  # the harness edited, reordered or summarized the history; a fresh render is still exact
        return None, "rewrite"
    prompt_token_ids = [int(token) for token in prompt]
    kept = len(session.token_ids) - getattr(tito_tokenizer, "max_trim_tokens", 0)
    if kept > 0 and prompt_token_ids[:kept] != list(session.token_ids[:kept]):
        return None, "mismatch"  # the merge did not extend the recorded prefix: never sample it, re-render instead
    if budget is not None and len(prompt_token_ids) + max_new_tokens > budget:
        return None, "budget"
    return prompt_token_ids, None


class PromptRenderer:
    """A session's history as prompt ids: a TITO merge when the history extends the last turn, else a full render."""

    def __init__(self, tokenizer, chat_template_kwargs: dict[str, Any] | None, tito_tokenizer=None) -> None:
        """Keep the HF tokenizer, the gateway's chat_template_kwargs and the optional injected TITOTokenizer."""
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tito_tokenizer = tito_tokenizer

    def prepare_pretokenized(
        self,
        session: TrajectorySession,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        override: dict[str, Any] | None,
        *,
        max_new_tokens: int,
        budget: int | None,
    ) -> tuple[list[int], bool, str | None, dict[str, Any] | None]:
        """(prompt_token_ids, inherits, reset_reason, request_args): TITO merge when it applies, else a full render."""
        validate_messages(request_messages)  # the merge path calls into the TITO matcher, which assumes dict messages
        if self.tito_tokenizer is None:
            return (
                render_prompt(request_messages, tools, self.template_kwargs(override), self.tokenizer),
                False,
                "no_tito",
                None,
            )
        request_args, continued = self._resolve_request_args(session, tools, override)
        template_args = _template_args(request_args)
        reason = "rewrite"  # the request cannot continue the recorded turn (its tools changed): a new segment
        if continued:
            prompt_token_ids, reason = _try_merge_tokens(
                session,
                request_messages,
                self.tito_tokenizer,
                template_args,
                max_new_tokens=max_new_tokens,
                budget=budget,
            )
            if prompt_token_ids is not None:
                return prompt_token_ids, True, None, request_args
        tito = self.tito_tokenizer
        ids = _rendered_ids(
            lambda: tito.apply_chat_template(
                request_messages, add_generation_prompt=True, tokenize=True, template_args=template_args
            )
        )
        return ids, False, reason, request_args

    def _resolve_request_args(
        self, session: TrajectorySession, tools: list[dict[str, Any]] | None, override: dict[str, Any] | None
    ) -> tuple[dict[str, Any], bool]:
        """The TITO family's resolved (chat_template_kwargs, tools) for this turn; False when it cannot continue."""
        if override is not None and not isinstance(override, dict):
            raise UserInputError("chat_template_kwargs must be an object")
        request = {"chat_template_kwargs": dict(override or {}), "tools": tools}
        if session.request_args is not None:
            try:  # omitted fields inherit the recorded turn's; tools that changed cannot reuse its prefix
                return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=session.request_args), True
            except ValueError:
                return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=None), False
        return self.tito_tokenizer.resolve_request_args(dict(request), turn_args=None), True

    def update_pretokenized_state(
        self,
        session: TrajectorySession,
        turn: Turn,
        request_messages: list[dict[str, Any]],
        assistant_message: dict,
        request_args: dict[str, Any] | None,
    ) -> None:
        """Remember the answered history, the assistant message and the resolved args for the next TITO merge."""
        if self.tito_tokenizer is not None:
            session.messages = [*request_messages, assistant_message]
            session.token_ids = array("i", [*turn.input_ids, *turn.output_ids])
            session.request_args = request_args

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
