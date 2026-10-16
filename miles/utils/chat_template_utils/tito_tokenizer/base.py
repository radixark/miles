"""Shared TITO tokenizer contract: ``FixedTemplate`` and the default ``TITOTokenizer``."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miles.utils.chat_template_utils import template
from miles.utils.chat_template_utils.message_matcher_hub import assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

# Bundled fixed-template files live under this directory; ``FixedTemplate.template``
# values are filenames relative to it.
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

# Roles that a fixed template may support after the pretokenized assistant
# prefix.  A family narrows this set only when its registered renderer cannot
# preserve every role append-only.
VALID_APPEND_ROLES: tuple[str, ...] = ("tool", "user", "system", "assistant")
ALL_APPEND_ROLES: frozenset[str] = frozenset(VALID_APPEND_ROLES)

_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


@dataclass(frozen=True)
class FixedTemplate:
    """A family's fixed chat template, required kwargs, and append surface.

    ``template`` is a path relative to ``TEMPLATE_DIR`` for a bundled fixed
    template, or ``None`` to keep the HF-native template (kwargs-only fix).
    ``extra_kwargs`` carry the family's preserve-think constants, so renders
    stay append-only.  ``allowed_append_roles`` defaults to the maximal
    four-role surface; a known restricted template must narrow it explicitly.
    ``consistant_kwargs`` lists fields that must retain their recorded value
    or absence when continuing a turn.
    """

    template: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    allowed_append_roles: frozenset[str] = ALL_APPEND_ROLES
    consistant_kwargs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        roles = frozenset(self.allowed_append_roles)
        invalid = roles - ALL_APPEND_ROLES
        if invalid:
            raise ValueError(
                f"Unknown FixedTemplate allowed_append_roles: {sorted(invalid)}; "
                f"supported roles are {sorted(ALL_APPEND_ROLES)}"
            )
        object.__setattr__(self, "allowed_append_roles", roles)


def extract_template_args(request_args: dict[str, Any]) -> dict[str, Any]:
    """Select renderer kwargs from an already resolved request, including tools."""
    args = dict(request_args.get("chat_template_kwargs") or {})
    if request_args.get("tools"):
        args["tools"] = request_args["tools"]
    return args


def _build_dummy_assistant(stored_assistant: dict[str, Any]) -> dict[str, Any]:
    """Build a dummy assistant that preserves the stored turn's tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": stored_assistant.get("tool_calls") or [],
    }


class TITOTokenizer:
    """Incremental tokenization and prefix merging for appended messages."""

    max_trim_tokens: int = 0
    trailing_token_ids: frozenset[int] = frozenset()

    # The family's fixed renderer contract. DEFAULT uses the model's native
    # template with the maximal best-effort append surface.
    FIXED_TEMPLATE: FixedTemplate = FixedTemplate()

    # sglang ``--reasoning-parser`` and ``--tool-call-parser`` values bound to
    # this family.
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        special_token_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._assistant_start_str = assistant_start_str
        self.allowed_append_roles = self.FIXED_TEMPLATE.allowed_append_roles
        self.special_token_ids: set[int] = special_token_ids
        launch_args = TITOTokenizer.resolve_request_args(self, {}, turn_args=None)
        self.chat_template_kwargs = extract_template_args(launch_args)

    def request_arg_rules(self) -> list[Callable[..., None]]:
        """Return this model's rules in execution order; subclasses may extend the list."""
        return [
            self.resolve_template_kwargs,
            self.resolve_tools,
            self.resolve_history_args,
            self.apply_fixed_template_kwargs,
        ]

    def resolve_request_args(
        self, request_args: dict[str, Any], *, turn_args: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply model rules in place and return the same full request.

        Rules read request_source, turn_args and launch kwargs without modifying
        them. Copy inherited mutable values before writing them into request_args.
        """
        request_source = dict(request_args)
        for rule in self.request_arg_rules():
            rule(request_args, request_source=request_source, turn_args=turn_args)
        return request_args

    def resolve_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        """Merge request template fields over history and launch defaults."""
        kwargs = {
            **self.chat_template_kwargs,
            **((turn_args or {}).get("chat_template_kwargs") or {}),
            **(request_source.get("chat_template_kwargs") or {}),
        }
        # Launch kwargs may contain tools; requests carry them at the top level.
        kwargs.pop("tools", None)
        request_args["chat_template_kwargs"] = deepcopy(kwargs)

    def resolve_tools(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        """Inherit omitted tools and reject changes to tools in a reused prefix."""
        tools = request_source.get("tools") or None
        if turn_args is None:
            tools = tools or self.chat_template_kwargs.get("tools")
        else:
            recorded = turn_args.get("tools")
            if tools is None:
                tools = recorded
            elif template.extract_tool_dicts(tools) != template.extract_tool_dicts(recorded):
                raise ValueError(
                    "tools changed on a continued turn: the turn being continued was rendered with different tools, "
                    "and this model family renders tools in the prompt prefix"
                )
        request_args["tools"] = deepcopy(tools)

    def apply_fixed_template_kwargs(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        """Override requested template values with the family's required settings."""
        if self.FIXED_TEMPLATE.extra_kwargs:
            request_args.setdefault("chat_template_kwargs", {}).update(deepcopy(self.FIXED_TEMPLATE.extra_kwargs))

    def resolve_history_args(
        self,
        request_args: dict[str, Any],
        *,
        request_source: dict[str, Any],
        turn_args: dict[str, Any] | None,
    ) -> None:
        """Keep selected historical fields, including their absence, in a continued turn."""
        if turn_args is None:
            return
        recorded = turn_args.get("chat_template_kwargs") or {}
        kwargs = request_args.setdefault("chat_template_kwargs", {})
        for key in self.FIXED_TEMPLATE.consistant_kwargs:
            kwargs.pop(key, None)
        kwargs.update(
            deepcopy({key: recorded[key] for key in self.FIXED_TEMPLATE.consistant_kwargs if key in recorded})
        )

    def create_comparator(self) -> TokenSeqComparator:
        """Create a :class:`TokenSeqComparator` configured with this
        tokenizer's model-specific settings."""
        return TokenSeqComparator(
            self.tokenizer,
            assistant_start_str=self._assistant_start_str,
            special_token_ids=self.special_token_ids,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def default_template_args(self, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return launch template kwargs with optional tools for direct rendering."""
        args = dict(self.chat_template_kwargs)
        if tools:
            args["tools"] = tools
        return args

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        template_args: dict[str, Any] | None = None,
    ) -> str | list[int]:
        """Render messages using resolved template kwargs and tools.

        Pass `template_args` as the complete argument set; it is not merged with
        launch defaults. `None` uses this tokenizer's launch defaults.
        """
        # TODO: Use the unified kwargs resolver for launch and request arguments once
        # available, then check whether callers still need this default fallback.
        args = self.chat_template_kwargs if template_args is None else template_args
        return template.apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **args,
        )

    def postprocess_completion(
        self,
        *,
        choice: dict[str, Any],
        assistant_message: dict[str, Any],
        completion_token_ids: list[int],
    ) -> dict[str, Any]:
        """Postprocess an upstream completion and return the message to store.

        The default path trusts SGLang's parsed message. Model families that
        need token-aware response handling override this hook and may update
        ``choice`` before returning their server-side message representation.
        """
        return assistant_message

    def preserve_server_message_state(
        self,
        stored_messages: list[dict[str, Any]],
        request_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Restore model-specific server-owned sidecars after client replay."""
        return list(request_messages)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render *base_messages* and *base_messages + appended_messages*, return
        tokens for the suffix.

        When *add_generation_prompt* is True and *appended_messages* is empty,
        this computes the generation-prompt suffix (the assistant opener tokens).
        """
        text_without = self.apply_chat_template(
            base_messages, add_generation_prompt=False, template_args=template_args
        )
        text_with = self.apply_chat_template(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            template_args=template_args,
        )
        if not text_with.startswith(text_without):
            roles = [msg["role"] for msg in appended_messages] if appended_messages else ["generation_prompt"]
            raise ValueError(f"rendered suffix diff failed for {roles}")
        return self._encode_text(text_with[len(text_without) :])

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for messages appended after the
        pretokenized prefix.

        Appended roles must be listed in ``self.allowed_append_roles``.  The
        method validates that *new_messages* is an append-only extension of
        *old_messages* via ``assert_messages_append_only_with_allowed_role``.

        Args:
            old_messages: Previously stored messages (prefix).
            new_messages: Full new message list (must be a superset of
                *old_messages* with only allowed-role messages appended).
            template_args: Resolved template kwargs and tools, or `None` for launch defaults.

        Returns:
            Incremental token IDs (including the generation prompt) that,
            when merged with pretokenized prefix via ``merge_tokens``,
            form the full prompt token IDs.
        """
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        appended_messages = new_messages[len(old_messages) :]
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM, _build_dummy_assistant(old_messages[-1])],
            appended_messages,
            template_args=template_args,
            add_generation_prompt=True,
        )

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        """Merge *pretokenized_token_ids* with incremental tokens to produce
        the complete prompt token IDs (including generation prompt).

        The default implementation is simple concatenation.  Subclasses
        override this to handle model-specific boundary token logic.
        """
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        return list(pretokenized_token_ids) + incremental
