"""TITO tokenizer — incremental tokenization for pretokenized prefix reuse.

``TITOTokenizer`` computes incremental token IDs for non-assistant messages
(tool responses, system injections) that follow the assistant's generated
token sequence, then merges them with the pretokenized prefix — handling
model-specific boundary tokens at the junction.

The default implementation uses a dummy-message diff: it tokenizes a
synthetic ``[dummy_user, dummy_assistant]`` base with and without the
appended messages, then takes the suffix difference as the incremental IDs.
Model-specific subclasses override ``merge_tokens`` to handle boundary
quirks at the junction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template, assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

_DUMMY_USER: dict[str, Any] = {"role": "user", "content": "dummy"}


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dummy assistant message with tool_calls matching *tool_responses*,
    so the template correctly renders the subsequent tool-response turn boundaries."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id") or f"call0000{i}",
                "type": "function",
                "function": {
                    "name": resp.get("name") or "dummy_func",
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }


# ---------------------------------------------------------------------------
# Base / default tokenizer (dummy-prefix diff)
# ---------------------------------------------------------------------------


class TITOTokenizer:
    """Incremental tokenization and prefix merging using dummy-message diff.

    A synthetic base ``[dummy_user, dummy_assistant]`` simulates the assistant
    turn boundary so that the diff captures the correct turn-transition tokens:

    1. ``tokens_without`` = tokenize(base, add_generation_prompt=False)
    2. ``tokens_with``    = tokenize(base + appended, add_generation_prompt=True)
    3. ``incremental_ids  = tokens_with[len(tokens_without):]``

    Subclasses override ``merge_tokens`` to handle model-specific boundary
    token quirks.
    """

    max_trim_tokens: int = 0
    trailing_token_ids: frozenset[int] = frozenset()

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._assistant_start_str = assistant_start_str
        self.allowed_append_roles: list[str] = allowed_append_roles if allowed_append_roles is not None else ["tool"]

    def create_comparator(self) -> TokenSeqComparator:
        """Create a :class:`TokenSeqComparator` configured with this
        tokenizer's model-specific settings."""
        return TokenSeqComparator(
            self.tokenizer,
            assistant_start_str=self._assistant_start_str,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for non-assistant messages appended
        after the pretokenized prefix.

        Only handles tool responses, system injections, etc. — never an
        assistant message.  Validates that *new_messages* is an append-only
        extension of *old_messages* via
        ``assert_messages_append_only_with_allowed_role``.

        Args:
            old_messages: Previously stored messages (prefix).
            new_messages: Full new message list (must be a superset of
                *old_messages* with only allowed-role messages appended).
            tools: Tool definitions in OpenAI format (may vary per call).

        Returns:
            Incremental token IDs (including the generation prompt) that,
            when merged with pretokenized prefix via ``merge_tokens``,
            form the full prompt token IDs.
        """
        assert_messages_append_only_with_allowed_role(old_messages, new_messages, self.allowed_append_roles)
        appended_messages = new_messages[len(old_messages) :]

        dummy_assistant = _build_dummy_assistant(appended_messages)
        base_messages = [_DUMMY_USER, dummy_assistant]

        tokens_without = apply_chat_template(
            base_messages,
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            **self.chat_template_kwargs,
        )
        tokens_with = apply_chat_template(
            base_messages + list(appended_messages),
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            **self.chat_template_kwargs,
        )

        return list(tokens_with[len(tokens_without) :])

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Merge *pretokenized_token_ids* with incremental tokens to produce
        the complete prompt token IDs (including generation prompt).

        The default implementation is simple concatenation.  Subclasses
        override this to handle model-specific boundary token logic.
        """
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        return list(pretokenized_token_ids) + incremental


# ---------------------------------------------------------------------------
# Qwen3 implementation
# ---------------------------------------------------------------------------


class Qwen3TITOTokenizer(TITOTokenizer):
    """Qwen3 variant: handles missing newline at the boundary.

    The Qwen3 chat template emits ``<|im_end|>\\n`` after every message, but
    the model stops at ``<|im_end|>`` without generating the trailing ``\\n``.
    ``merge_tokens`` inserts the missing newline so that the pretokenized
    prefix matches the canonical template output.
    """

    _default_assistant_start_str: str = "<|im_start|>assistant"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
        return prefix + incremental


# ---------------------------------------------------------------------------
# GLM 4.7 implementation
# ---------------------------------------------------------------------------


class GLM47TITOTokenizer(TITOTokenizer):
    """GLM 4.7 variant: handles ambiguous boundary tokens in ``merge_tokens``.

    ``<|user|>`` and ``<|observation|>`` are both assistant stop tokens *and*
    next-message start tokens in the chat template.  In ``merge_tokens``,
    the last token of the pretokenized prefix is always stripped when it is
    one of these boundary tokens — whether it matches the first incremental
    token (overlap) or differs (e.g. model stopped with ``<|observation|>`` but
    next turn is ``<|user|>`` because the tool call failed and a system message
    is injected instead).
    """

    max_trim_tokens: int = 1
    _default_assistant_start_str: str = "<|assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}
        self.trailing_token_ids = frozenset(self._ambiguous_boundary_ids)

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
        return prefix + incremental


# ---------------------------------------------------------------------------
# Enum + Registry + Factory
# ---------------------------------------------------------------------------


class TITOTokenizerType(str, Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    GLM47 = "glm47"


_TOKENIZER_REGISTRY: dict[TITOTokenizerType, type[TITOTokenizer]] = {
    TITOTokenizerType.DEFAULT: TITOTokenizer,
    TITOTokenizerType.QWEN3: Qwen3TITOTokenizer,
    TITOTokenizerType.GLM47: GLM47TITOTokenizer,
}


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
    assistant_start_str: str | None = None,
    allowed_append_roles: list[str] | None = None,
) -> TITOTokenizer:
    """Create a ``TITOTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--tito-model`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template``.
        assistant_start_str: Decoded text prefix identifying assistant content
            segments (e.g. ``"<|im_start|>assistant"``).  Auto-detected from
            the chat template by default; pass explicitly to override.
        allowed_append_roles: Roles allowed in appended messages.  Defaults to
            ``["tool"]``.  Passed to
            ``assert_messages_append_only_with_allowed_role``.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must not be None")
    if isinstance(tokenizer_type, str):
        tokenizer_type = TITOTokenizerType(tokenizer_type)
    cls = _TOKENIZER_REGISTRY[tokenizer_type]
    kwargs: dict[str, Any] = {"chat_template_kwargs": chat_template_kwargs}
    if assistant_start_str is not None:
        kwargs["assistant_start_str"] = assistant_start_str
    if allowed_append_roles is not None:
        kwargs["allowed_append_roles"] = allowed_append_roles
    return cls(tokenizer, **kwargs)
