"""TITO tokenizer — incremental message tokenization for pretokenized prefix reuse.

``TITOTokenizer`` computes the token IDs for messages appended after an
already-tokenized prefix, and merges them with the prefix handling
model-specific boundary tokens.

The default implementation uses a dummy-message diff
(mirrors sglang's ``calc_additional_message_tokenization_by_dummy``).
Model-specific subclasses handle quirks such as GLM 4.7's ambiguous boundary
tokens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template, assert_messages_append_only

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy message helpers (mirrors sglang utils.py)
# ---------------------------------------------------------------------------

_DUMMY_USER: dict[str, Any] = {"role": "user", "content": "dummy"}


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dummy assistant message whose tool_calls match *tool_responses*."""
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
# ABC
# ---------------------------------------------------------------------------


class TITOTokenizer(ABC):
    """Base class for incremental tokenization and prefix merging."""

    _max_trim_tokens: int = 0
    _trailing_token_ids: frozenset[int] = frozenset()

    def get_comparator_ignore_trailing_ids(self) -> set[int] | None:
        """Return token IDs that the comparator should strip from sequence
        tails before comparison, or ``None`` if no stripping is needed."""
        return set(self._trailing_token_ids) if self._trailing_token_ids else None

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    @abstractmethod
    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for non-assistant messages appended
        after the pretokenized prefix.

        Only handles tool responses, system injections, etc. — never an
        assistant message.  Validates that *new_messages* is an append-only
        extension of *old_messages* via ``assert_messages_append_only``.

        Args:
            old_messages: Previously stored messages (prefix).
            new_messages: Full new message list (must be a superset of
                *old_messages* with only tool/system messages appended).
            pretokenized_token_ids: Token IDs covering everything up to
                *old_messages*, including the last assistant turn's
                completion tokens.
            tools: Tool definitions in OpenAI format (may vary per call).

        Returns:
            Incremental token IDs (including the generation prompt) that,
            when merged with *pretokenized_token_ids* via ``merge_tokens``,
            form the full prompt token IDs.
        """
        ...

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Merge *pretokenized_token_ids* with incremental tokens to produce
        the complete prompt token IDs (including generation prompt).

        Handles model-specific boundary token logic (e.g. GLM 4.7's
        ambiguous stop-token / next-turn-start overlap).

        Default implementation: simple concatenation.
        """
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, pretokenized_token_ids, tools)
        return list(pretokenized_token_ids) + incremental


# ---------------------------------------------------------------------------
# Default implementation (dummy-prefix diff)
# ---------------------------------------------------------------------------


class DefaultTITOTokenizer(TITOTokenizer):
    """Dummy-prefix diff approach.

    1. Build dummy base: ``[dummy_user, dummy_assistant]``
    2. ``tokens_without`` = tokenize base (no generation prompt)
    3. ``tokens_with``    = tokenize base + *appended_messages* (with generation prompt)
    4. ``incremental_ids  = tokens_with[len(tokens_without):]``
    """

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        assert_messages_append_only(old_messages, new_messages)
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


# ---------------------------------------------------------------------------
# Qwen3 implementation
# ---------------------------------------------------------------------------


class Qwen3TITOTokenizer(DefaultTITOTokenizer):
    """Qwen3 variant.

    TODO: may need to insert/delete a trailing newline at the boundary
    between pretokenized prefix and incremental tokens.  Currently uses
    default concatenation — to be refined after further investigation.
    """


# ---------------------------------------------------------------------------
# GLM 4.7 implementation
# ---------------------------------------------------------------------------


class GLM47TITOTokenizer(DefaultTITOTokenizer):
    """GLM 4.7 variant: handles ambiguous boundary tokens in ``merge_tokens``.

    ``<|user|>`` and ``<|observation|>`` are both assistant stop tokens *and*
    next-message start tokens in the chat template.  In ``merge_tokens``,
    the last token of the pretokenized prefix is always stripped when it is
    one of these boundary tokens — whether it matches the first incremental
    token (avoiding duplication) or differs (replacing a wrong prediction).
    """

    _max_trim_tokens: int = 1

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(tokenizer, chat_template_kwargs)
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}
        self._trailing_token_ids = frozenset(self._ambiguous_boundary_ids)

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, pretokenized_token_ids, tools)
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
    TITOTokenizerType.DEFAULT: DefaultTITOTokenizer,
    TITOTokenizerType.QWEN3: Qwen3TITOTokenizer,
    TITOTokenizerType.GLM47: GLM47TITOTokenizer,
}


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> TITOTokenizer:
    """Create a ``TITOTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--tito-model`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template``.
    """
    if isinstance(tokenizer_type, str):
        resolved = TITOTokenizerType(tokenizer_type)
    else:
        resolved = tokenizer_type

    cls = _TOKENIZER_REGISTRY[resolved]
    return cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
