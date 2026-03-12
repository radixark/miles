"""Incremental message tokenization for pretokenized prefix reuse.

``AdditionalMessageTokenizer`` computes the token IDs for messages appended
after an already-tokenized prefix.  The default implementation uses a
dummy-message diff (mirrors sglang's ``calc_additional_message_tokenization_by_dummy``).
Model-specific subclasses handle quirks such as GLM 4.7's ambiguous boundary
tokens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template

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


class AdditionalMessageTokenizer(ABC):
    """Base class for computing incremental token IDs for new messages."""

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    @abstractmethod
    def tokenize_additional(
        self,
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for *new_messages*.

        Args:
            new_messages: Messages to tokenize (tool responses, system
                injections, etc.) appended after the pretokenized prefix.
            pretokenized_token_ids: Token IDs covering everything before
                *new_messages*, including the last assistant turn's
                completion tokens.
            tools: Tool definitions in OpenAI format (may vary per call).

        Returns:
            Incremental token IDs that, when concatenated to
            *pretokenized_token_ids*, form the full prompt token IDs.
        """
        ...

    def should_strip_trailing_stop_token(self, token_ids: list[int]) -> bool:
        """Check if the last token is a model stop token that should be stripped.

        Model stop tokens (e.g. GLM 4.7's ``<|user|>``/``<|observation|>``) are
        generation artifacts, not part of the template rendering.

        Default: ``False`` (never strip).
        """
        return False

    def get_trim_trailing_ids(self) -> set[int]:
        """Token IDs to trim from sequence tails before comparison.

        When comparing expected (full re-tokenization) vs actual (prompt +
        completion) sequences, the two may differ at the boundaries:

        - Qwen3: expected ends with ``\\n`` after ``<|im_end|>`` (template
          artifact), actual does not.
        - GLM 4.7: actual ends with ``<|user|>`` or ``<|observation|>``
          (stop token reused as next-turn start), expected does not.

        Returns an empty set by default (no trimming).
        """
        return set()


# ---------------------------------------------------------------------------
# Default implementation (dummy-prefix diff)
# ---------------------------------------------------------------------------


class DefaultAdditionalMessageTokenizer(AdditionalMessageTokenizer):
    """Dummy-prefix diff approach.

    1. Build dummy base: ``[dummy_user, dummy_assistant]``
    2. ``tokens_without`` = tokenize base (no generation prompt)
    3. ``tokens_with``    = tokenize base + *new_messages* (with generation prompt)
    4. ``incremental_ids  = tokens_with[len(tokens_without):]``
    5. Call ``postprocess`` hook (identity by default, subclasses may override)
    """

    def tokenize_additional(
        self,
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        dummy_assistant = _build_dummy_assistant(new_messages)
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
            base_messages + list(new_messages),
            tokenizer=self.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            **self.chat_template_kwargs,
        )

        incremental_ids = list(tokens_with[len(tokens_without) :])

        return self.postprocess(incremental_ids, tokens_without, pretokenized_token_ids)

    def postprocess(
        self,
        incremental_ids: list[int],
        tokens_without: list[int],
        pretokenized_token_ids: list[int],
    ) -> list[int]:
        """Post-process incremental IDs.  Default: return as-is."""
        return incremental_ids


# ---------------------------------------------------------------------------
# Qwen3 implementation (trailing whitespace alignment)
# ---------------------------------------------------------------------------


class Qwen3AdditionalMessageTokenizer(DefaultAdditionalMessageTokenizer):
    """Qwen3 variant: prepends trailing whitespace token for alignment.

    Qwen3 templates insert a trailing ``\\n`` after the last assistant
    message (loss-mask 0).  When this token differs from the real
    pretokenized suffix, we prepend it to ``incremental_ids`` so that
    concatenation with the stored prefix produces the correct sequence.
    """

    def get_trim_trailing_ids(self) -> set[int]:
        return {self.tokenizer.encode("\n", add_special_tokens=False)[-1]}

    def postprocess(
        self,
        incremental_ids: list[int],
        tokens_without: list[int],
        pretokenized_token_ids: list[int],
    ) -> list[int]:
        if (
            pretokenized_token_ids
            and tokens_without
            and tokens_without[-1] != pretokenized_token_ids[-1]
            and self.tokenizer.decode([tokens_without[-1]]).strip() == ""
        ):
            incremental_ids = [tokens_without[-1]] + incremental_ids

        return incremental_ids


# ---------------------------------------------------------------------------
# GLM 4.7 implementation
# ---------------------------------------------------------------------------


class GLM47AdditionalMessageTokenizer(DefaultAdditionalMessageTokenizer):
    """GLM 4.7 variant: strips trailing stop token from stored token IDs.

    ``<|user|>`` and ``<|observation|>`` are both assistant stop tokens *and*
    next-message start tokens in the chat template.  By stripping the stop
    token from ``session.token_ids`` before the next turn, the template's
    boundary token (e.g. ``<|observation|>`` for tool responses, ``<|system|>``
    for retry messages) is always the first token of ``incremental_ids``,
    producing an exact match with full template rendering.
    """

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(tokenizer, chat_template_kwargs)
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}

    def should_strip_trailing_stop_token(self, token_ids: list[int]) -> bool:
        return bool(token_ids) and token_ids[-1] in self._ambiguous_boundary_ids

    def get_trim_trailing_ids(self) -> set[int]:
        return set(self._ambiguous_boundary_ids)


# ---------------------------------------------------------------------------
# Enum + Registry + Factory
# ---------------------------------------------------------------------------


class AdditionalMessageTokenizerType(str, Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    GLM47 = "glm47"


_TOKENIZER_REGISTRY: dict[AdditionalMessageTokenizerType, type[AdditionalMessageTokenizer]] = {
    AdditionalMessageTokenizerType.DEFAULT: DefaultAdditionalMessageTokenizer,
    AdditionalMessageTokenizerType.QWEN3: Qwen3AdditionalMessageTokenizer,
    AdditionalMessageTokenizerType.GLM47: GLM47AdditionalMessageTokenizer,
}


def get_additional_message_tokenizer(
    tokenizer: Any,
    tokenizer_type: AdditionalMessageTokenizerType | str = AdditionalMessageTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> AdditionalMessageTokenizer:
    """Create an ``AdditionalMessageTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--additional-tokenizer`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``apply_chat_template``.
    """
    if isinstance(tokenizer_type, str):
        resolved = AdditionalMessageTokenizerType(tokenizer_type)
    else:
        resolved = tokenizer_type

    cls = _TOKENIZER_REGISTRY[resolved]
    return cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
