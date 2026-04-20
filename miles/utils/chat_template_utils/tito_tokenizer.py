"""TITO tokenizer — incremental tokenization for pretokenized prefix reuse.

``TITOTokenizer`` computes incremental token IDs for non-assistant messages
(tool responses, user follow-ups, system injections) that follow the
assistant's generated token sequence, then merges them with the pretokenized
prefix — handling model-specific boundary tokens at the junction.

The default implementation incrementally tokenizes appended non-assistant turns
with role-specific synthetic prefixes:

- contiguous ``tool`` runs use ``[dummy_system, dummy_assistant]``
- each ``user`` or ``system`` message uses ``[dummy_system]``

The appended suffix is processed left-to-right, then the generation prompt for
the next assistant turn is appended once at the end.  Model-specific
subclasses only override ``merge_tokens`` for boundary quirks at the prefix
junction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template, assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


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
# Base / default tokenizer
# ---------------------------------------------------------------------------


class TITOTokenizer:
    """Incremental tokenization and prefix merging for appended non-assistant turns."""

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

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        return apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _split_appended_segments(self, appended_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        segments: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(appended_messages):
            role = appended_messages[i]["role"]
            # Many templates wrap a contiguous tool-response run as one logical
            # block, so tool messages are diffed together instead of one-by-one.
            if role == "tool":
                j = i + 1
                while j < len(appended_messages) and appended_messages[j]["role"] == "tool":
                    j += 1
                segments.append(appended_messages[i:j])
                i = j
                continue
            if role in {"user", "system"}:
                segments.append([appended_messages[i]])
                i += 1
                continue
            raise ValueError(f"unsupported appended role for TITO segmentation: {role}")

        return segments

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render *base_messages* and *base_messages + appended_messages*, return
        tokens for the suffix.

        When *add_generation_prompt* is True and *appended_messages* is empty,
        this computes the generation-prompt suffix (the assistant opener tokens).
        """
        text_without = self._render_messages(base_messages, add_generation_prompt=False, tools=tools)
        text_with = self._render_messages(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if not text_with.startswith(text_without):
            roles = [msg["role"] for msg in appended_messages] if appended_messages else ["generation_prompt"]
            raise ValueError(f"rendered suffix diff failed for {roles}")
        return self._encode_text(text_with[len(text_without) :])

    def _tokenize_tool_segment(
        self,
        appended_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # No dummy user to avoid cut think issues.
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM, _build_dummy_assistant(appended_messages)],
            appended_messages,
            tools=tools,
        )

    def _tokenize_user_and_system_segment(
        self,
        appended_message: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # User/system single-message appends share one synthetic context.
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM],
            [appended_message],
            tools=tools,
        )

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for non-assistant messages appended
        after the pretokenized prefix.

        Handles tool responses, user, and system messages —
        never an assistant message.  Validates that *new_messages* is an
        append-only extension of *old_messages* via
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
        incremental: list[int] = []

        # Incremental non-assistant content is assembled segment-by-segment
        # using the smallest synthetic context that preserves each role's
        # boundary tokens.
        for segment in self._split_appended_segments(appended_messages):
            role = segment[0]["role"]
            if role == "tool":
                incremental.extend(self._tokenize_tool_segment(segment, tools))
            elif role == "user" or role == "system":
                incremental.extend(self._tokenize_user_and_system_segment(segment[0], tools))
            else:
                raise ValueError(f"unsupported appended role for TITO tokenization: {role}")

        # The next assistant opener depends on the full post-append history, so
        # it is derived from the real messages once and appended only at the end.
        return incremental + self._tokenize_rendered_suffix(
            new_messages,
            [],
            tools=tools,
            add_generation_prompt=True,
        )

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
