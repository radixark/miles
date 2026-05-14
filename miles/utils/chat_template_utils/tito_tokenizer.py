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

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from miles.utils.chat_template_utils.template import apply_chat_template, assert_messages_append_only_with_allowed_role
from miles.utils.chat_template_utils.token_seq_comparator import TokenSeqComparator

logger = logging.getLogger(__name__)

# Bundled fixed-template files live under this directory; ``FixedTemplateRow.template``
# values are filenames relative to it.
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Roles the TITO merge logic understands; passing anything else is a typo.
_VALID_ROLES = frozenset({"tool", "user", "system"})

_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


@dataclass(frozen=True)
class FixedTemplateRow:
    """A ``(roles, template, extra_kwargs)`` row owned by a TITO tokenizer family.

    Each row says: when the session is configured for ``allowed_roles``, this
    family expects the given chat template plus the given extra kwargs.
    ``template`` is a path relative to ``TEMPLATE_DIR`` for a bundled fixed
    template, or ``None`` to keep the HF-native template (kwargs-only fix).
    """

    allowed_roles: frozenset[str]
    template: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


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
# TODO: split different model's TITO tokenizer into different files


class TITOTokenizer:
    """Incremental tokenization and prefix merging for appended non-assistant turns."""

    max_trim_tokens: int = 0
    trailing_token_ids: frozenset[int] = frozenset()

    # ``(roles, template, extra_kwargs)`` rows this family supports.  Resolved
    # by ``resolve_fixed_chat_template`` via smallest-superset match against
    # the caller's ``allowed_append_roles``.
    SUPPORTED_TEMPLATES: tuple[FixedTemplateRow, ...] = ()

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
        allowed_append_roles: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._assistant_start_str = assistant_start_str
        self.allowed_append_roles: list[str] = allowed_append_roles if allowed_append_roles is not None else ["tool"]
        self.special_token_ids: set[int] = special_token_ids

    def create_comparator(self) -> TokenSeqComparator:
        """Create a :class:`TokenSeqComparator` configured with this
        tokenizer's model-specific settings."""
        return TokenSeqComparator(
            self.tokenizer,
            assistant_start_str=self._assistant_start_str,
            special_token_ids=self.special_token_ids,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
    ) -> str | list[int]:
        return apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tokenize=tokenize,
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
        text_without = self.render_messages(base_messages, add_generation_prompt=False, tools=tools)
        text_with = self.render_messages(
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

    reasoning_parser = "qwen3"
    tool_call_parser = "qwen25"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )

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


# Qwen3.5 and Qwen3-Next-Thinking share the ``<|im_end|>`` boundary handling
# with Qwen3, so they reuse Qwen3TITOTokenizer's token-level logic via plain
# inheritance.  They are still split into named subclasses because each owns
# its own ``SUPPORTED_TEMPLATES`` row pointing to a distinct fixed jinja, even
# though their boundary behavior is identical.


class Qwen35TITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3.5 — same boundary behavior as Qwen3, distinct fixed template."""

    tool_call_parser = "qwen3_coder"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3.5_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3.5_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )


class QwenNextTITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3-Thinking-2507 / Qwen3-Next-Thinking — same boundary behavior as
    Qwen3, distinct (shared) fixed template."""

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3_thinking_2507_and_next_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3_thinking_2507_and_next_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )


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

    reasoning_parser = "glm45"
    tool_call_parser = "glm47"

    # GLM's HF-native chat template already exposes a ``clear_thinking`` kwarg,
    # so no fixed-jinja patch is needed for either append surface.
    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"clear_thinking": False},
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user", "system"}),
            template=None,
            extra_kwargs={"clear_thinking": False},
        ),
    )

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
# Nemotron 3 implementation
# ---------------------------------------------------------------------------


class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):
    """NVIDIA Nemotron 3 family: ``<|im_end|>\\n`` message boundaries.

    Inherits Qwen3's boundary handling — Nemotron 3 emits the same
    ``<|im_end|>\\n`` after every message and the model stops at
    ``<|im_end|>`` without the trailing newline.

    No fixed jinja is shipped — HF native template is append-only when
    ``truncate_history_thinking=False``.  Multi-user-turn surfaces
    auto-merge that kwarg via ``extra_kwargs`` below; ``{tool}``-only does
    not need it (no user-turn boundary to truncate across).

    The plain-text assistant turn does not roundtrip cleanly under
    sglang's upstream ``nemotron_3`` reasoning parser (it keeps a trailing
    ``\\n`` in ``reasoning_content``), so step-4 ``assistant_text`` soft
    assertion is expected to fail until the parser is patched upstream —
    out of scope for this family registration.
    """

    reasoning_parser = "nemotron_3"
    tool_call_parser = "qwen3_coder"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"truncate_history_thinking": False},
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user", "system"}),
            template=None,
            extra_kwargs={"truncate_history_thinking": False},
        ),
    )

    _default_assistant_start_str: str = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Kimi K2 implementation
# ---------------------------------------------------------------------------


def _kimi_segment_special_token_ids(tokenizer: Any) -> set[int]:
    """Kimi specials minus ``<|im_middle|>`` (intra-turn role-name/body
    separator, not a role boundary; must not be a segment boundary)."""
    return TokenSeqComparator._collect_special_ids(tokenizer) - {tokenizer.convert_tokens_to_ids("<|im_middle|>")}


class Kimi25TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.5: ``<|im_end|>`` boundary (no trailing newline).

    K2.5 has no kwarg escape hatch for the "drop reasoning of prior assistants
    once a new non-tool-call assistant arrives" behavior.  Ships a
    bundled fixed jinja that wraps the ``last_non_tool_call_assistant_msg``
    loop in ``{%- if not preserve_thinking -%}`` so multi-user-turn rollout
    can pass ``preserve_thinking=True`` to keep history append-only.  Only the
    ``{tool, user}`` surface is registered (per current onboarding scope).
    """

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="kimi_k25_fixed.jinja",
            extra_kwargs={"preserve_thinking": True},
        ),
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

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
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
            allowed_append_roles=allowed_append_roles,
        )


class Kimi26TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.6: same boundary as K2.5 + native ``preserve_thinking`` kwarg.

    K2.6's HF-native template already carries the ``preserve_thinking`` gate
    that K2.5 needs patched in.  No bundled fixed
    template required; ``{tool, user}`` row registers ``template=None`` and
    auto-merges ``preserve_thinking=True`` for multi-user-turn rollout.

    Tool-call parser is bound to ``kimi_k2_raw_id`` rather than ``kimi_k2``:
    RL trajectories need the model-emitted ``tool_call_id`` to round-trip
    verbatim across turns (no ``history_tool_calls_cnt`` renumbering), and
    miles is the primary consumer of this TITO family.
    """

    reasoning_parser = "kimi_k2"
    tool_call_parser = "kimi_k2_raw_id"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"preserve_thinking": True},
        ),
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

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
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
            allowed_append_roles=allowed_append_roles,
        )


# ---------------------------------------------------------------------------
# MiniMax M2 implementation
# ---------------------------------------------------------------------------


class MinimaxM2TITOTokenizer(TITOTokenizer):
    """MiniMax-M2 family: bespoke ``]~!b[`` / ``[e~[`` / ``]~b]`` tag set.

    M2 chat template uses ``]~!b[`` (BOS), ``[e~[`` (EOS), ``]~b]`` (role
    open marker).  Like Qwen3, the chat template emits ``[e~[\\n`` after
    every message, but the model stops at ``[e~[`` without generating the
    trailing ``\\n``.  ``merge_tokens`` inserts the missing newline so that
    the pretokenized prefix matches the canonical template output.

    Reasoning is gated by a per-message ``last_user_index`` check:
    ``reasoning_content`` is only rendered for assistant turns *after* the
    last ``user`` — appending a new ``user`` therefore strips prior assistant
    ``<think>`` blocks and breaks append-only.  Only ``{tool}`` surface is
    registered for that reason; multi-user-turn requires a fixed jinja that
    always preserves history reasoning.
    """

    reasoning_parser = "minimax-append-think"
    tool_call_parser = "minimax-m2"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="minimax_m2_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )

    _default_assistant_start_str: str = "]~b]ai"

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
        self._eos_id: int = tokenizer.convert_tokens_to_ids("[e~[")
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
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
        return prefix + incremental


# ---------------------------------------------------------------------------
# Enum + Registry + Factory
# ---------------------------------------------------------------------------


class TITOTokenizerType(str, Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    QWENNEXT = "qwennext"
    GLM47 = "glm47"
    NEMOTRON3 = "nemotron3"
    KIMI25 = "kimi25"
    KIMI26 = "kimi26"
    MINIMAX_M2 = "minimax_m2"


_TOKENIZER_REGISTRY: dict[TITOTokenizerType, type[TITOTokenizer]] = {
    TITOTokenizerType.DEFAULT: TITOTokenizer,
    TITOTokenizerType.QWEN3: Qwen3TITOTokenizer,
    TITOTokenizerType.QWEN35: Qwen35TITOTokenizer,
    TITOTokenizerType.QWENNEXT: QwenNextTITOTokenizer,
    TITOTokenizerType.GLM47: GLM47TITOTokenizer,
    TITOTokenizerType.NEMOTRON3: Nemotron3TITOTokenizer,
    TITOTokenizerType.KIMI25: Kimi25TITOTokenizer,
    TITOTokenizerType.KIMI26: Kimi26TITOTokenizer,
    TITOTokenizerType.MINIMAX_M2: MinimaxM2TITOTokenizer,
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


# ---------------------------------------------------------------------------
# Fixed-template resolution (smallest-superset over SUPPORTED_TEMPLATES)
# ---------------------------------------------------------------------------


def resolve_fixed_chat_template(
    tito_model: TITOTokenizerType | str,
    allowed_append_roles: Iterable[str],
) -> tuple[str | None, dict[str, Any]]:
    """Smallest-superset lookup over the requested family's ``SUPPORTED_TEMPLATES``.

    Returns ``(template_path, extra_kwargs)``:

    - ``template_path``: absolute path to a bundled ``.jinja`` file, or ``None``
      when the matched row registers HF-native (kwargs-only fix) or when no
      row matches at all.
    - ``extra_kwargs``: kwargs the caller should merge into
      ``apply_chat_template`` (caller's explicit user kwargs win on conflict).
      Empty when no row matches or the matched row needs none.

    Raises ``ValueError`` on equally-minimal supersets — register a stricter
    row to disambiguate.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)

    requested = frozenset(allowed_append_roles)
    invalid = requested - _VALID_ROLES
    if invalid:
        raise ValueError(
            f"Unknown roles in allowed_append_roles: {sorted(invalid)}. " f"Supported: {sorted(_VALID_ROLES)}."
        )

    cls = _TOKENIZER_REGISTRY[tito_model]
    candidates = [row for row in cls.SUPPORTED_TEMPLATES if requested.issubset(row.allowed_roles)]
    if not candidates:
        raise ValueError(
            f"No SUPPORTED_TEMPLATES row registered for tito_model={tito_model.value} "
            f"with allowed_append_roles={sorted(requested)}. Register a row in "
            f"{cls.__name__}.SUPPORTED_TEMPLATES (template=None for HF-native models)."
        )

    # Pick the most specific superset. Ties surface registration mistakes
    # immediately rather than depending on iteration order.
    min_size = min(len(row.allowed_roles) for row in candidates)
    minimal = [row for row in candidates if len(row.allowed_roles) == min_size]
    if len(minimal) > 1:
        raise ValueError(
            f"Ambiguous fixed-template registration for tito_model={tito_model.value}, "
            f"requested_roles={sorted(requested)}: multiple equally-minimal supersets "
            f"{[sorted(row.allowed_roles) for row in minimal]}. Register a stricter row to disambiguate."
        )
    row = minimal[0]

    path = str(TEMPLATE_DIR / row.template) if row.template else None
    logger.info(
        "tito_model=%s requested_roles=%s -> matched registered_roles=%s -> template=%s kwargs=%s",
        tito_model.value,
        sorted(requested),
        sorted(row.allowed_roles),
        path,
        row.extra_kwargs,
    )
    return path, dict(row.extra_kwargs)


# ---------------------------------------------------------------------------
# sglang parser resolution (per-family binding + assert-equal on user input)
# ---------------------------------------------------------------------------


def resolve_reasoning_and_tool_call_parser(
    tito_model: TITOTokenizerType | str,
    user_reasoning_parser: str | None = None,
    user_tool_call_parser: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve sglang ``--reasoning-parser`` and ``--tool-call-parser`` for the
    given TITO family.

    Both parsers are bound on the TITO subclass as class attributes because
    the model's reasoning / tool-call emission shapes are per-family facts.
    For each parser independently:

    * If the user didn't pass a value, return the family's bound value
      (which may itself be ``None`` for ``DEFAULT`` or unbound subclasses
      — the caller is then responsible for supplying one downstream).
    * If the user passed a value and the family is bound, assert equality;
      a mismatch is a configuration bug and raises ``ValueError`` rather
      than silently overriding.
    * If the user passed a value and the family is unbound, accept it.

    Returns ``(reasoning_parser, tool_call_parser)``.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)
    cls = _TOKENIZER_REGISTRY[tito_model]

    def _resolve_one(field: str, bound: str | None, user: str | None) -> str | None:
        if user is None:
            return bound
        if bound is None:
            return user
        if user != bound:
            raise ValueError(
                f"--{field.replace('_', '-')}={user!r} disagrees with the parser "
                f"registered for tito_model={tito_model.value!r}: {bound!r}. The "
                f"parser is bound on the TITO subclass; either pass {bound!r} or "
                f"omit the flag to auto-resolve."
            )
        return user

    return (
        _resolve_one("reasoning_parser", cls.reasoning_parser, user_reasoning_parser),
        _resolve_one("tool_call_parser", cls.tool_call_parser, user_tool_call_parser),
    )
