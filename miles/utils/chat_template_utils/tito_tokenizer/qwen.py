"""Qwen-family TITO tokenizers: Qwen3, Qwen3.5, Qwen3.6, Qwen3.8 Small, and Qwen3-Next."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer


class Qwen3TITOTokenizer(TITOTokenizer):
    """Qwen3 variant: handles missing newline at the boundary.

    The Qwen3 chat template emits ``<|im_end|>\\n`` after every message, but
    the model stops at ``<|im_end|>`` without generating the trailing ``\\n``.
    ``merge_tokens`` inserts the missing newline so that the pretokenized
    prefix matches the canonical template output.
    """

    reasoning_parser = "qwen3"
    tool_call_parser = "qwen25"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
    )

    _default_assistant_start_str: str = "<|im_start|>assistant"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
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
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        prefix = list(pretokenized_token_ids)
        # Weakly post-trained Qwen3 models, notably 0.6B, may emit the pretraining/padding token `<|endoftext|>`.
        # Seen in TITO's March 2026 bring-up; rare in larger models. See https://github.com/radixark/miles/issues/3113.
        # This is model degeneration, not a valid `<|im_end|>` alias; keep the strict checker reporting it.
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
        return prefix + incremental


# Qwen3.5/3.6 and Qwen3-Next-Thinking share the ``<|im_end|>`` boundary
# handling with Qwen3. Their subclasses only own the distinct fixed-template
# contracts layered on that token boundary.


class Qwen35TITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3.5 template and Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.5_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id"],
    )


class Qwen36TITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3.6 template and Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.6_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id"],
    )


class Qwen38SmallTITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3.8 reasoning-effort template with the Qwen3 token boundary."""

    tool_call_parser = "qwen3_coder"

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3.8_small_and_flash_next_fixed.jinja",
        extra_kwargs={"preserve_thinking": True},
        allowed_append_roles=frozenset({"tool", "user", "assistant"}),
        consistant_kwargs=["add_vision_id", "enable_thinking", "reasoning_effort"],
    )


class QwenNextTITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3-Thinking-2507 / Qwen3-Next-Thinking — same boundary behavior as
    Qwen3, distinct (shared) fixed template."""

    FIXED_TEMPLATE = FixedTemplate(
        template="qwen3_thinking_2507_and_next_fixed.jinja",
        extra_kwargs={"clear_thinking": False},
    )
