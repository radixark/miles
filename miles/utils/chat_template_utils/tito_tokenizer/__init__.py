"""TITO tokenizer — incremental tokenization for pretokenized prefix reuse.

``TITOTokenizer`` computes incremental token IDs for messages appended after the assistant's generated token sequence, then merges them with the pretokenized prefix — handling model-specific boundary tokens at the junction.

The default implementation renders the complete appended suffix and the next generation prompt once under a synthetic ``[dummy_system, dummy_assistant]`` prefix.  Model-specific subclasses customize request rules and token-boundary handling.

``TITOTokenizer`` and ``FixedTemplate`` live in ``base``; each model family lives in its own module.
"""

from __future__ import annotations

import logging

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import (
    ALL_APPEND_ROLES,
    TEMPLATE_DIR,
    VALID_APPEND_ROLES,
    FixedTemplate,
    TITOTokenizer,
    extract_template_args,
)
from miles.utils.chat_template_utils.tito_tokenizer.deepseek import (
    DeepSeekV4TITOTokenizer,
    DeepSeekV32TITOTokenizer,
    DeepSeekV41TITOTokenizer,
)
from miles.utils.chat_template_utils.tito_tokenizer.glm import GLM47TITOTokenizer, GLM53TITOTokenizer
from miles.utils.chat_template_utils.tito_tokenizer.inkling import InklingTITOTokenizer
from miles.utils.chat_template_utils.tito_tokenizer.kimi import Kimi25TITOTokenizer, Kimi26TITOTokenizer
from miles.utils.chat_template_utils.tito_tokenizer.minimax import MinimaxM25TITOTokenizer, MinimaxM27TITOTokenizer
from miles.utils.chat_template_utils.tito_tokenizer.nemotron import Nemotron3TITOTokenizer
from miles.utils.chat_template_utils.tito_tokenizer.qwen import (
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    Qwen36TITOTokenizer,
    Qwen38SmallTITOTokenizer,
    QwenNextTITOTokenizer,
)

__all__ = [
    "ALL_APPEND_ROLES",
    "TEMPLATE_DIR",
    "VALID_APPEND_ROLES",
    "FixedTemplate",
    "TITOTokenizer",
    "TITOTokenizerType",
    "extract_template_args",
    "get_tito_tokenizer",
    "resolve_fixed_chat_template",
    "resolve_reasoning_and_tool_call_parser",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enum + Factory
# ---------------------------------------------------------------------------


class TITOTokenizerType(StrEnum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    QWEN36 = "qwen36"
    QWEN38_SMALL = "qwen38small"
    QWEN4_EXP = "qwen4exp"
    QWENNEXT = "qwennext"
    GLM47 = "glm47"
    GLM53 = "glm53"
    NEMOTRON3 = "nemotron3"
    KIMI25 = "kimi25"
    KIMI26 = "kimi26"
    MINIMAX_M25 = "minimax_m25"
    MINIMAX_M27 = "minimax_m27"
    DEEPSEEKV32 = "deepseekv32"
    DEEPSEEKV4 = "deepseekv4"
    DEEPSEEKV41 = "deepseekv41"
    INKLING = "inkling"

    @classmethod
    def get_tokenizer_class(cls, t: TITOTokenizerType) -> type[TITOTokenizer]:
        """Resolve the concrete ``TITOTokenizer`` subclass for *t*."""
        match t:
            case cls.DEFAULT:
                return TITOTokenizer
            case cls.QWEN3:
                return Qwen3TITOTokenizer
            case cls.QWEN35:
                return Qwen35TITOTokenizer
            case cls.QWEN36:
                return Qwen36TITOTokenizer
            case cls.QWEN38_SMALL | cls.QWEN4_EXP:
                return Qwen38SmallTITOTokenizer
            case cls.QWENNEXT:
                return QwenNextTITOTokenizer
            case cls.GLM47:
                return GLM47TITOTokenizer
            case cls.GLM53:
                return GLM53TITOTokenizer
            case cls.NEMOTRON3:
                return Nemotron3TITOTokenizer
            case cls.KIMI25:
                return Kimi25TITOTokenizer
            case cls.KIMI26:
                return Kimi26TITOTokenizer
            case cls.MINIMAX_M25:
                return MinimaxM25TITOTokenizer
            case cls.MINIMAX_M27:
                return MinimaxM27TITOTokenizer
            case cls.DEEPSEEKV32:
                return DeepSeekV32TITOTokenizer
            case cls.DEEPSEEKV4:
                return DeepSeekV4TITOTokenizer
            case cls.DEEPSEEKV41:
                return DeepSeekV41TITOTokenizer
            case cls.INKLING:
                return InklingTITOTokenizer
            case _:
                raise ValueError(f"Unknown TITOTokenizerType: {t!r}")


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
    assistant_start_str: str | None = None,
) -> TITOTokenizer:
    """Create a ``TITOTokenizer`` instance.

    Args:
        tokenizer: HuggingFace tokenizer object.
        tokenizer_type: Explicit type (string or enum).  Corresponds to the
            ``--tito-model`` CLI argument.
        chat_template_kwargs: Extra kwargs forwarded to ``template.apply_chat_template``.
        assistant_start_str: Decoded text prefix identifying assistant content
            segments (e.g. ``"<|im_start|>assistant"``).  Auto-detected from
            the chat template by default; pass explicitly to override.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must not be None")
    if isinstance(tokenizer_type, str):
        tokenizer_type = TITOTokenizerType(tokenizer_type)
    cls = TITOTokenizerType.get_tokenizer_class(tokenizer_type)
    kwargs: dict[str, Any] = {"chat_template_kwargs": chat_template_kwargs}
    if assistant_start_str is not None:
        kwargs["assistant_start_str"] = assistant_start_str
    return cls(tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# Fixed-template resolution (one template per family)
# ---------------------------------------------------------------------------


def resolve_fixed_chat_template(
    tito_model: TITOTokenizerType | str,
) -> tuple[str | None, dict[str, Any]]:
    """The family's fixed chat template and required kwargs.

    Returns ``(template_path, extra_kwargs)``:

    - ``template_path``: absolute path to a bundled ``.jinja`` file, or ``None``
      when the family registers HF-native (kwargs-only fix).
    - ``extra_kwargs``: kwargs owned by the registration and merged into
      ``template.apply_chat_template``.  Conflicting caller values are invalid.

    Template resolution depends only on ``tito_model``.  The DEFAULT family
    resolves to its native template (``None``) with no fixed kwargs.
    """
    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)

    cls = TITOTokenizerType.get_tokenizer_class(tito_model)
    fixed = cls.FIXED_TEMPLATE

    path = str(TEMPLATE_DIR / fixed.template) if fixed.template else None
    logger.info(
        "tito_model=%s -> template=%s kwargs=%s allowed_append_roles=%s",
        tito_model.value,
        path,
        fixed.extra_kwargs,
        sorted(fixed.allowed_append_roles),
    )
    return path, dict(fixed.extra_kwargs)


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
    cls = TITOTokenizerType.get_tokenizer_class(tito_model)

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
