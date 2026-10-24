"""GLM-family TITO tokenizers: GLM-4.7 and GLM-5.3."""

from __future__ import annotations

from typing import Any

from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer


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
    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False},
    )

    max_trim_tokens: int = 1
    _default_assistant_start_str: str = "<|assistant|>"

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
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}
        self.trailing_token_ids = frozenset(self._ambiguous_boundary_ids)

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
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
        return prefix + incremental


class GLM53TITOTokenizer(GLM47TITOTokenizer):
    """GLM-5.3 native text renderer with the shared GLM token boundary.

    The GLM-5.3 and GLM-5.3-Flash templates start generation with ``<think>`` even when ``enable_thinking=False``, so this family pins ``enable_thinking=True``. Flash support covers tokenizer text inputs, not multimodal processor inputs.
    """

    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False, "enable_thinking": True},
        consistant_kwargs=["reasoning_effort"],
    )
