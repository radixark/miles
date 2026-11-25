"""GLM-family TITO tokenizers: GLM-4.7 and GLM-5.3."""

from __future__ import annotations

import logging
from typing import Any

from miles.rollout.session.errors import MessageValidationError
from miles.utils.chat_template_utils.tito_tokenizer.base import FixedTemplate, TITOTokenizer
from miles.utils.chat_template_utils.truncation import MarkerPair, TruncationGrammar
from miles.utils.reasoning import REASONING_DELIMITERS

logger = logging.getLogger(__name__)


# No end-of-turn token: the next message's role token ends the turn.
_GLM_TRUNCATION = TruncationGrammar(
    think=MarkerPair(*REASONING_DELIMITERS["think"]),
    tool_blocks=(
        MarkerPair("<tool_call>", "</tool_call>"),
        MarkerPair("<arg_key>", "</arg_key>"),
        MarkerPair("<arg_value>", "</arg_value>"),
    ),
    end_of_turn=None,
)


class GLM47TITOTokenizer(TITOTokenizer):
    """GLM 4.7 variant: handles ambiguous boundary tokens in ``merge_tokens``.

    ``<|user|>`` and ``<|observation|>`` are both assistant stop tokens *and*
    next-message start tokens in the chat template.  In ``merge_tokens``,
    the last token of the pretokenized prefix is stripped when it is one of
    these boundary tokens.  When the first incremental token is the same token
    (a tool message after ``<|observation|>``, a user message after
    ``<|user|>``), the strip only drops the duplicate.  When it differs (e.g.
    the model stopped with ``<|observation|>`` but the next message is a user
    message because the tool call failed to parse or went unanswered), the
    strip replaces the sampled stop token: allowed by default for this family
    with one warning per replacement, rejected under
    ``--no-allow-stop-token-replace``.

    Because of that replacement the runtime accepts every role after an
    assistant with or without ``tool_calls``, so the continuation roles keep
    their maximal default even though the model expects ``tool`` after a call.
    """

    reasoning_parser = "glm45"
    tool_call_parser = "glm47"

    # GLM's HF-native chat template already exposes a ``clear_thinking`` kwarg,
    # so no fixed-jinja patch is needed for either append surface.
    FIXED_TEMPLATE = FixedTemplate(
        template=None,
        extra_kwargs={"clear_thinking": False},
        truncation_closer=_GLM_TRUNCATION,
    )

    max_trim_tokens: int = 1
    allow_stop_token_replace: bool = True
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
            if prefix[-1] != incremental[0]:
                sampled, replacement = self.tokenizer.convert_ids_to_tokens([prefix[-1], incremental[0]])
                detail = (
                    f"sampled stop token {sampled} is replaced by {replacement}, "
                    "the first token of the appended render"
                )
                if not self.allow_stop_token_replace:
                    raise MessageValidationError(f"{detail}; rejected by --no-allow-stop-token-replace")
                logger.warning("%s; that turn's training sample drops the sampled token", detail)
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
        truncation_closer=_GLM_TRUNCATION,
    )
