from dataclasses import dataclass, field

from sglang.srt.endpoints.openai.protocol import ChatCompletionMessageParam as ChatMessage

from miles.utils.chat_message_utils import trim_think_tokens


@dataclass
class Turn:
    """
    A turn is a multiple message turn, end with an assistant message.
    """

    messages: list[ChatMessage]
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_log_probs: list[float]

    def __init__(
        self,
        messages: list[ChatMessage],
        prompt_tokens: list[int],
        response_tokens: list[int],
        response_log_probs: list[float],
    ):
        assert len(messages) > 0 and messages[-1].role == "assistant", "The last message must be an assistant message."
        self.messages = messages
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.response_log_probs = response_log_probs


@dataclass
class SeqTrajectory:
    """
    Sequence trajectory state.
    Can only maintain the token info for the last turn.
    """

    num_turns: int = 0
    model_name: str = ""
    # History for all turns.
    turns: list[Turn] = field(default_factory=list)

    # History for all tokens.
    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    def _remove_tokens(self, start_index: int, end_index: int):
        # Notice: the end index is exclusive.
        self.token_ids = self.token_ids[start_index:end_index]
        self.log_probs = self.log_probs[start_index:end_index]
        self.loss_mask = self.loss_mask[start_index:end_index]

    def _insert_tokens_info(self, tokens: list[int], log_probs: list[float], loss_mask: list[int]):
        self.token_ids.extend(tokens)
        self.log_probs.extend(log_probs)
        self.loss_mask.extend(loss_mask)

    def _remove_last_assistant_think(self):
        last = self.turns[-1].messages[-1]
        self._remove_tokens(len(self.token_ids) - len(last.response_tokens), len(self.token_ids))

        trimmed_tokens = trim_think_tokens(last.response_tokens, self.model_name)
        # Notice: after trimming, the old answer tokens cannot be used to calculate loss, so logp and loss mask are set to 0.
        self._insert_tokens_info(trimmed_tokens, [0.0] * len(trimmed_tokens), [0] * len(trimmed_tokens))

    def insert_new_response(self, turn: Turn):
        if self.num_turns > 0 and self.turns[-1].messages[-1].role == "assistant":
            self._remove_last_assistant_think()
        self._insert_tokens_info(turn.response_tokens, turn.response_log_probs, [1] * len(turn.response_tokens))
        self.turns.append(turn)
        self.num_turns += 1
