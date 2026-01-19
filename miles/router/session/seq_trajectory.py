import copy
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from miles.rollout.generate_utils.tokenize_utils import tokenize_messages
from miles.utils.chat_message_utils import calc_last_think_part_index

logger = logging.getLogger(__name__)


class TokenInfo(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    token_ids: list[int] = Field(default_factory=list)
    log_probs: list[float] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)

    def remove_tokens(self, start_index: int, end_index: int):
        # Notice: the end index is exclusive.
        self.tokens = self.tokens[start_index:end_index]
        self.token_ids = self.token_ids[start_index:end_index]
        self.log_probs = self.log_probs[start_index:end_index]
        self.loss_mask = self.loss_mask[start_index:end_index]

    def insert_tokens(self, tokens: list[str], token_ids: list[int], log_probs: list[float], loss_mask: list[int]):
        self.tokens.extend(tokens)
        self.token_ids.extend(token_ids)
        self.log_probs.extend(log_probs)
        self.loss_mask.extend(loss_mask)

    def append(self, token: str, token_id: int, log_prob: float, loss_mask: int):
        self.tokens.append(token)
        self.token_ids.append(token_id)
        self.log_probs.append(log_prob)
        self.loss_mask.append(loss_mask)

    def __add__(self, other: "TokenInfo") -> "TokenInfo":
        return TokenInfo(
            tokens=self.tokens + other.tokens,
            token_ids=self.token_ids + other.token_ids,
            log_probs=self.log_probs + other.log_probs,
            loss_mask=self.loss_mask + other.loss_mask,
        )

    @staticmethod
    def remove_last_assistant_think_and_handle_truncated_message(
        token_info: "TokenInfo", model_name: str
    ) -> "TokenInfo":
        raise NotImplementedError("Not implemented yet.")
        tmp = copy.deepcopy(token_info)
        start, end = calc_last_think_part_index(tmp.token_ids, model_name)
        if start is None:
            # No think part found, or think part is truncated, we will not trim.
            return tmp
        # Notice: after trimming, the old answer tokens cannot be used to calculate loss, so logp and loss mask are set to 0.
        if end is not None:
            tmp.remove_tokens(start, end + 1)
            if end + 1 < len(token_info.token_ids):
                n = len(token_info.token_ids)
                tmp.insert_tokens(
                    token_info.tokens[end + 1 :],
                    token_info.token_ids[end + 1 :],
                    [0.0] * (n - end - 1),
                    [0] * (n - end - 1),
                )
        # Handle truncated message.

        return tmp


class Turn(BaseModel):
    """
    A turn is a multiple message turn, end with an assistant message.
    """

    messages: list[dict[str, Any]]
    prompt_tokens: TokenInfo
    response_tokens: TokenInfo

    def __init__(
        self,
        messages: list[dict[str, Any]],
        prompt_tokens: TokenInfo,
        response_tokens: TokenInfo,
    ):
        assert (
            len(messages) > 0 and messages[-1]["role"] == "assistant"
        ), "The last message must be an assistant message."
        self.messages = messages
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens

    def match_prefix_messages_and_return_remaining(self, other: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """
        If the messages match with other's prefix, return the remaining messages. Otherwise, return None.
        """
        if len(self.messages) < len(other):
            return None
        for i in range(len(other)):
            if self.messages[i] != other[i]:
                return None
        return self.messages[len(other) :]

    def handle_token_out_for_next_turn(self, model_name: str) -> TokenInfo:
        raise NotImplementedError("Not implemented yet.")
        trimmed_tokens = TokenInfo.remove_last_assistant_think(self.prompt_tokens + self.response_tokens, model_name)
        return trimmed_tokens


class SeqTrajectory(BaseModel):
    """
    Sequence trajectory state.
    Can only maintain the token info for the last turn.
    It should not have any state. Which means `token_ids` should always include the final chat templated text.
    (Note: if seq trajectory has state, when a reqeust crash, bug will happen.)
    """

    num_turns: int = 0
    model_name: str = ""
    # History for all turns.
    turns: list[Turn] = Field(default_factory=list)

    def insert_new_turn(self, turn: Turn):
        self.turns.append(turn)
        self.num_turns += 1

    def match_prefix_turns_and_return_last_turn(
        self, messages: list[dict[str, Any]], n: int | None = None
    ) -> tuple[Turn, list[dict[str, Any]]]:
        if n is None:
            n = self.num_turns
        remain_messages = messages
        for i in range(n):
            turn = self.turns[i]
            remain_messages = turn.match_prefix_messages_and_return_remaining(remain_messages)
            if remain_messages is None:
                raise ValueError(
                    "Under sequence trajectory, messages prefix should match, but unmatched messages: {remain_messages}"
                )
        return self.turns[n - 1], remain_messages

    def calc_prompt_tokens_info(
        self,
        messages: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        cross_turn_token_out: bool = True,
        inherit_last_assistant: bool = True,
    ) -> TokenInfo:
        if cross_turn_token_out and self.num_turns > 0:
            if inherit_last_assistant:
                raise NotImplementedError("Not implemented yet.")
                turn, remain_messages = self.match_prefix_messages_and_return_last_turn(messages)
                token_info = turn.handle_token_out_for_next_turn(self.model_name)
            else:
                turn, remain_messages = self.match_prefix_messages_and_return_last_turn(messages, self.num_turns - 1)
                old_token_ids = turn.prompt_tokens.token_ids + turn.response_tokens.token_ids
                new_token_ids = tokenize_messages(remain_messages, tokenizer, add_generation_prompt=True)
                token_ids = old_token_ids + new_token_ids
                # Old token logprobs and loss mask are set to 0.
                log_probs = [0.0] * len(token_ids)
                loss_mask = [0] * len(token_ids)
                token_info = TokenInfo(
                    tokens=tokenizer.convert_ids_to_tokens(token_ids),
                    token_ids=token_ids,
                    log_probs=log_probs,
                    loss_mask=loss_mask,
                )
        else:
            # Retokenize all trajectory tokens, and set logprobs and loss mask to 0.
            token_ids = tokenizer.apply_chat_template(
                self.turns[-1].messages, tokenize=True, add_generation_prompt=True
            )
            log_probs = [0.0] * len(token_ids)
            loss_mask = [0] * len(token_ids)
            token_info = TokenInfo(
                tokens=tokenizer.convert_ids_to_tokens(token_ids),
                token_ids=token_ids,
                log_probs=log_probs,
                loss_mask=loss_mask,
            )

        return token_info

    def get_last_turn_token_info(self) -> TokenInfo:
        return self.turns[-1].prompt_tokens + self.turns[-1].response_tokens


class SeqTrajectoryManager:
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.sessions: dict[str, SeqTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SeqTrajectory()
        return session_id

    def get_session_by_id(self, session_id: str) -> TokenInfo | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.get_last_turn_token_info()

    def calc_prompt_tokens(self, session_id: str, messages: list[dict[str, Any]]) -> TokenInfo | None:
        # Notice: Sequence trajectory manager will support the prefix of input messages match with the only history.
        session = self.sessions.get(session_id)
        if session is None:
            return None
        token_info: TokenInfo = session.calc_prompt_tokens_info(
            messages,
            self.tokenizer,
            cross_turn_token_out=self.args.cross_turn_token_out,
            inherit_last_assistant=self.args.inherit_last_assistant,
        )
        return token_info
        # if remain_messages is None:
        # TODO(jiajun): Should we truncate think part of the last turn's assistant message, if the new turn does not include any new message?
        # Turn 1: sys | user | assistant | tool | assistant
        # Turn 2: sys | user | assistant | tool | assistant | ???
        # Noral:  sys | user | assistant | tool | assistant | ???
        # Not hard to fix, but temporarily leave this TODO.
        # raise ValueError("Currently, we do not support consecutive assistant message input.")

    def delete_session_by_id(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id)
        if session is None:
            return False
        return True

    def add_record(self, session_id: str, turn: Turn) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found.")
        session.insert_new_turn(turn)
        return True
