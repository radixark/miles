import uuid
from typing import Any

from pydantic import BaseModel, Field
from transformers import AutoTokenizer


class TokenInfo(BaseModel):
    token_ids: list[int] = Field(default_factory=list)
    log_probs: list[float] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)

    def remove_tokens(self, start_index: int, end_index: int):
        self.token_ids = self.token_ids[start_index:end_index]
        self.log_probs = self.log_probs[start_index:end_index]
        self.loss_mask = self.loss_mask[start_index:end_index]

    def insert_tokens(self, token_ids: list[int], log_probs: list[float], loss_mask: list[int]):
        self.token_ids.extend(token_ids)
        self.log_probs.extend(log_probs)
        self.loss_mask.extend(loss_mask)

    def append(self, token_id: int, log_prob: float, loss_mask: int):
        self.token_ids.append(token_id)
        self.log_probs.append(log_prob)
        self.loss_mask.append(loss_mask)

    def __add__(self, other: "TokenInfo") -> "TokenInfo":
        return TokenInfo(
            token_ids=self.token_ids + other.token_ids,
            log_probs=self.log_probs + other.log_probs,
            loss_mask=self.loss_mask + other.loss_mask,
        )


class NaiveTrajectory(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    token_info: TokenInfo = Field(default_factory=TokenInfo)

    def get_token_info(self) -> TokenInfo:
        return self.token_info

    def update(self, messages: list[dict[str, Any]], token_info: TokenInfo):
        self.messages = messages
        self.token_info = token_info


class NaiveTrajectoryManager:
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.sessions: dict[str, NaiveTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = NaiveTrajectory()
        return session_id

    def get_token_info_by_id(self, session_id: str) -> TokenInfo | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return session.get_token_info()

    def calc_prompt_tokens(self, session_id: str, messages: list[dict[str, Any]]) -> TokenInfo:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found.")
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        token_info = TokenInfo(
            token_ids=token_ids,
            log_probs=[0.0] * len(token_ids),
            loss_mask=[0] * len(token_ids),
        )
        return token_info

    def delete_session_by_id(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return False
        return True

    def update_record(self, session_id: str, messages: list[dict[str, Any]], token_info: TokenInfo) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found.")
        session.update(messages, token_info)
        return True
