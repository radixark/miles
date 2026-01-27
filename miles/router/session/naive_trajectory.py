import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field

from miles.router.session.session_types import SessionRecord


class NaiveTrajectory(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    records: list[SessionRecord] = Field(default_factory=list)

    def append_session_record(self, record: SessionRecord):
        self.records.append(record)


# This is only a naive trajectory manager to store history session record.
# Cross turn token input not implemented.
class NaiveTrajectoryManager:
    def __init__(self, args, tokenizer: Any):
        self.sessions: dict[str, NaiveTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()

    def create_session(self) -> str:
        with self._lock:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = NaiveTrajectory()
            return session_id

    def get_session_records_by_id(self, session_id: str) -> list[SessionRecord] | None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            return session.records

    def calc_prompt_tokens(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> list[int] | None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_special_tokens=False,
                add_generation_prompt=True,
            )
            return token_ids

    def delete_session_by_id(self, session_id: str) -> bool | None:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                return None
            return True

    def append_session_record(self, session_id: str, record: SessionRecord) -> bool | None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            session.append_session_record(record)
            return True
