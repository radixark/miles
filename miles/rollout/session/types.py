from typing import Any

from pydantic import BaseModel, Field, model_serializer


class SessionRecord(BaseModel):
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int
    replayed_messages: list[dict] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_replay_audit(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.replayed_messages is None:
            data.pop("replayed_messages", None)
        return data


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
