from pydantic import BaseModel, Field


class SessionRecord(BaseModel):
    timestamp: float  # stamped after the backend responds (= call END)
    # arrival stamp: bounds the agent-side gap between chat calls at the
    # server edge (tool time sampling); optional for wire compatibility
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
