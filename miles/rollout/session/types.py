from pydantic import BaseModel, Field

SESSION_GENERATION_HEADER = "x-miles-session-generation"
SESSION_RECORD_ERROR_CODE = "session_record_unavailable"


class SessionRecord(BaseModel):
    timestamp: float
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
