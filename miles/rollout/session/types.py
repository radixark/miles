from pydantic import BaseModel, Field, StrictBool

from miles.utils.pydantic_utils import StrictBaseModel


class CreateSessionRequest(StrictBaseModel):
    evaluation: StrictBool = False


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
