from typing import Any

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt

# Chat request fields a session fills when a request omits them.
SESSION_SAMPLING_FIELDS = ("temperature", "top_p", "top_k")

from miles.utils.pydantic_utils import StrictBaseModel


class CreateSessionRequest(StrictBaseModel):
    evaluation: StrictBool = False
    # the caller resolves rollout/eval/dataset values; the session only fills fields a request omits
    temperature: StrictFloat | None = None
    top_p: StrictFloat | None = None
    top_k: StrictInt | None = None

    def sampling_defaults(self) -> dict[str, Any]:
        """The sampling fields this request provides, keyed like a chat completion request."""
        return {key: value for key in SESSION_SAMPLING_FIELDS if (value := getattr(self, key)) is not None}


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
