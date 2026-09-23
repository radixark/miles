from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, field_validator

from miles.utils.pydantic_utils import StrictBaseModel


class CreateSessionRequest(StrictBaseModel):
    evaluation: StrictBool = False
    # the caller resolves rollout/eval/dataset values; the session only fills fields a request omits
    temperature: StrictFloat | None = None
    top_p: StrictFloat | None = None
    top_k: StrictInt | None = None

    @field_validator("top_k", mode="before")
    @classmethod
    def _integral_float_as_int(cls, value):
        # an eval dataset YAML may spell top_k as 40.0; a fractional top_k stays an error
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


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
