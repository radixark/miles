from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
