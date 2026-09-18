from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrozenPartialBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, from_attributes=True, populate_by_name=True)


class FrozenOpenBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
