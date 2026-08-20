from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class FrozenStrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")


class FrozenPartialBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore", frozen=True, from_attributes=True, populate_by_name=True, ser_json_inf_nan="constants"
    )


class FrozenOpenBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, ser_json_inf_nan="constants")
