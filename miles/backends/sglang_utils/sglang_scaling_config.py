import pydantic

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ServerGroupScalingConfig(FrozenStrictBaseModel):
    num_gpus: int = pydantic.Field(gt=0)
    gpu_offset: int = pydantic.Field(ge=0)
    engine_offset: int = pydantic.Field(ge=0)


class SglangScalingConfig(FrozenStrictBaseModel):
    groups: dict[str, list[ServerGroupScalingConfig]]

    def group(self, *, model_name: str, group_index: int) -> ServerGroupScalingConfig:
        return self.groups[model_name][group_index]
