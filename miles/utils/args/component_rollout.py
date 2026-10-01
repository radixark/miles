from typing import ClassVar

from pydantic import ConfigDict, Field

from miles.utils.args.schema import BaseConfig
from miles.utils.pydantic_utils import StrictBaseModel


class InferenceRuntimeMutState(StrictBaseModel):
    engine_count: int = 0
    gpu_count: int = 0
    eval_engine_count: int = 0

    def set_(self, other: "InferenceRuntimeMutState") -> None:
        self.engine_count = other.engine_count
        self.gpu_count = other.gpu_count
        self.eval_engine_count = other.eval_engine_count


class InferenceRuntimeImmutState(InferenceRuntimeMutState):
    model_config = ConfigDict(frozen=True)


class RolloutOnlyConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"inference_runtime_mut_state"})

    inference_runtime_mut_state: InferenceRuntimeMutState = Field(default_factory=InferenceRuntimeMutState)


class InferenceControllerOnlyConfig(BaseConfig):
    pass
