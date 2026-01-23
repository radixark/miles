from __future__ import annotations

from argparse import Namespace
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.modular_rollout.orchestration_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return True


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


# TODO: may add add_arguments
# TODO: may add save/load if need it to be stateful
# Duck typing, users do not need to extend this class
@runtime_checkable
class RolloutFnProtocol(Protocol):
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput | Awaitable[RolloutFnOutput]: ...


# TODO maybe put to modular_rollout folder depending on overall folder structure
@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


# TODO: may add add_arguments
# TODO: may add save/load if need it to be stateful
@runtime_checkable
class GenerateFnProtocol(Protocol):
    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput: ...


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    """Legacy rollout function call interface. Used when MILES_EXPERIMENTAL_ROLLOUT_REFACTOR is disabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)


# TODO: may add add_arguments
# TODO: may add save/load if need it to be stateful
# Duck typing, users do not need to extend this class
@runtime_checkable
class RolloutFnProtocol(Protocol):
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput | Awaitable[RolloutFnOutput]: ...


# TODO maybe put to modular_rollout folder depending on overall folder structure
@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


# TODO: may add add_arguments
# TODO: may add save/load if need it to be stateful
@runtime_checkable
class GenerateFnProtocol(Protocol):
    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput: ...
