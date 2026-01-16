from argparse import Namespace
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample


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


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


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
