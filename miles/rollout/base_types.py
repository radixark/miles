from argparse import Namespace, ArgumentParser
from dataclasses import dataclass
from typing import Any, Protocol

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
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


class RolloutFnProtocol(Protocol):
    def __init__(self, input: RolloutFnConstructorInput):
        ...

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        ...

    def __call__(self, input: RolloutFnTrainInput | RolloutFnEvalInput) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
        ...


# TODO move / refactor
def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
