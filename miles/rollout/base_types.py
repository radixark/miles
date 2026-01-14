from argparse import Namespace
from dataclasses import dataclass
from typing import Any

from miles.utils.types import Sample


@dataclass
class RolloutFnBaseInput:
    args: Namespace
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


@dataclass
class RolloutFnTrainInput(RolloutFnBaseInput):
    @property
    def evaluation(self):
        return False


@dataclass
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


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
