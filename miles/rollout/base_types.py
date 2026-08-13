from __future__ import annotations

import abc
from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


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
    # engine weight version, None before the first weight update
    weight_version: int | None = None
    # which policy model asked for this rollout; None when the run trains one policy
    trainer_model_id: str | None = None

    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

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


class BaseRolloutFn(abc.ABC):
    def __init__(self, input: RolloutFnConstructorInput) -> None:
        self.constructor_input = input

    @abc.abstractmethod
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        raise NotImplementedError

    def save(self, rollout_id: int) -> None:
        return None

    def load(self, rollout_id: int | None) -> None:
        return None


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


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    """Legacy rollout function call interface. Used when MILES_USE_LEGACY_ROLLOUT_V1 is enabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
