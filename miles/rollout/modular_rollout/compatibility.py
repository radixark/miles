import inspect
from collections.abc import Callable

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnProtocol,
    RolloutFnTrainOutput,
)
from miles.utils.async_utils import run
from miles.utils.misc import load_function


class LegacyRolloutFnAdapter:
    def __init__(self, input: RolloutFnConstructorInput, fn: Callable):
        self.args = input.args
        self.data_source = input.data_source
        self.fn = fn

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        output = self.fn(self.args, input.rollout_id, self.data_source, evaluation=input.evaluation)

        # compatibility for legacy version
        if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
            output = RolloutFnEvalOutput(data=output) if input.evaluation else RolloutFnTrainOutput(samples=output)

        return output


assert issubclass(LegacyRolloutFnAdapter, RolloutFnProtocol)


def load_rollout_function(input: RolloutFnConstructorInput, path: str):
    fn = load_function(path)

    if inspect.isclass(fn):
        return fn(input)
    else:
        return LegacyRolloutFnAdapter(input, fn)


def call_rollout_function(fn: RolloutFnProtocol, input: RolloutFnInput) -> RolloutFnOutput:
    output = fn(input)

    if inspect.iscoroutine(output):
        output = run(output)

    return output
