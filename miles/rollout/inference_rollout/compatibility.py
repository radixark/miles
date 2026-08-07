import asyncio
import inspect
from collections.abc import Callable

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.utils.async_utils import get_async_loop, run
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


def load_rollout_function(input: RolloutFnConstructorInput, path: str):
    fn = load_function(path)

    if inspect.isclass(fn):
        return fn(input)
    else:
        return LegacyRolloutFnAdapter(input, fn)


def call_rollout_function(fn, input: RolloutFnInput) -> RolloutFnOutput:
    output = fn(input)

    if inspect.iscoroutine(output):
        output = run(output)

    return output


async def maybe_close(fn) -> None:
    """Release a rollout fn's resources via its optional aclose/close hook.
    Plain legacy functions have neither hook and are skipped."""
    if (aclose := getattr(fn, "aclose", None)) is not None:
        # The fn's tasks live on the shared rollout loop (``run``); awaiting
        # them from another loop raises, so aclose must execute there.
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(aclose(), get_async_loop().loop))
    elif (close := getattr(fn, "close", None)) is not None:
        close()


class LegacyGenerateFnAdapter:
    def __init__(self, fn: Callable):
        self.fn = fn
        self._has_evaluation_param = "evaluation" in inspect.signature(fn).parameters

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        if self._has_evaluation_param:
            output = await self.fn(input.args, input.sample, input.sampling_params, evaluation=input.evaluation)
        else:
            output = await self.fn(input.args, input.sample, input.sampling_params)

        if not isinstance(output, GenerateFnOutput):
            output = GenerateFnOutput(samples=output)

        return output


def load_generate_function(path: str):
    fn = load_function(path)
    if fn is None:
        return None

    if inspect.isclass(fn):
        return fn()
    elif _is_legacy_generate_fn(fn):
        return LegacyGenerateFnAdapter(fn)
    else:
        return fn


def _is_legacy_generate_fn(fn: Callable) -> bool:
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return len(params) >= 3 and params[0] != "input"
