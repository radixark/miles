from dataclasses import dataclass
from typing import Any

from miles.rollout.base_types import BaseRolloutFn, GenerateFnInput, GenerateFnOutput, RolloutFnInput, RolloutFnOutput


class _HookBoundaryReached(Exception):
    pass


@dataclass
class _RecordingSource:
    args: Any


class _RecordingRollout(BaseRolloutFn):
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        raise _HookBoundaryReached


async def _generate(input: GenerateFnInput) -> GenerateFnOutput:
    return GenerateFnOutput(samples=input.sample)
