import asyncio

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.modular_rollout.orchestration_common import GenerateState
from miles.rollout.modular_rollout.orchestration_eval import eval_rollout_single_dataset
from miles.rollout.modular_rollout.orchestration_train import generate_rollout_async


# TODO may move `orchestration_*`
class SimpleRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.data_source = input.data_source
        self.prompt_dataset_cache = {}
        self.state = GenerateState(input.args)

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._exec_eval(input)
        else:
            return await self._exec_train(input)

    async def _exec_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output, aborted_samples = await generate_rollout_async(
            self.state, input.rollout_id, self.data_source.get_samples
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _exec_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"

        coros = []
        for dataset_cfg in getattr(self.state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.state, dataset_cfg, self.prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
