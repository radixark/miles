from copy import copy

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.rollout.inference_rollout.inference_rollout_train import generate_rollout_async

from .prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill


class VideoInferenceRolloutFn(InferenceRolloutFn):
    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.prefill_args = input.args
        self.state.args = copy(input.args)
        self.state.args.recompute_logprobs_via_prefill = False

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output, aborted_samples = await generate_rollout_async(
            self.state,
            input.rollout_id,
            self.data_source.get_samples,
        )
        await recompute_samples_rollout_logprobs_via_prefill(
            self.prefill_args,
            [sample for group in output.samples for sample in group],
            url=f"http://{self.state.args.sglang_router_ip}:{self.state.args.sglang_router_port}/generate",
            sampling_params=self.state.sampling_params,
            tokenizer=self.state.tokenizer,
        )
        self.data_source.add_samples(aborted_samples)
        return output
