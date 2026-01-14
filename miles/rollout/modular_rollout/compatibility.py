from miles.rollout.base_types import RolloutFnInput, RolloutFnOutput, RolloutFnConstructorInput, RolloutFnTrainOutput, \
    RolloutFnEvalOutput


class LegacyRolloutFnAdapter:
    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.fn = TODO

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        output = self.fn(self.args, input.rollout_id, self.data_source, evaluation=input.evaluation)

        # compatibility for legacy version
        if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
            output = RolloutFnEvalOutput(data=output) if input.evaluation else RolloutFnTrainOutput(samples=output)

        return output
