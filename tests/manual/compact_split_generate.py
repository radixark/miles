"""E2E fixture: compact generate fn splitting one rollout execution into two
training samples that share the parent's rollout_id. Use via
--custom-generate-function-path tests.manual.compact_split_generate.generate"""

import copy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    output = await single_turn_generate(input)
    sample = output.samples
    assert isinstance(sample, Sample)

    sample.rollout_id = sample.index
    if sample.response_length < 2 or sample.status == Sample.Status.ABORTED:
        return GenerateFnOutput(samples=[sample])

    prompt_len = len(sample.tokens) - sample.response_length
    cut = sample.response_length // 2

    first = copy.deepcopy(sample)
    first.tokens = sample.tokens[: prompt_len + cut]
    first.response_length = cut
    first.loss_mask = sample.loss_mask[:cut] if sample.loss_mask is not None else None
    if first.rollout_log_probs is not None:
        first.rollout_log_probs = sample.rollout_log_probs[:cut]
    first.status = Sample.Status.COMPLETED

    second = copy.deepcopy(sample)
    second.loss_mask = [0] * cut + list(sample.loss_mask[cut:]) if sample.loss_mask is not None else None

    return GenerateFnOutput(samples=[first, second])
