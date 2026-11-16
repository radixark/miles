import copy

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = (await single_turn_generate(input)).samples
    return GenerateFnOutput(samples=_split(sample, num_pieces=2))


async def generate_one_to_three(input: GenerateFnInput) -> GenerateFnOutput:
    sample = await sglang_rollout.generate(
        input.args, input.sample, input.sampling_params, evaluation=input.evaluation
    )
    return GenerateFnOutput(samples=_split(sample, num_pieces=1 + sample.index % 3))


async def generate_two(input: GenerateFnInput) -> GenerateFnOutput:
    sample = await sglang_rollout.generate(
        input.args, input.sample, input.sampling_params, evaluation=input.evaluation
    )
    return GenerateFnOutput(samples=_split(sample, num_pieces=2))


def _split(sample: Sample, *, num_pieces: int) -> list[Sample]:
    assert isinstance(sample, Sample)

    sample.rollout_id = sample.index
    num_pieces = min(num_pieces, sample.response_length)
    if num_pieces < 2 or sample.status == Sample.Status.ABORTED:
        return [sample]

    prompt_len = len(sample.tokens) - sample.response_length
    loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
    cuts = [sample.response_length * k // num_pieces for k in range(num_pieces + 1)]
    pieces = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        piece = copy.deepcopy(sample)
        piece.tokens = sample.tokens[: prompt_len + end]
        piece.response_length = end
        piece.loss_mask = [0] * start + list(loss_mask[start:end])
        if sample.rollout_log_probs is not None:
            piece.rollout_log_probs = sample.rollout_log_probs[:end]
        if end < sample.response_length:
            piece.status = Sample.Status.COMPLETED
        pieces.append(piece)
    return pieces
