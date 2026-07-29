"""Example compact generate fn: one rollout execution -> multiple training samples.

Wraps single-turn generation, then splits the response into two training
samples at the midpoint of the response tokens. Both siblings share the parent
sample's ``rollout_id`` so the by-rollout scheduler keeps them in one training
step and the per-rollout loss reducer counts the rollout once.

This mirrors the compact / subagent pattern (multi-agent systems, thinking-token
removal) in a form small enough for e2e validation; see
``--custom-generate-function-path miles.rollout.generate_hub.compact_split_example.generate``.
"""

import copy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    output = await single_turn_generate(input)
    sample = output.samples
    assert isinstance(sample, Sample)

    # Too short to split, or aborted mid-generation: emit as-is (still stamp
    # rollout_id so downstream accounting is uniform).
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
    # The reward belongs to the full rollout; both siblings carry it so
    # advantage computation sees the rollout's outcome on every piece.
    first.status = Sample.Status.COMPLETED

    second = copy.deepcopy(sample)
    second.loss_mask = [0] * cut + list(sample.loss_mask[cut:]) if sample.loss_mask is not None else None

    return GenerateFnOutput(samples=[first, second])
