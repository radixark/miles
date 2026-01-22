"""
Simple agentic demo with tool calling.
"""

import argparse
from collections.abc import Callable


from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.misc import load_function


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"
    await custom_agent_function(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        sampling_params=input.sampling_params,
    )

    records = await tracer.collect_records()
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)


generate.add_arguments = _add_arguments
