"""
Simple agentic demo with tool calling.
"""

import argparse
from copy import deepcopy
from typing import Any

from openai import AsyncOpenAI

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.generate_utils.tool_call_utils import execute_tool_calls
from miles.utils.misc import load_function


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    await _run_blackbox_tool_call_agent(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        max_turns=input.args.generate_max_turns,
        tool_specs_path=input.args.generate_tool_specs_path,
        execute_tool_function_path=input.args.generate_execute_tool_function_path,
    )

    records = await tracer.collect_records()
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


async def _run_blackbox_tool_call_agent(
    base_url: str,
    prompt: list[dict[str, Any]],
    max_turns: int,
    tool_specs_path: str,
    execute_tool_function_path: str,
):
    """
    Imagine this is a black-box agent, e.g. SWE-agent, which does arbitrarily complex work,
    only understands OpenAI compatible API, and never understands Miles or the Sample data structure.
    """

    # ----------------------- Setup -------------------------

    client = AsyncOpenAI(base_url=base_url, api_key="empty")
    execute_tool_function = load_function(execute_tool_function_path)
    tool_specs = load_function(tool_specs_path)

    # ----------------------- Initial prompts -------------------------

    messages = deepcopy(prompt)

    for _turn in range(max_turns):
        # ----------------------- Call inference endpoint -------------------------

        response = await client.chat.completions.create(model="default", messages=messages, tools=tool_specs)

        choice = response.choices[0]
        messages.append(choice.message.model_dump())

        if choice.finish_reason in ("stop", "length"):
            break

        # ----------------------- Execute tools -------------------------

        if x := choice.message.tool_calls:
            messages += await execute_tool_calls(x, execute_tool_function)
