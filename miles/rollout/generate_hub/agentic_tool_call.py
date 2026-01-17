"""
Simple agentic demo with tool calling.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.oai_endpoint_wrapper import OpenAIEndpointTracer
from miles.rollout.generate_hub.tool_call_utils import execute_tool_calls
from miles.utils.misc import load_function


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = OpenAIEndpointTracer()

    agent = _BlackboxToolCallAgent(
        base_url=tracer.base_url,
        prompt=input.sample.prompt,
        **{k: v for k, v in vars(input.args).items() if k.startswith("generate_")},
    )
    await agent.run()

    return tracer.collect()


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


@dataclass
class _BlackboxToolCallAgent:
    """
    Imagine this is a black-box agent, e.g. SWE-agent, which does arbitrarily complex work,
    only understands OpenAI compatible API, and never understands Miles or the Sample data structure.
    """

    base_url: str
    prompt: list[dict[str, Any]]
    generate_max_turns: int
    generate_tool_specs_path: str
    generate_tool_call_parser: str
    generate_execute_tool_function_path: str
    generate_multi_samples: bool

    async def run(self):
        execute_tool_function = load_function(self.generate_execute_tool_function_path)
        tool_specs = load_function(self.generate_tool_specs_path)

        messages = deepcopy(self.prompt)

        for turn in range(self.generate_max_turns):
            # ----------------------- Call inference endpoint -------------------------

            output = await post(url, payload)
            await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

            if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
                break

            # ----------------------- Execute tools -------------------------

            messages += await execute_tool_calls(tool_calls, execute_tool_function)
