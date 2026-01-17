"""
Simple agentic demo with tool calling.
"""

import argparse
from dataclasses import dataclass

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.oai_endpoint_wrapper import OpenAIEndpointTracer


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = OpenAIEndpointTracer()

    agent = _BlackboxToolCallAgent(
        base_url=tracer.base_url,
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
    generate_max_turns: int
    generate_tool_specs_path: str
    generate_tool_call_parser: str
    generate_execute_tool_function_path: str
    generate_multi_samples: bool

    async def run(self):
        TODO
