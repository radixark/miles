"""
Simple agentic demo with tool calling.
"""

import argparse
from dataclasses import dataclass

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    endpoint_tracer = TODO()
    agent = _BlackboxToolCallAgent(
        base_url=endpoint_tracer.base_url,
        **{k: v for k, v in vars(input.args).items() if k.startswith("generate_")},
    )
    await agent.run()
    return endpoint_tracer.collect()


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
    max_turns: int
    tool_specs_path: str
    tool_call_parser: str
    execute_tool_function_path: str

    async def run(self):
        TODO
