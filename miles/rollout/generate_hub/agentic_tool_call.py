"""
Simple agentic demo with tool calling.
"""

import argparse

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    TODO


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments


class _BlackboxToolCallAgent:
    """
    Imagine this is a black-box agent, e.g. SWE-agent, which does arbitrarily complex work,
    only understands OpenAI compatible API, and never understands Miles or the Sample data structure.
    """
    async def run(self):
        TODO
