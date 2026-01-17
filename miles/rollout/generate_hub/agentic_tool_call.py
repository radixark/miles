"""
Simple agentic demo with tool calling.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

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
    generate_execute_tool_function_path: str

    async def run(self):
        # ----------------------- Setup -------------------------

        client = AsyncOpenAI(base_url=self.base_url, api_key="empty")
        execute_tool_function = load_function(self.generate_execute_tool_function_path)
        tool_specs = load_function(self.generate_tool_specs_path)

        # ----------------------- Initial prompts -------------------------

        messages = deepcopy(self.prompt)

        for turn in range(self.generate_max_turns):
            # ----------------------- Call inference endpoint -------------------------

            response = await client.chat.completions.create(model="default", messages=messages, tools=tool_specs)

            choice = response.choices[0]
            messages.append(choice.message.model_dump())

            if choice.finish_reason in ("stop", "length"):
                break

            # ----------------------- Execute tools -------------------------

            if x := choice.message.tool_calls:
                messages += await execute_tool_calls(x, execute_tool_function)
