"""
Simple multi-turn generation with tool calling.
"""

import argparse

from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.generate_endpoint_wrapper import compute_request_payload, update_sample_from_response
from miles.rollout.generate_hub.tool_call_utils import execute_tool_calls, update_sample_with_tool_responses
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    tokenizer = input.state.tokenizer

    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    assert isinstance(tool_specs, list)

    tool_call_parser = FunctionCallParser(
        tools=TypeAdapter(list[Tool]).validate_python(tool_specs),
        tool_call_parser=args.generate_tool_call_parser,
    )

    prompt = sample.prompt
    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, tools=tool_specs)
    prompt_tokens_ids = tokenizer.encode(prompt, add_special_tokens=False)

    assert sample.tokens == []
    assert sample.response == ""
    assert sample.response_length == 0
    assert sample.loss_mask is None
    sample.loss_mask = []
    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        # TODO handle separately
        # Check if total length exceeds max context length
        total_length = len(sample.tokens)
        if args.rollout_max_context_len is not None:
            max_context_length = args.rollout_max_context_len
        else:
            max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # ----------------------- Call inference endpoint -------------------------

        payload = compute_request_payload(
            args,
            input_ids=sample.tokens,
            sampling_params=input.sampling_params,
        )

        output = await post(url, payload)

        await update_sample_from_response(args, sample, payload=payload, output=output)

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        # ----------------------- Execute tools -------------------------

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)


generate.add_arguments = _add_arguments
