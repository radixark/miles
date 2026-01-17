"""
Simple multi-turn generation with tool calling.
"""

import argparse
import json
import uuid

from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.tool_call_utils import tokenize_tool_responses
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
        tools=(TypeAdapter(list[Tool]).validate_python(tool_specs)),
        tool_call_parser=args.generate_tool_call_parser,
    )

    prompt = sample.prompt
    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, tools=tool_specs)
    prompt_tokens_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    response = ""
    response_token_ids = []
    loss_masks = []

    for turn in range(args.generate_max_turns):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_context_length = args.rollout_max_context_len
        else:
            max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": input.sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return GenerateFnOutput(samples=sample)

        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response = tokenizer.decode(cur_response_token_ids)
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += cur_log_probs

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        _, parsed_tool_calls = tool_call_parser.parse_non_stream(cur_response)
        if len(parsed_tool_calls) == 0:
            break

        tool_messages = await _execute_tool_calls(parsed_tool_calls, execute_tool_function)

        next_obs_tokens_ids: list[int] = tokenize_tool_responses(tool_messages, tokenizer=tokenizer)
        # TODO is this ok?
        response += tokenizer.decode(next_obs_tokens_ids)
        response_token_ids += next_obs_tokens_ids
        loss_masks += [0] * len(next_obs_tokens_ids)

        sample.rollout_log_probs += [0.0] * len(next_obs_tokens_ids)

        assert len(response_token_ids) == len(
            sample.rollout_log_probs
        ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if turn >= args.generate_max_tool_calls:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Set status
    sample.update_from_meta_info(args, output["meta_info"])

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-max-tool-calls", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)


generate.add_arguments = _add_arguments


async def _execute_tool_calls(parsed_tool_calls, execute_one) -> list[dict]:
    tool_messages = []
    for call in parsed_tool_calls:
        params = json.loads(call.parameters) if call.parameters else {}
        result = await execute_one(call.name, params)
        assert isinstance(result, str)
        tool_messages.append(
            {
                "role": "tool",
                # src: serving_chat.py :: _process_tool_call_id
                "tool_call_id": f"call_{uuid.uuid4().hex[:24]}",
                "content": result,
                "name": call.name,
            }
        )
    return tool_messages
