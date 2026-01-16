"""
Simple multi-turn generation with tool calling.
"""

import argparse

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    tokenizer = input.state.tokenizer

    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.execute_tool_function_path)

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = load_function(args.generate_tool_specs_path)
    assert isinstance(tool_specs, list)
    prompt = tokenizer.apply_chat_template(sample.prompt, tokenize=False, add_generation_prompt=True, tools=tool_specs)

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

        # TODO decide execute_tool_function API
        out = await execute_tool_function(TODO)
        next_obs, done = out["next_obs"], out["done"]

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

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
    parser.add_argument("--generate-execute-tool-function-path", type=str)


generate.add_arguments = _add_arguments
