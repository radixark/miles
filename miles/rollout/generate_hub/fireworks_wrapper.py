"""
Simple agentic demo with tool calling.
"""

import argparse
from typing import Any

from eval_protocol import InitRequest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import (
    OpenAIEndpointTracer,
    compute_samples_from_openai_records,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.utils.http_utils import post


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    tracer = await OpenAIEndpointTracer.create(input.args)

    init_url = f"{input.args.agent_base_url}/init"
    init_request = InitRequest(
        model_base_url=tracer.base_url,
        completion_params=build_chat_completion_params(input.sampling_params),
        messages=input.sample.prompt,
        tools=input.sample.metadata.get("tools", None),
    )
    resp = await post(init_url, init_request.model_dump(exclude_none=True))
    if resp["status"] != "success":
        raise ValueError(f"Failed to initialize agent: {resp['error']}")

    records = await tracer.collect_records()
    samples = compute_samples_from_openai_records(input.sample, records, input.state.tokenizer)
    if not input.args.generate_multi_samples:
        samples = merge_samples(samples, input.state.tokenizer)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--agent-base-url", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true", default=False)


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_completion_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    completion_params = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in completion_params:
            if dst not in completion_params:
                completion_params[dst] = completion_params[src]
            completion_params.pop(src, None)

    # Notice: Here we force the inference backend to return token information and start from 0
    # The start len should be 0 to make sure prompt token ids and be correctly returned from SGLang.
    completion_params["logprobs"] = True
    completion_params["logprob_start_len"] = 0

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in completion_params.items() if key in allowed_keys and value is not None}
