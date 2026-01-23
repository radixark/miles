"""
Simple single-turn generation.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.generate_endpoint_wrapper import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = await compute_prompt_ids_from_sample(input.state, sample)
    payload, halt_status = await compute_request_payload(input.state, sample, prompt_ids, input.sampling_params)

    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload)

    await update_sample_from_response(args, sample, payload=payload, output=output)

    return GenerateFnOutput(samples=sample)
