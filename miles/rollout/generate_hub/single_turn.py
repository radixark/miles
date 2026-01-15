from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.sglang_generate_wrapper import (
    compute_prompt_ids,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate using traditional SGLang router with token-based workflow"""
    args = input.args
    sample = input.sample

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = await compute_prompt_ids(sample, input.state)
    payload = await compute_request_payload(args, prompt_ids, sample, input.sampling_params)

    if payload["sampling_params"]["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return GenerateFnOutput(samples=sample)

    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = prompt_ids

    output = await post(url, payload)

    await update_sample_from_response(args, sample, output)

    return GenerateFnOutput(samples=sample)
