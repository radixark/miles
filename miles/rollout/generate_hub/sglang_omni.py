"""Single-turn rollout using SGLang Omni's processed multimodal contract.

Select with ``--custom-generate-function-path
miles.rollout.generate_hub.sglang_omni.generate``.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    multimodal_route_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    }, f"{sample.status=}"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
    has_multimodal_inputs = sample.multimodal_inputs and any(
        value is not None for value in sample.multimodal_inputs.values()
    )
    if has_multimodal_inputs and sample.multimodal_train_inputs is None:
        raise ValueError(
            "SGLang Omni multimodal rollout requires processor-produced " "sample.multimodal_train_inputs"
        )

    if sample.response:
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
        multimodal_train_inputs=sample.multimodal_train_inputs,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    headers = compute_routing_headers(args, sample) or {}
    headers.update(multimodal_route_headers(payload.get("multimodal_train_inputs")) or {})
    output = await post(url, payload, headers=headers or None)
    await update_sample_from_response(
        args,
        sample,
        payload=payload,
        output=output,
    )
    return GenerateFnOutput(samples=sample)
