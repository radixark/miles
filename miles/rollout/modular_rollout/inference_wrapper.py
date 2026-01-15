import numpy as np
import pybase64

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate using traditional SGLang router with token-based workflow"""
    state = input.state
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

    prompt_ids = await _compute_prompt_ids(sample, state)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert sampling_params["max_new_tokens"] >= 0
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return GenerateFnOutput(samples=sample)

    # Prepare payload for sglang server
    payload = {
        # Use existing tokens for multi-turn or tokenize the new prompt
        "input_ids": sample.tokens if len(sample.response) > 0 else prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
    }

    if image_data := (sample.multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and (not sample.tokens):
        sample.tokens = prompt_ids

    output = await post(url, payload)

    await _fill_sample_by_response(args, sample, output)

    return GenerateFnOutput(samples=sample)


async def _compute_prompt_ids(sample, state):
    if state.processor:
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        # TODO shall we put it here?
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    return prompt_ids


async def _fill_sample_by_response(args, sample, output):
    if args.use_miles_router and "RadixTreeMiddleware" in args.miles_router_middleware_paths:
        from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if x := output["meta_info"].get("output_token_logprobs"):
            new_response_tokens = [item[1] for item in x]
            new_response_log_probs = [item[0] for item in x]
        else:
            new_response_tokens, new_response_log_probs = [], []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    sample.rollout_routed_experts = _get_rollout_routed_experts_from_response(args, sample, output)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _get_rollout_routed_experts_from_response(args, sample, output):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None

    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    x = x.reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
    return x
