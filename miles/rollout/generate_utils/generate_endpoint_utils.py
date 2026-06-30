"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # TODO shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    if getattr(args, "keep_sampling_mask", False):
        payload["return_sampling_mask"] = True
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload, None


def extract_response_tokens_and_log_probs(args, meta_info: dict) -> tuple[list[int], list[float]]:
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if output_token_logprobs:
        response_tokens = [int(item[1]) for item in output_token_logprobs]
    else:
        response_tokens = []

    if getattr(args, "keep_sampling_mask", False):
        sampling_logprobs = meta_info.get("output_token_sampling_logprobs")
        if not isinstance(sampling_logprobs, list):
            raise ValueError("SGLang response missing output_token_sampling_logprobs")
        if len(sampling_logprobs) != len(response_tokens):
            raise ValueError(
                "SGLang output_token_sampling_logprobs length "
                f"{len(sampling_logprobs)} != response token length {len(response_tokens)}"
            )
        if any(logprob is None for logprob in sampling_logprobs):
            raise ValueError("SGLang response contained None output_token_sampling_logprobs")
        return response_tokens, [float(logprob) for logprob in sampling_logprobs]

    if output_token_logprobs:
        return response_tokens, [float(item[0]) for item in output_token_logprobs]
    return [], []


def extract_response_sampling_masks(args, meta_info: dict, response_length: int) -> list[list[int]] | None:
    if not getattr(args, "keep_sampling_mask", False):
        return None

    sampling_masks = meta_info.get("output_token_sampling_mask")
    if not isinstance(sampling_masks, list):
        raise ValueError("SGLang response missing output_token_sampling_mask")
    if len(sampling_masks) != response_length:
        raise ValueError(
            "SGLang output_token_sampling_mask length "
            f"{len(sampling_masks)} != response token length {response_length}"
        )
    if any(mask is None for mask in sampling_masks):
        raise ValueError("SGLang response contained None output_token_sampling_mask")
    return [[int(token_id) for token_id in mask] for mask in sampling_masks]


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    meta_info = output["meta_info"]
    new_response_tokens, new_response_log_probs = extract_response_tokens_and_log_probs(args, meta_info)

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs
    new_sampling_masks = extract_response_sampling_masks(args, meta_info, len(new_response_tokens))
    if new_sampling_masks is not None:
        if sample.rollout_sampling_masks is None:
            sample.rollout_sampling_masks = []
        sample.rollout_sampling_masks += new_sampling_masks

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    # TODO handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, output, sample)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, meta_info)


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, sample):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    return _decode_topk_buffer(info, len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)


def get_indexer_topk_from_response(args, output, sample):
    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    return _decode_topk_buffer(info, len(sample.tokens) - 1, num_layers, -1)
