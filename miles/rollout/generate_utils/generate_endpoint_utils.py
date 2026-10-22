"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata, should_return_sampling_mask
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.media_refs import has_media, load_media_for_processor, rollout_media_payload
from miles.utils.processing_utils import extract_multimodal_train_inputs
from miles.utils.types import Sample


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    # In token space SGLang expands the media placeholders and the trainer builds the processor tensors.
    if state.processor and has_media(sample.multimodal_inputs) and not state.args.multimodal_token_space_processing:
        processor_output = state.processor(text=prompt, **load_media_for_processor(sample.multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]

        # TODO shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def policy_uses_routing_key(args) -> bool:
    return args.sglang_router_policy in ("consistent_hashing", "manual")


def compute_routing_headers(args, sample: Sample) -> dict[str, str] | None:
    if policy_uses_routing_key(args) and not sample.routing_key:
        raise ValueError(
            f"router policy {args.sglang_router_policy} routes by X-SMG-Routing-Key, "
            f"but sample (index={sample.index}) has no routing_key set"
        )
    if sample.routing_key:
        return {"X-SMG-Routing-Key": sample.routing_key}
    return None


def token_space_request_fields(mm_token_expansion_start_len: int) -> dict[str, Any]:
    """Ask SGLang to expand media placeholders after the already expanded prefix and return the prompt IDs."""
    # The request carries every media reference; SGLang matches the placeholders after the prefix
    # to the trailing references, so a resumed request with a fully expanded prefix expands nothing.
    return {"mm_token_expansion_start_len": mm_token_expansion_start_len, "return_prompt_token_ids": True}


def adopt_expanded_prompt_ids(sample: Sample, payload: dict, output: dict) -> None:
    """Replace the request's prompt IDs with the ones SGLang expanded in token space.

    Args:
        sample: The sample the request was built from; its tokens, and on a later turn its response fields, grow.
        payload: The request sent to SGLang; nothing happens unless it asked for ``return_prompt_token_ids``.
        output: SGLang's response, carrying the expanded ``prompt_token_ids``.
    """
    if not payload.get("return_prompt_token_ids"):
        return
    prompt_token_ids = output["prompt_token_ids"]
    added_prompt_tokens = len(prompt_token_ids) - len(sample.tokens)
    if sample.response_length:
        # Expansion after the first response belongs to the zero-loss input suffix of a later turn.
        sample.response_length += added_prompt_tokens
        if sample.loss_mask is not None:
            sample.loss_mask += [0] * added_prompt_tokens
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * added_prompt_tokens
    sample.tokens = list(prompt_token_ids)


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    *,
    evaluation: bool = False,
    kv_cache_namespace: str | None = None,
    mm_token_expansion_start_len: int = 0,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    return_sampling_mask = should_return_sampling_mask(args, sampling_params, evaluation=evaluation)

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    if return_sampling_mask:
        payload["return_sampling_mask"] = True
    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    payload.update(rollout_media_payload(multimodal_inputs))
    if has_media(multimodal_inputs) and args.multimodal_token_space_processing:
        payload.update(token_space_request_fields(mm_token_expansion_start_len))
    if kv_cache_namespace is not None:
        payload["extra_key"] = kv_cache_namespace

    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)
    return payload, None


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]
    adopt_expanded_prompt_ids(sample, payload, output)

    if x := output["meta_info"].get("output_token_logprobs"):
        new_response_tokens = [item[1] for item in x]
        new_response_log_probs = [item[0] for item in x]
    else:
        new_response_tokens, new_response_log_probs = [], []

    if payload.get("return_sampling_mask", False):
        new_response_log_probs = append_sampling_metadata(
            sample,
            new_response_tokens,
            output["meta_info"],
            sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
        )

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs
    if payload.get("top_logprobs_num") or payload.get("sampling_logprobs_mode") == "support":
        append_rollout_topk_logprobs(
            sample,
            output["meta_info"],
            args.rollout_top_logprobs_num,
            sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
        )

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    # TODO handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, output, len(sample.tokens) - 1)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    # validate=True: SGLang sends plain base64; pybase64 skips its slow non-alphabet filtering pass.
    x = np.frombuffer(pybase64.b64decode(info, validate=True), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, num_tokens: int):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    routed_experts = _decode_topk_buffer(info, num_tokens, args.num_layers, -1)
    assert routed_experts.size == 0 or routed_experts.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return routed_experts


def get_indexer_topk_from_response(args, output, sample):
    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    expected_num_streams = getattr(args, "rollout_indexer_topk_num_streams", None)
    assert expected_num_streams is None or num_layers == expected_num_streams, (
        f"Server returned indexer_topk with {num_layers} streams but the model has "
        f"{expected_num_streams} indexer layers; replaying it would map streams to the wrong layers."
    )
    return _decode_topk_buffer(info, len(sample.tokens) - 1, num_layers, -1)
