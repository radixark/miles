"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.processing_utils import (
    call_processor,
    encode_image_for_rollout_engine,
    extract_multimodal_train_inputs,
)
from miles.utils.types import Sample


def build_rollout_media_payload(
    multimodal_inputs: dict[str, Any] | None,
    video_sources: list[str] | None,
) -> dict[str, list[str]]:
    payload = {}
    if images := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
    if video_sources:
        payload["video_data"] = video_sources
    return payload


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt
    has_processor_inputs = bool(
        state.processor
        and sample.multimodal_inputs
        and any(value is not None for value in sample.multimodal_inputs.values())
    )

    if not has_processor_inputs:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )

        sample.rollout_prompt_ids = None
        return state.tokenizer.encode(prompt, add_special_tokens=False)

    if sample.rollout_video_sources:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )

        sample.rollout_prompt_ids = state.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
    else:
        sample.rollout_prompt_ids = None

    processor_output = call_processor(state.processor, prompt, sample.multimodal_inputs)
    prompt_ids = processor_output["input_ids"][0]
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()

    # TODO shall we move it to other places? then can make this function immutable
    sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)

    return prompt_ids


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


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    rollout_video_sources: list[str] | None = None,
    rollout_input_ids: list[int] | None = None,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    payload = {
        "input_ids": rollout_input_ids if rollout_input_ids is not None else input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    payload.update(build_rollout_media_payload(multimodal_inputs, rollout_video_sources))

    return payload, None


def compute_rollout_input_ids(sample: Sample, input_ids: list[int], processor_prompt_ids: list[int]) -> list[int]:
    if sample.rollout_prompt_ids is None:
        return input_ids

    return sample.rollout_prompt_ids + input_ids[len(processor_prompt_ids) :]


def validate_video_prompt_expansion(sample: Sample, meta_info: dict[str, Any]) -> None:
    """Ensure SGLang and the trainer expanded a compact video prompt identically."""
    if sample.rollout_prompt_ids is None:
        return

    engine_prompt_tokens = meta_info.get("prompt_tokens")
    if isinstance(engine_prompt_tokens, bool):
        engine_prompt_tokens = None
    try:
        engine_prompt_tokens = int(engine_prompt_tokens)
    except (TypeError, ValueError):
        engine_prompt_tokens = -1

    if engine_prompt_tokens != len(sample.tokens):
        raise RuntimeError(
            "engine/trainer video expansion mismatch: engine prompt is "
            f"{engine_prompt_tokens} tokens, trainer expansion is "
            f"{len(sample.tokens)}; frame sampling or timestamp conventions differ"
        )


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    validate_video_prompt_expansion(sample, output["meta_info"])

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
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, num_tokens: int):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    return _decode_topk_buffer(info, num_tokens, args.num_layers, -1)


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
