"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.rollout.generate_utils.output_store import ReplayOutputs, output_store_enabled, resolve_replay_outputs
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata, should_return_sampling_mask
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.processing_utils import encode_image_for_rollout_engine, extract_multimodal_train_inputs
from miles.utils.types import Sample


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
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


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    *,
    evaluation: bool = False,
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
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)
        maybe_request_outputs_via_store(args, payload)
    return payload, None


def maybe_request_outputs_via_store(args, payload: dict[str, Any]) -> None:
    """Ask SGLang to return a training request's replay outputs through the output store.

    Call it for training requests only: eval engines may run without the backend.
    """
    if output_store_enabled(args) and any(
        payload.get(key) for key in ("return_routed_experts", "return_indexer_topk", "return_sampling_mask")
    ):
        payload["return_outputs_via_store"] = True


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Read the output-store bundle first, so a bad one leaves the Sample untouched.
    replay = await resolve_replay_outputs(output["meta_info"])

    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

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
            replay=replay,
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
    sample.rollout_routed_experts = get_routed_experts_from_response(
        args, output, len(sample.tokens) - 1, replay=replay
    )
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample, replay=replay)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def _check_topk_rows(array: np.ndarray, name: str, *, num_tokens: int, num_layers: int) -> np.ndarray:
    """An output-store (tokens, layers, topk) array must match what the inline decode would reshape to."""
    if array.shape[:2] != (max(num_tokens, 0), num_layers):
        raise ValueError(
            f"{name} from the output store has shape {array.shape}, expected ({num_tokens}, {num_layers}, topk)"
        )
    return array


def get_routed_experts_from_response(args, output, num_tokens: int, replay: ReplayOutputs | None = None):
    if replay is not None and replay.routed_experts is not None:
        routed_experts = _check_topk_rows(
            replay.routed_experts, "routed_experts", num_tokens=num_tokens, num_layers=args.num_layers
        )
    else:
        info = output["meta_info"].get("routed_experts")
        if info is None:
            return None
        routed_experts = _decode_topk_buffer(info, num_tokens, args.num_layers, -1)
    assert routed_experts.size == 0 or routed_experts.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return routed_experts


def get_indexer_topk_from_response(args, output, sample, replay: ReplayOutputs | None = None):
    num_tokens = len(sample.tokens) - 1
    if replay is not None and replay.indexer_topk is not None:
        # An output-store response carries no indexer_topk_num_layers; the array shape holds it.
        num_layers = replay.indexer_topk.shape[1]
        _assert_indexer_streams(args, num_layers)
        return _check_topk_rows(replay.indexer_topk, "indexer_topk", num_tokens=num_tokens, num_layers=num_layers)

    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    _assert_indexer_streams(args, num_layers)
    return _decode_topk_buffer(info, num_tokens, num_layers, -1)


def _assert_indexer_streams(args, num_layers: int) -> None:
    expected_num_streams = getattr(args, "rollout_indexer_topk_num_streams", None)
    assert expected_num_streams is None or num_layers == expected_num_streams, (
        f"Server returned indexer_topk with {num_layers} streams but the model has "
        f"{expected_num_streams} indexer layers; replaying it would map streams to the wrong layers."
    )
