"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.generate_utils.sampling_mask import should_return_sampling_mask
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.processing_utils import encode_image_for_rollout_engine, extract_multimodal_train_inputs
from miles.utils.replay_index import (
    VIA_STORE_REQUEST_KEY,
    ReplayIndex,
    sampled_logprobs_from_meta_info,
    set_outputs_via_store,
)
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
    kv_cache_namespace: str | None = None,
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
    if kv_cache_namespace is not None:
        payload["extra_key"] = kv_cache_namespace

    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)
    set_outputs_via_store(payload, use_output_store=args.use_output_store, evaluation=evaluation)
    return payload, None


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    if x := output["meta_info"].get("output_token_logprobs"):
        new_response_tokens = [item[1] for item in x]
        new_response_log_probs = [item[0] for item in x]
    else:
        new_response_tokens, new_response_log_probs = [], []

    if payload.get("return_sampling_mask", False):
        new_response_log_probs = sampled_logprobs_from_meta_info(
            output["meta_info"],
            output_token_ids=new_response_tokens,
            support_logprobs=payload.get("sampling_logprobs_mode") == "support",
            aborted=False,
        )

    previous_response_length = sample.response_length
    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs
    if payload.get("top_logprobs_num"):
        append_rollout_topk_logprobs(sample, output["meta_info"], args.rollout_top_logprobs_num)

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    replay_index = compute_replay_index_from_response(
        args,
        request=payload,
        meta_info=output["meta_info"],
        routed_experts_rows=len(sample.tokens) - 1,
        indexer_topk_rows=len(sample.tokens) - 1,
        num_output_tokens=len(new_response_tokens),
    )
    sample.replay_index = sample.replay_index.append_response(replay_index, response_length=previous_response_length)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def compute_replay_index_from_response(
    config,
    *,
    request: Mapping[str, Any],
    meta_info: Mapping[str, Any],
    routed_experts_rows: int | None,
    indexer_topk_rows: int | None,
    num_output_tokens: int,
    aborted: bool = False,
    allow_extra_routed_experts_row: bool = False,
) -> ReplayIndex:
    """Parse one response's replay fields; ``config`` is the run args or the session server config."""
    return ReplayIndex.from_meta_info(
        meta_info,
        via_store=bool(request.get(VIA_STORE_REQUEST_KEY)),
        routed_experts_rows=routed_experts_rows,
        num_layers=getattr(config, "num_layers", None),
        indexer_topk_rows=indexer_topk_rows,
        indexer_num_streams=getattr(config, "rollout_indexer_topk_num_streams", None),
        return_sampling_mask=bool(request.get("return_sampling_mask")),
        support_logprobs=request.get("sampling_logprobs_mode") == "support",
        num_output_tokens=num_output_tokens,
        aborted=aborted,
        allow_extra_routed_experts_row=allow_extra_routed_experts_row,
    )
