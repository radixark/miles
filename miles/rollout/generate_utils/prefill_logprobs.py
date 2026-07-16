from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from miles.utils.http_utils import post
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.types import Sample

from .generate_endpoint_utils import build_rollout_media_payload


def _build_prefill_scoring_payload(
    args: Any,
    sample: Sample,
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len <= 0:
        raise ValueError(
            "Cannot recompute rollout logprobs via prefill without at least one prompt token: "
            f"tokens={len(sample.tokens)}, response_length={sample.response_length}"
        )

    rollout_input_ids = sample.tokens
    if sample.rollout_prompt_ids is not None:
        rollout_input_ids = sample.rollout_prompt_ids + sample.tokens[prompt_len:]
    payload = {
        "input_ids": rollout_input_ids,
        "sampling_params": {
            **dict(sampling_params),
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # SGLang returns input_token_logprobs aligned to tokens from logprob_start_len,
        # with the first value None. Start one token before the response so the
        # returned tail contains every response-token logprob.
        "logprob_start_len": prompt_len - 1,
    }

    if is_lora_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME

    payload.update(build_rollout_media_payload(sample.multimodal_inputs, sample.rollout_video_sources))

    return payload


def _can_batch_prefill_score(args: Any, samples: list[Sample]) -> bool:
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing":
        return False

    for sample in samples:
        multimodal_inputs = sample.multimodal_inputs or {}
        if multimodal_inputs.get("images") or multimodal_inputs.get("videos") or sample.rollout_video_sources:
            return False

    return True


def _build_batch_prefill_scoring_payload(
    args: Any,
    samples: list[Sample],
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = [_build_prefill_scoring_payload(args, sample, sampling_params) for sample in samples]
    logprob_start_len = payloads[0]["logprob_start_len"]
    if any(payload["logprob_start_len"] != logprob_start_len for payload in payloads):
        raise ValueError("Batched SGLang prefill scoring requires a shared logprob_start_len")

    batch_payload: dict[str, Any] = {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": payloads[0]["sampling_params"],
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    if "lora_path" in payloads[0]:
        batch_payload["lora_path"] = payloads[0]["lora_path"]
    return batch_payload


def _extract_response_logprobs(sample: Sample, meta_info: Mapping[str, Any]) -> list[float]:
    input_token_logprobs = meta_info.get("input_token_logprobs")
    if not input_token_logprobs:
        raise ValueError("SGLang prefill scoring response did not include input_token_logprobs")

    response_items = input_token_logprobs[-sample.response_length :]
    response_tokens = sample.tokens[-sample.response_length :]
    scored_tokens = [item[1] for item in response_items]
    if scored_tokens != response_tokens:
        raise ValueError(
            "SGLang prefill scoring token alignment mismatch: "
            f"expected response tail {response_tokens[:8]}... len={len(response_tokens)}, "
            f"got {scored_tokens[:8]}... len={len(scored_tokens)}"
        )

    response_logprobs = [item[0] for item in response_items]
    if any(logprob is None for logprob in response_logprobs):
        raise ValueError("SGLang prefill scoring returned None for a response-token logprob")

    return response_logprobs


async def recompute_rollout_logprobs_via_prefill(
    args: Any,
    sample: Sample,
    *,
    url: str,
    sampling_params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> None:
    if not getattr(args, "recompute_logprobs_via_prefill", False):
        return
    if sample.response_length == 0:
        sample.rollout_log_probs = []
        return
    if sample.status == Sample.Status.ABORTED:
        return

    payload = _build_prefill_scoring_payload(args, sample, sampling_params)
    output = await post(url, payload, headers=headers)
    sample.rollout_log_probs = _extract_response_logprobs(sample, output["meta_info"])
    sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"


async def recompute_samples_rollout_logprobs_via_prefill(
    args: Any,
    samples: list[Sample],
    *,
    url: str,
    sampling_params: Mapping[str, Any],
) -> None:
    if not getattr(args, "recompute_logprobs_via_prefill", False):
        return

    samples_to_score = [
        sample for sample in samples if sample.response_length != 0 and sample.status != Sample.Status.ABORTED
    ]
    if not samples_to_score:
        return

    flush_url = url.rsplit("/", 1)[0] + "/flush_cache"

    if _can_batch_prefill_score(args, samples_to_score):
        samples_by_logprob_start_len: dict[int, list[Sample]] = defaultdict(list)
        for sample in samples_to_score:
            prompt_len = len(sample.tokens) - sample.response_length
            samples_by_logprob_start_len[prompt_len - 1].append(sample)

        for batch_samples in samples_by_logprob_start_len.values():
            # SGLang can serve scoring requests from radix/KV cache. Flush before
            # each scoring group so every group uses the same clean-prefill path.
            await post(flush_url, {})
            payload = _build_batch_prefill_scoring_payload(args, batch_samples, sampling_params)
            outputs = await post(url, payload)
            if not isinstance(outputs, list):
                raise ValueError(f"SGLang batch prefill scoring returned {type(outputs).__name__}, expected list")
            if len(outputs) != len(batch_samples):
                raise ValueError(
                    "SGLang batch prefill scoring output count mismatch: "
                    f"expected {len(batch_samples)}, got {len(outputs)}"
                )
            for sample, output in zip(batch_samples, outputs, strict=True):
                sample.rollout_log_probs = _extract_response_logprobs(sample, output["meta_info"])
                sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"
        return

    for sample in samples_to_score:
        headers = None
        uses_consistent_hashing = getattr(args, "sglang_router_policy", None) == "consistent_hashing"
        if uses_consistent_hashing and sample.session_id:
            headers = {"X-SMG-Routing-Key": sample.session_id}

        await post(flush_url, {}, headers=headers)
        await recompute_rollout_logprobs_via_prefill(
            args,
            sample,
            url=url,
            sampling_params=sampling_params,
            headers=headers,
        )
