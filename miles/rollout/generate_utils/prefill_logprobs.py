from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from miles.utils.http_utils import post
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.multi_lora import slot_lora_name
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample


def _lora_path_for_sample(args: Any, sample: Sample) -> str | None:
    """The adapter name this sample's scoring request must be served with.

    Multi-LoRA samples carry an adapter ref and must score under their own
    slot's adapter — the engines only know ``__miles_slot_{N}`` names there.
    Single-adapter LoRA uses the fixed name; base models use none.
    """
    if sample.adapter is not None:
        return slot_lora_name(sample.adapter.slot)
    if is_lora_enabled(args):
        return LORA_ADAPTER_NAME
    return None


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

    payload = {
        "input_ids": sample.tokens,
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

    if (lora_path := _lora_path_for_sample(args, sample)) is not None:
        payload["lora_path"] = lora_path

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload


def _can_batch_prefill_score(args: Any, samples: list[Sample]) -> bool:
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing":
        return False
    return not any(sample.multimodal_inputs and sample.multimodal_inputs.get("images") for sample in samples)


def _build_batch_prefill_scoring_payload(
    args: Any,
    samples: list[Sample],
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = [_build_prefill_scoring_payload(args, sample, sampling_params) for sample in samples]
    logprob_start_len = payloads[0]["logprob_start_len"]
    if any(payload["logprob_start_len"] != logprob_start_len for payload in payloads):
        raise ValueError("Batched SGLang prefill scoring requires a shared logprob_start_len")

    lora_paths = {payload.get("lora_path") for payload in payloads}
    if len(lora_paths) > 1:
        # A batch payload carries a single lora_path; scoring a mixed-adapter
        # batch under one adapter would silently corrupt the other samples'
        # logprobs. Callers must group by adapter first.
        raise ValueError("Batched SGLang prefill scoring requires a shared lora_path")

    batch_payload: dict[str, Any] = {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": payloads[0]["sampling_params"],
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    if (lora_path := next(iter(lora_paths))) is not None:
        batch_payload["lora_path"] = lora_path
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
        # A batch shares one logprob_start_len and one lora_path, so group by
        # both: mixed-adapter batches (multi-LoRA) must not score every sample
        # under whichever adapter came first.
        samples_by_batch_key: dict[tuple[int, str | None], list[Sample]] = defaultdict(list)
        for sample in samples_to_score:
            prompt_len = len(sample.tokens) - sample.response_length
            samples_by_batch_key[(prompt_len - 1, _lora_path_for_sample(args, sample))].append(sample)

        for batch_samples in samples_by_batch_key.values():
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
