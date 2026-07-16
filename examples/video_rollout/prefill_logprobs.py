from collections.abc import Mapping
from typing import Any

from miles.rollout.generate_utils.prefill_logprobs import (
    _build_prefill_scoring_payload,
    _extract_response_logprobs,
    recompute_samples_rollout_logprobs_via_prefill as recompute_default_samples,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

from .generate import render_prompt


def _build_video_prefill_scoring_payload(
    args: Any,
    sample: Sample,
    sampling_params: Mapping[str, Any],
    tokenizer,
    tools=None,
) -> dict[str, Any]:
    payload = _build_prefill_scoring_payload(args, sample, sampling_params)
    prompt_len = len(sample.tokens) - sample.response_length
    prompt = render_prompt(tokenizer, sample.prompt, tools=tools)
    rollout_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    payload["input_ids"] = rollout_prompt_ids + sample.tokens[prompt_len:]
    payload["video_data"] = sample.rollout_video_sources
    return payload


async def recompute_samples_rollout_logprobs_via_prefill(
    args: Any,
    samples: list[Sample],
    *,
    url: str,
    sampling_params: Mapping[str, Any],
    tokenizer,
) -> None:
    if not args.recompute_logprobs_via_prefill:
        return

    video_samples = [sample for sample in samples if sample.rollout_video_sources]
    await recompute_default_samples(
        args,
        [sample for sample in samples if not sample.rollout_video_sources],
        url=url,
        sampling_params=sampling_params,
    )

    tool_specs_path = getattr(args, "generate_tool_specs_path", None)
    tools = load_function(tool_specs_path) if tool_specs_path else None
    flush_url = url.rsplit("/", 1)[0] + "/flush_cache"

    for sample in video_samples:
        if sample.response_length == 0 or sample.status == Sample.Status.ABORTED:
            continue

        headers = None
        if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and sample.session_id:
            headers = {"X-SMG-Routing-Key": sample.session_id}

        await post(flush_url, {}, headers=headers)
        payload = _build_video_prefill_scoring_payload(args, sample, sampling_params, tokenizer, tools=tools)
        output = await post(url, payload, headers=headers)
        sample.rollout_log_probs = _extract_response_logprobs(sample, output["meta_info"])
        sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"
