"""Single-turn token-in-token-out (TITO) generation against vLLM's ``/inference/v1/generate``."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_routing_headers,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample, WeightVersionsPerCall


def _normalize_token_ids(raw_ids: Any) -> list[int]:
    to_list = getattr(
        raw_ids, "tolist", None
    )  # config-access-exempt: tokenizers/processors may return numpy arrays, tensors, or lists
    if callable(to_list):
        raw_ids = to_list()
    return [int(tok) for tok in raw_ids]


def build_vllm_generate_payload(
    args: Any,
    *,
    input_ids: list[int],
    sampling_params: Mapping[str, Any],
    kv_cache_namespace: str | None = None,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    """Build a vLLM 0.32+ ``POST /inference/v1/generate`` payload from Miles sampling params."""
    sp = deepcopy(dict(sampling_params))
    max_new_tokens = sp.pop("max_new_tokens", args.rollout_max_response_len)
    if (max_seq_len := args.max_seq_len) is not None and max_new_tokens is not None:
        max_new_tokens = min(max_new_tokens, max_seq_len - len(input_ids))
    if max_new_tokens is not None and max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    if args.rollout_top_logprobs_num > 0:
        raise NotImplementedError(
            "vllm_single_turn records only the sampled token's logprob; --rollout-top-logprobs-num needs the "
            "SGLang generate path"
        )

    # one logprob: vLLM always returns the sampled token's, which is all the trainer reads
    vllm_sp: dict[str, Any] = {"logprobs": 1}
    if max_new_tokens is not None:
        vllm_sp["max_tokens"] = int(max_new_tokens)
    if (temperature := sp.get("temperature")) is not None:
        vllm_sp["temperature"] = temperature
    if (top_p := sp.get("top_p")) is not None:
        vllm_sp["top_p"] = top_p
    if (top_k := sp.get("top_k")) is not None:
        vllm_sp["top_k"] = top_k
    if stop := sp.get("stop"):
        vllm_sp["stop"] = stop
    if stop_token_ids := sp.get("stop_token_ids"):
        vllm_sp["stop_token_ids"] = stop_token_ids
    if (skip_special := sp.get("skip_special_tokens")) is not None:
        vllm_sp["skip_special_tokens"] = skip_special
    if (spaces := sp.get("spaces_between_special_tokens")) is not None:
        vllm_sp["spaces_between_special_tokens"] = spaces
    if sp.get("no_stop_trim"):
        vllm_sp["include_stop_str_in_output"] = True
    seed = sp.get("sampling_seed") if sp.get("sampling_seed") is not None else sp.get("seed")
    if seed is not None:
        vllm_sp["seed"] = seed

    payload: dict[str, Any] = {
        "token_ids": list(input_ids),
        "sampling_params": vllm_sp,
        "output_mode": "tokens",
        "return_token_ids": True,
    }
    if kv_cache_namespace:
        payload["cache_salt"] = kv_cache_namespace
    return payload, None


def _extract_choice_logprobs(choice: Mapping[str, Any], expected_len: int) -> list[float]:
    if expected_len == 0:
        return []
    logprobs_obj = choice.get("logprobs")
    content = logprobs_obj.get("content") if isinstance(logprobs_obj, Mapping) else None
    if content is None:
        raise ValueError(
            f"vLLM /inference/v1/generate returned {expected_len} tokens without logprobs.content "
            f"(requested logprobs=1)"
        )
    if len(content) != expected_len:
        raise ValueError(f"vLLM /inference/v1/generate returned {len(content)} logprobs for {expected_len} tokens")
    return [float(entry["logprob"]) for entry in content]


def _normalize_weight_version_spans(raw_spans: Any) -> list[dict[str, Any]] | None:
    if raw_spans is None:
        return None
    normalized: list[dict[str, Any]] = []
    for span in raw_spans:
        version = span["version"]
        if version is None or str(version) == "":
            continue
        normalized.append(
            {
                "version": str(version),
                "start": int(span["start"]),
                "end": int(span["end"]),
            }
        )
    return normalized


def _status_from_finish_reason(raw_finish_reason: Any) -> Sample.Status:
    if isinstance(raw_finish_reason, Mapping):
        reason = raw_finish_reason.get("type", "stop")
    else:
        reason = raw_finish_reason or "stop"
    match reason:
        case "length":
            return Sample.Status.TRUNCATED
        case "abort" | "cancelled":
            return Sample.Status.ABORTED
        case _:
            return Sample.Status.COMPLETED


def apply_vllm_generate_response(
    sample: Sample,
    *,
    input_ids: list[int],
    output: Mapping[str, Any],
    tokenizer: Any,
    skip_special_tokens: bool = False,
) -> Sample:
    """Apply a vLLM ``/inference/v1/generate`` JSON response onto ``sample``."""
    if len(sample.response) == 0 and not sample.tokens:
        prompt_token_ids = output.get("prompt_token_ids")
        sample.tokens = _normalize_token_ids(prompt_token_ids) if prompt_token_ids is not None else list(input_ids)

    choices = output.get("choices")
    if not choices:
        raise ValueError(f"vLLM /inference/v1/generate response has no choices: {output!r}")
    choice = choices[0]

    new_tokens = [int(t) for t in (choice.get("token_ids") or [])]
    new_log_probs = _extract_choice_logprobs(choice, expected_len=len(new_tokens))

    if (text := choice.get("text")) is not None:
        new_text = str(text)
    elif new_tokens:
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
    else:
        new_text = ""

    sample.tokens = sample.tokens + new_tokens
    sample.response_length += len(new_tokens)
    sample.response += new_text

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_log_probs

    if sample.loss_mask is not None:
        sample.loss_mask += [1] * len(new_tokens)

    usage = output.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens") or 0
    if prompt_tokens is not None:
        sample.prefix_cache_info.add({"prompt_tokens": int(prompt_tokens), "cached_tokens": int(cached_tokens)})

    meta_info: dict[str, Any] = {
        "output_token_logprobs": list(zip(new_log_probs, new_tokens, strict=True)),
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else len(input_ids),
    }
    normalized_spans = _normalize_weight_version_spans(choice.get("weight_versions"))
    if normalized_spans is not None:
        meta_info["weight_versions"] = normalized_spans
    sample.weight_versions.append(WeightVersionsPerCall.from_meta_info(meta_info, output_end=len(sample.tokens)))

    sample.status = _status_from_finish_reason(choice.get("finish_reason"))
    return sample


async def generate(input: GenerateFnInput, url: str | None = None) -> GenerateFnOutput:
    """Single-turn generation via vLLM's ``POST /inference/v1/generate`` TITO endpoint."""
    args = input.args
    sample = input.sample
    sampling_params = deepcopy(input.sampling_params)
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"
    url = url or f"http://{args.sglang_router_ip}:{args.sglang_router_port}/inference/v1/generate"

    prompt_ids = _normalize_token_ids(compute_prompt_ids_from_sample(input.state, sample))

    # Handle partial rollout resuming
    if len(sample.response) > 0:
        input_ids = list(sample.tokens)
        if sampling_params.get("max_new_tokens") is not None:
            sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
            assert sampling_params["max_new_tokens"] >= 0
            if sampling_params["max_new_tokens"] == 0:
                sample.status = Sample.Status.TRUNCATED
                return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = build_vllm_generate_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        kv_cache_namespace=sample.kv_cache_namespace,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
    apply_vllm_generate_response(
        sample,
        input_ids=input_ids,
        output=output,
        tokenizer=input.state.tokenizer,
        skip_special_tokens=bool(sampling_params.get("skip_special_tokens", False)),
    )
    return GenerateFnOutput(samples=sample)
