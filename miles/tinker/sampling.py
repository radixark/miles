"""Direct sampling: fan one official sample operation out as neutral SGLang calls, no Ray hop."""

from __future__ import annotations

import asyncio
from typing import Any

from miles.rollout.sglang_rollout import call_sglang_generate_endpoint, parse_output_token_logprobs
from miles.tinker import codec


async def run_sample_operation(
    args,
    *,
    request_id: str,
    binding: dict[str, Any] | None,
    payload: dict[str, Any],
    post_json,
    child_semaphore: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], list[float | None] | None]:
    """Run one /asample operation: bounded child fanout, one-shot posts, ordered results."""
    input_ids = codec.prompt_tokens_from_wire(payload["prompt"])
    sampling_params = codec.sglang_sampling_params(payload.get("sampling_params") or {})
    include_prompt_logprobs = bool(payload.get("prompt_logprobs"))
    num_samples = int(payload.get("num_samples") or 1)
    lora_path = binding.get("lora_path") if binding else None
    extra_key = binding.get("extra_key") if binding else None
    root_rid = request_id if binding is None else f"{binding['name']}::{request_id}"
    extra_payload = {"logprob_start_len": 0} if include_prompt_logprobs else None

    async def one_child(index: int) -> dict[str, Any]:
        async with child_semaphore:
            return await call_sglang_generate_endpoint(
                args,
                input_ids=input_ids,
                sampling_params=dict(sampling_params),
                lora_path=lora_path,
                rid=f"{root_rid}::c{index}",
                extra_key=extra_key,
                extra_payload=extra_payload,
                post_json=post_json,
            )

    outputs = await asyncio.gather(*(one_child(index) for index in range(num_samples)))
    sequences = []
    for output in outputs:
        tokens, logprobs = parse_output_token_logprobs(output)
        stop_reason = "length" if output["meta_info"]["finish_reason"]["type"] == "length" else "stop"
        sequences.append({"stop_reason": stop_reason, "tokens": tokens, "logprobs": logprobs})
    prompt_logprobs = None
    if include_prompt_logprobs:
        prompt_logprobs = [item[0] for item in outputs[0]["meta_info"]["input_token_logprobs"]]
    return sequences, prompt_logprobs
