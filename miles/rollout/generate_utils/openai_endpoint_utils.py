"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
from argparse import Namespace
from copy import deepcopy

from miles.rollout.generate_utils.generate_endpoint_utils import get_rollout_topk_from_response
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str, session_server_instance_id: str | None = None):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id

    @staticmethod
    async def create(args: Namespace):
        session_ip = getattr(args, "session_server_ip", None)
        session_port = getattr(args, "session_server_port", None)
        if not session_ip or not session_port:
            raise RuntimeError(
                "session_server_ip/session_server_port are not set. "
                "Pass --use-session-server to start the session server."
            )
        session_url = f"http://{session_ip}:{session_port}"
        session_server_instance_id = None
        try:
            health = await post(f"{session_url}/health", {}, action="get")
            if isinstance(health, dict):
                session_server_instance_id = health.get("session_server_instance_id")
                if session_server_instance_id is not None:
                    args.session_server_instance_id = session_server_instance_id
        except Exception as e:
            logger.warning("Failed to get session server health from %s: %s", session_url, e)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
        )

    async def collect_records(self) -> tuple[list[SessionRecord], dict]:
        try:
            response = await asyncio.wait_for(
                post(self.base_url, {}, action="get"),
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timed out waiting for session {self.session_id} records after {_SESSION_REQUEST_TIMEOUT}s "
                f"(likely stale HTTP keepalive connection). Returning empty records."
            )
            # Still attempt to clean up the session.
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception:
                logger.warning(f"Failed to delete session {self.session_id} after timeout")
            return [], {}
        except Exception as e:
            logger.warning(f"Failed to get session {self.session_id} records: {e}")
            raise
        response = GetSessionResponse.model_validate(response)
        records = response.records
        metadata = response.metadata

        try:
            await asyncio.wait_for(
                post(self.base_url, {}, action="delete"),
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Failed to delete session {self.session_id} after collecting records: {e}")

        return (records or []), metadata


def compute_samples_from_openai_records(
    args: Namespace,
    input_sample: Sample,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    """Convert per-turn session records into training Samples, aligning each
    turn's output tokens against the TITO accumulated token sequence.

    Each record carries its own ``prompt_token_ids`` and ``output_token_ids``
    (with logprobs).  We want to reuse those per-turn logprobs directly
    instead of re-decoding, but we must first trim "trailing tokens" — stop
    tokens the model emitted that the chat template also renders as the next
    turn's delimiter — to avoid double-counting.

    See ``TestTITOTrailingTokenTrim`` in
    ``tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py``
    for a concrete worked example with token-level walkthroughs.
    """
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.response["choices"][0]["prompt_token_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            # Step 1: position cursor right after this turn's prompt
            cursor = len(prompt_ids)

            # Step 2: greedily match output_ids against accumulated[cursor:]
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break

            # Step 3: unmatched trailing tokens were consumed by the next
            # turn's template rendering (e.g. stop tokens that double as
            # the next message delimiter) — strip them from the sample.
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={accumulated_token_ids[cursor:cursor+3]}"
            )

            # Step 4: advance cursor past matched output to the next turn
            cursor += matched

        sample = _compute_sample_from_openai_record(args, input_sample, record, tokenizer, trim_count)
        samples.append(sample)

    if accumulated_token_ids is not None:
        # Step 5: verify the entire accumulated sequence was consumed
        assert cursor == len(accumulated_token_ids), (
            f"cursor {cursor} != len(accumulated_token_ids) {len(accumulated_token_ids)} "
            f"after processing all {len(records)} records"
        )

    return samples


def _compute_sample_from_openai_record(
    args: Namespace, input_sample: Sample, record: SessionRecord, tokenizer, trim_count: int = 0
) -> Sample:
    choice = record.response["choices"][0]

    if "prompt_token_ids" in choice:
        prompt_token_ids = choice["prompt_token_ids"]
    else:
        raise ValueError("prompt_token_ids not found in response choice — ensure return_prompt_token_ids=True is set")

    output_token_ids = [item[1] for item in choice["meta_info"]["output_token_logprobs"]]
    output_log_probs = [item[0] for item in choice["meta_info"]["output_token_logprobs"]]

    sample = deepcopy(input_sample)
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        assert (
            request_input_ids == prompt_token_ids
        ), "for prompt part, input_ids return by sglang should match with the request input_ids"

    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, choice, sample, "routed_experts")

    if trim_count > 0:
        sample.strip_last_output_tokens(trim_count, tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    return sample


def truncate_samples_by_total_tokens(
    samples: list[Sample],
    max_seq_len: int,
    tokenizer,
) -> list[Sample]:
    """Truncate samples so the total token count (prompt + output, including
    env responses) does not exceed ``max_seq_len``.
    """
    result: list[Sample] = []

    for sample in samples:
        total = len(sample.tokens)
        if total <= max_seq_len:
            result.append(sample)
            continue

        overshoot = total - max_seq_len
        allowed_output = sample.response_length - overshoot
        if allowed_output <= 0:
            break

        _truncate_sample_output(sample, allowed_output, tokenizer)
        result.append(sample)
        break

    return result


def _truncate_sample_output(sample: Sample, keep_tokens: int, tokenizer) -> None:
    """Truncate a sample's output in-place to exactly ``keep_tokens`` tokens."""
    prompt_len = len(sample.tokens) - sample.response_length
    kept_ids = sample.tokens[prompt_len : prompt_len + keep_tokens]

    sample.tokens = sample.tokens[:prompt_len] + kept_ids
    sample.response = tokenizer.decode(kept_ids)
    sample.response_length = keep_tokens
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs = sample.rollout_log_probs[:keep_tokens]
    if sample.loss_mask is not None:
        sample.loss_mask = sample.loss_mask[:keep_tokens]
    sample.status = Sample.Status.TRUNCATED
