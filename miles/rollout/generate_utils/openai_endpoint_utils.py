"""
Utilities for the OpenAI endpoint
"""

import logging
from argparse import Namespace
from copy import deepcopy

from miles.rollout.generate_utils.generate_endpoint_utils import get_rollout_topk_from_response
from miles.router.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"

    @staticmethod
    async def create(args: Namespace):
        router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        response = await post(f"{router_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(router_url=router_url, session_id=session_id)

    async def collect_records(self) -> tuple[list[SessionRecord], dict]:
        try:
            response = await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="get")
        except Exception as e:
            logger.warning(f"Failed to get session {self.session_id} records: {e}")
            raise
        response = GetSessionResponse.model_validate(response)
        records = response.records
        metadata = response.metadata

        try:
            await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="delete")
        except Exception as e:
            logger.warning(f"Failed to delete session {self.session_id} after collecting records: {e}")

        return records or [], metadata


def compute_samples_from_openai_records(
    args: Namespace,
    input_sample: Sample,
    records: list[SessionRecord],
    tokenizer,
    accumulated_token_ids: list[int] | None = None,
    max_trim_tokens: int = 0,
) -> list[Sample]:
    samples = []
    cursor = 0

    for i, record in enumerate(records):
        is_last = i == len(records) - 1
        prompt_ids = record.response["choices"][0]["prompt_token_ids"]
        output_ids = [t[1] for t in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]

        trim_count = 0
        if accumulated_token_ids is not None:
            cursor = len(prompt_ids)
            matched = 0
            for j in range(len(output_ids)):
                idx = cursor + j
                if idx < len(accumulated_token_ids) and output_ids[j] == accumulated_token_ids[idx]:
                    matched += 1
                else:
                    break
            trim_count = len(output_ids) - matched
            allowed = 0 if is_last else max_trim_tokens
            assert trim_count <= allowed, (
                f"trim_count {trim_count} exceeds allowed={allowed} "
                f"(is_last={is_last}, max_trim_tokens={max_trim_tokens}); "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[cursor:cursor+3]={accumulated_token_ids[cursor:cursor+3]}"
            )
            cursor += matched

        sample = _compute_sample_from_openai_record(args, input_sample, record, tokenizer, trim_count)
        samples.append(sample)

    if accumulated_token_ids is not None:
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
