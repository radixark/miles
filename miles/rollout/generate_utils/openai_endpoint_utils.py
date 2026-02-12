"""
Utilities for the OpenAI endpoint
"""

import logging
from argparse import Namespace
from copy import deepcopy

from miles.router.session.sessions import GetSessionResponse, SessionRecord
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

    async def collect_records(self) -> list[SessionRecord]:
        try:
            response = await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="get")
        except Exception as e:
            logger.warning(f"Failed to get session {self.session_id} records: {e}")
            raise
        response = GetSessionResponse.model_validate(response)
        records = response.records

        try:
            await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="delete")
        except Exception as e:
            logger.warning(f"Failed to delete session {self.session_id} after collecting records: {e}")

        return records or []


def compute_samples_from_openai_records(input_sample: Sample, records: list[SessionRecord], tokenizer) -> list[Sample]:
    return [_compute_sample_from_openai_record(input_sample, record, tokenizer) for record in records]


def _compute_sample_from_openai_record(input_sample: Sample, record: SessionRecord, tokenizer) -> Sample:
    # TODO may refine after @guapisolo's implementation
    choice = record.response["choices"][0]

    input_token_ids = choice["input_token_ids"]
    output_token_ids = [item["token_id"] for item in choice["logprobs"]["content"]]
    output_log_probs = [item["logprob"] for item in choice["logprobs"]["content"]]

    sample = deepcopy(input_sample)
    # sample.tokens = record.request["input_ids"] + output_token_ids
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        assert (
            request_input_ids == input_token_ids
        ), "for prompt part, input_ids return by sglang should match with the request input_ids"
    sample.tokens = input_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    return sample
