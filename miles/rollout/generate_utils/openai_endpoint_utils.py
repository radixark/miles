"""
Utilities for the OpenAI endpoint
"""

import logging
from argparse import Namespace
from copy import deepcopy

from miles.rollout.generate_utils.generate_endpoint_utils import get_rollout_topk_from_response
from miles.router.session.sessions import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
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
    args: Namespace, input_sample: Sample, records: list[SessionRecord], tokenizer
) -> list[Sample]:
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
    )
    return [
        _compute_sample_from_openai_record(
            args, input_sample, record, tokenizer, tito_tokenizer, is_last=(i == len(records) - 1)
        )
        for i, record in enumerate(records)
    ]


def _compute_sample_from_openai_record(
    args: Namespace, input_sample: Sample, record: SessionRecord, tokenizer, tito_tokenizer, *, is_last: bool
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

    # Build full sample first (routed_experts reshape depends on len(tokens)-1).
    sample.tokens = prompt_token_ids + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, choice, sample, "routed_experts")

    # Strip trailing stop token from intermediate turns so merge_samples
    # prefix check works: turn N+1's prompt_token_ids comes from stripped
    # pretokenized, so turn N's sample.tokens must also be stripped to align.
    # The last sample keeps its stop token (no subsequent turn to match).
    if not is_last and tito_tokenizer.should_strip_trailing_stop_token(output_token_ids):
        sample.strip_last_output_token(tokenizer)

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    return sample
