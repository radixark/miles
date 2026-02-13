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


def _infer_input_token_ids(choice: dict, record: SessionRecord, tokenizer) -> list[int]:
    input_token_ids = choice.get("input_token_ids")
    if input_token_ids is not None:
        return input_token_ids

    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        return request_input_ids

    messages = record.request.get("messages")
    if messages is None:
        raise KeyError("input_token_ids")

    tools = record.request.get("tools")
    if tools is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,
            tools=tools,
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=True,
    )


def _extract_output_tokens(choice: dict, tokenizer) -> tuple[list[int], list[float]]:
    logprob_items = choice.get("logprobs", {}).get("content")
    if logprob_items is None:
        raise KeyError("logprobs.content")

    output_token_ids: list[int] = []
    output_log_probs: list[float] = []
    for item in logprob_items:
        token_id = item.get("token_id")
        if token_id is None:
            token = item.get("token")
            if token is None:
                raise KeyError("token_id")
            token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(token_ids) != 1:
                raise RuntimeError(f"Cannot infer stable token_id from token={token!r}")
            token_id = token_ids[0]

        output_token_ids.append(int(token_id))
        output_log_probs.append(item["logprob"])

    return output_token_ids, output_log_probs


def _normalize_finish_reason(choice: dict) -> str | None:
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    return finish_reason


def _compute_sample_from_openai_record(input_sample: Sample, record: SessionRecord, tokenizer) -> Sample:
    # TODO may refine after @guapisolo's implementation
    choice = record.response["choices"][0]

    input_token_ids = _infer_input_token_ids(choice, record, tokenizer)
    output_token_ids, output_log_probs = _extract_output_tokens(choice, tokenizer)

    sample = deepcopy(input_sample)
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        assert (
            request_input_ids == input_token_ids
        ), "for prompt part, input_ids return by sglang should match with the request input_ids"
    sample.tokens = list(input_token_ids) + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = tokenizer.decode(output_token_ids)
    sample.response_length = len(output_token_ids)
    sample.loss_mask = [1] * len(output_token_ids)

    # TODO unify with Sample.update_from_meta_info
    match _normalize_finish_reason(choice):
        case "stop" | "tool_calls":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case _:
            sample.status = Sample.Status.ABORTED

    return sample
