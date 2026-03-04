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
    if not records:
        raise RuntimeError("No OAI session records were collected.")

    samples: list[Sample] = []
    for index, record in enumerate(records):
        try:
            samples.append(_compute_sample_from_openai_record(input_sample, record, tokenizer))
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to parse OAI session record at "
                f"index={index}, timestamp={record.timestamp}, status_code={record.status_code}, path={record.path}"
            ) from exc

    return samples


def _infer_input_token_ids(choice: dict, record: SessionRecord, tokenizer) -> list[int]:
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        return list(request_input_ids)

    input_token_ids = choice.get("input_token_ids")
    if input_token_ids is not None:
        return list(input_token_ids)

    raise KeyError("missing input_token_ids and request.input_ids")


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


def _extract_choice_content(choice: dict) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _normalize_finish_reason(choice: dict) -> str | None:
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    return finish_reason


def _compute_sample_from_openai_record(input_sample: Sample, record: SessionRecord, tokenizer) -> Sample:
    # TODO may refine after @guapisolo's implementation
    choice = record.response["choices"][0]

    input_token_ids = _infer_input_token_ids(choice, record, tokenizer)
    # IMPORTANT: For multi-turn agentic tasks, later prompts are constructed by re-tokenizing
    # the message content strings. Tokenization is not guaranteed to be invertible at the
    # token-id level (decode->encode can change segmentation). To keep turn-to-turn token
    # streams consistent for merge_samples(), derive output_token_ids from the returned
    # message content, and only keep rollout logprobs when they align exactly.
    content = _extract_choice_content(choice)
    output_token_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    output_log_probs: list[float] | None = None
    if isinstance(choice.get("logprobs", {}).get("content"), list):
        extracted_ids, extracted_log_probs = _extract_output_tokens(choice, tokenizer)
        if extracted_ids == output_token_ids:
            output_log_probs = extracted_log_probs

    sample = deepcopy(input_sample)
    request_input_ids = record.request.get("input_ids")
    if request_input_ids is not None:
        assert (
            request_input_ids == input_token_ids
        ), "for prompt part, input_ids return by sglang should match with the request input_ids"
    sample.tokens = list(input_token_ids) + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = content
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
