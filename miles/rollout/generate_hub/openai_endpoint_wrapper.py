from argparse import Namespace
from copy import deepcopy

from miles.router.sessions import DeleteSessionResponse, SessionRecord
from miles.utils.http_utils import post
from miles.utils.types import Sample


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"

    @staticmethod
    async def create(args: Namespace):
        router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        session_id = (await post(f"{router_url}/sessions", {}))["session_id"]
        return OpenAIEndpointTracer(router_url=router_url, session_id=session_id)

    async def collect_records(self) -> list[SessionRecord]:
        # TODO: for fault tolerance, we may want to change to GET + DELETE
        response = await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="delete")
        response = DeleteSessionResponse.model_validate(response)
        return response.records


def compute_samples_from_openai_records(input_sample: Sample, records: list[SessionRecord]) -> list[Sample]:
    return [
        _compute_sample_from_openai_record(input_sample, record)
        for record in records
    ]

def _compute_sample_from_openai_record(input_sample: Sample, record: SessionRecord) -> Sample:
    sample = deepcopy(input_sample)
    sample.tokens = record.extras.input_ids + TODO
    sample.loss_mask = TODO
    sample.rollout_log_probs = TODO
    sample.response = TODO
    sample.response_length = TODO

    num_tool_response_tokens = len(prompt_ids) - len(sample.tokens)
    if num_tool_response_tokens > 0:
        sample.tokens += prompt_ids[-num_tool_response_tokens:]
        sample.loss_mask += [0] * num_tool_response_tokens
        sample.rollout_log_probs += [0.0] * num_tool_response_tokens
        sample.response_length += num_tool_response_tokens

    sample.tokens += gen_token_ids
    sample.loss_mask += [1] * len(gen_token_ids)
    sample.rollout_log_probs += gen_log_probs
    sample.response += gen_text
    sample.response_length += len(gen_token_ids)

    _update_sample_status_from_oai_response(sample, record.response)

    return sample


def _extract_generation_from_oai_response(resp: dict) -> tuple[list[int], list[float], str]:
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content") or ""

    logprobs_data = choice.get("logprobs", {})
    content = logprobs_data.get("content") or []

    token_ids = [item["token_id"] for item in content]
    log_probs = [item["logprob"] for item in content]

    return token_ids, log_probs, text


def _update_sample_status_from_oai_response(sample: Sample, resp: dict):
    choice = resp.get("choices", [{}])[0]
    finish_reason = choice.get("finish_reason", "")

    match finish_reason:
        case "stop":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
