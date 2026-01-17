from argparse import Namespace
from copy import deepcopy

from miles.router.sessions import DeleteSessionResponse, SessionRecord
from miles.utils.http_utils import post
from miles.utils.mask_utils import get_response_lengths
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
    return [_compute_sample_from_openai_record(input_sample, record) for record in records]


# NOTE: Do not assign `loss_mask`, since here it is a single-turn
def _compute_sample_from_openai_record(input_sample: Sample, record: SessionRecord) -> Sample:
    choice = record.response["choices"][0]
    output_token_ids = [item["token_id"] for item in choice["logprobs"]["content"]]
    output_log_probs = [item["logprob"] for item in choice["logprobs"]["content"]]

    # TODO refine after @guapisolo's implementation
    sample = deepcopy(input_sample)
    sample.tokens = record.request["input_ids"] + output_token_ids
    sample.rollout_log_probs = output_log_probs
    sample.response = choice["message"]["content"]
    sample.response_length = get_response_lengths([sample.loss_mask])[0]

    # TODO unify with Sample.update_from_meta_info
    match choice["finish_reason"]:
        case "stop":
            sample.status = Sample.Status.COMPLETED
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED

    return sample
