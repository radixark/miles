from argparse import Namespace

from miles.utils.http_utils import post


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

    async def collect(self):
        # TODO: for fault tolerance, we may want to change to GET + DELETE
        response = await post(f"{self.router_url}/sessions/{self.session_id}", {}, action="delete")
        return response["records"]


def compute_samples_from_openai_endpoint_records(records):
    return TODO
