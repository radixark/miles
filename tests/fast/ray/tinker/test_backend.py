import asyncio
from pathlib import Path

import pytest

from miles.ray.tinker.backend import TinkerBackend
from miles.ray.tinker.state import Operation


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {
            "output_ids": [123],
            "meta_info": {
                "output_token_logprobs": [[-0.25, 123]],
                "finish_reason": {"type": "length"},
                "input_token_logprobs": [[None, 10], [-0.5, 11]],
                "input_top_logprobs": [None, [[-0.1, 12], [-0.2, 13]]],
            },
        }


class _Client:
    def __init__(self):
        self.requests = []

    async def post(self, url, json):
        self.requests.append((url, json))
        return _Response()


@pytest.mark.asyncio
async def test_sample_payload_maps_tinker_seed_to_sglang_sampling_seed():
    backend = object.__new__(TinkerBackend)
    backend.client = _Client()
    backend.router_url = "http://router"
    payload = {
        "num_samples": 2,
        "prompt": [10, 11],
        "sampling_params": {
            "max_tokens": 3,
            "seed": 41,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.9,
        },
        "prompt_logprobs": True,
        "topk_prompt_logprobs": 2,
        "adapter_name": "adapter",
    }

    output = await backend._sample_payload(payload)

    assert len(backend.client.requests) == 2
    first_params = backend.client.requests[0][1]["sampling_params"]
    second_params = backend.client.requests[1][1]["sampling_params"]
    assert first_params["sampling_seed"] == 41
    assert second_params["sampling_seed"] == 42
    assert "seed" not in first_params
    assert first_params["max_new_tokens"] == 3
    assert backend.client.requests[0][1]["lora_path"] == "adapter"
    assert output["sequences"][0]["tokens"] == [123]
    assert output["prompt_logprobs"] == [None, -0.5]
    assert output["topk_prompt_logprobs"][1] == [(12, -0.1), (13, -0.2)]


@pytest.mark.asyncio
async def test_sample_payload_omits_sampling_seed_when_not_requested():
    backend = object.__new__(TinkerBackend)
    backend.client = _Client()
    backend.router_url = "http://router"
    payload = {
        "num_samples": 1,
        "prompt": [10],
        "sampling_params": {},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
        "adapter_name": None,
    }

    await backend._sample_payload(payload)

    params = backend.client.requests[0][1]["sampling_params"]
    assert "sampling_seed" not in params


@pytest.mark.asyncio
async def test_run_sample_refreshes_unpinned_adapter_before_generation():
    backend = object.__new__(TinkerBackend)
    backend.sample_semaphore = asyncio.Semaphore(1)
    completed = {}
    calls = []

    class _State:
        def complete_future(self, request_id, response):
            completed[request_id] = response

        def fail_future(self, request_id, error, category):
            raise AssertionError((request_id, error, category))

    async def load_adapter(adapter_name, adapter_path):
        calls.append(("load", adapter_name, adapter_path))

    async def sample_payload(payload):
        calls.append(("sample", payload["adapter_name"]))
        return {"type": "sample", "sequences": []}

    backend.state = _State()
    backend._load_sampler_adapter = load_adapter
    backend._sample_payload = sample_payload
    operation = Operation(
        request_id="request-1",
        kind="sample",
        model_id=None,
        payload={
            "adapter_name": "adapter-1",
            "adapter_path": "/tmp/adapter-1",
        },
    )

    await backend._run_sample(operation)

    assert calls == [
        ("load", "adapter-1", Path("/tmp/adapter-1")),
        ("sample", "adapter-1"),
    ]
    assert completed["request-1"]["type"] == "sample"
