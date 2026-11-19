"""Describe and release use the generic engine client's JSON POST transport."""

from unittest.mock import AsyncMock

import httpx

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.gpu_delta.client import (
    clear_gpu_delta_state,
    get_gpu_delta_info,
)
from miles.utils.http_utils import GeneralHttpClientProvider


async def test_receiver_controls_use_the_engine_client(monkeypatch):
    server_url = "http://fake-host:1234"
    transport = AsyncMock()
    transport.post.return_value = httpx.Response(200, json={}, request=httpx.Request("POST", server_url))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: transport)
    client = SGLangApiClient(server_url=server_url)
    await get_gpu_delta_info(client, "engine-1")
    await clear_gpu_delta_state(client)
    assert [(call.args[0], call.kwargs["json"]) for call in transport.post.await_args_list] == [
        (f"{server_url}/get_gpu_delta_info", {"engine_id": "engine-1"}),
        (f"{server_url}/clear_gpu_delta_state", {}),
    ]
