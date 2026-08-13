"""Offline tests: no GPU, no network, no Daytona, no live model.

sglang_compat.py's translation: the request/response rewrite between
HUD's vLLM-flavoured token flags and stock sglang.
The HUD side of the seam is faked at exactly the boundary the real code
touches.
"""

from __future__ import annotations

import asyncio
import json

import pytest

hud = pytest.importorskip("hud", reason="pip install hud -- the recipe's one extra dependency")

from examples.experimental.hud.sglang_compat import _SglangTokenIds


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _CannedTransport:
    """Fake inner transport: captures the outgoing request, returns a canned
    sglang-shaped response."""

    def __init__(self, payload):
        self.payload = payload
        self.seen_body = None

    async def handle_async_request(self, request):
        import httpx

        self.seen_body = json.loads(request.content or b"{}")
        return httpx.Response(200, json=self.payload, request=request)


def _roundtrip(request_body, response_payload):
    import httpx

    shim = _SglangTokenIds()
    canned = _CannedTransport(response_payload)
    shim._inner = canned
    request = httpx.Request("POST", "http://x/v1/chat/completions", content=json.dumps(request_body).encode())
    response = _run(shim.handle_async_request(request))
    return canned.seen_body, json.loads(response.content)


def test_shim_rewrites_flags_and_synthesizes_token_ids():
    sent, got = _roundtrip(
        {"model": "m", "return_token_ids": True, "prompt_token_ids": [1, 2]},
        {"choices": [{"meta_info": {"output_token_logprobs": [[-0.1, 5, "a"], [-0.2, 6, "b"]]}}]},
    )
    assert "return_token_ids" not in sent and "prompt_token_ids" not in sent
    assert sent["return_prompt_token_ids"] is True and sent["return_meta_info"] is True
    assert got["choices"][0]["token_ids"] == [5, 6]


def test_shim_leaves_other_endpoints_alone():
    import httpx

    shim = _SglangTokenIds()
    canned = _CannedTransport({"data": []})
    shim._inner = canned
    request = httpx.Request("GET", "http://x/v1/models")
    _run(shim.handle_async_request(request))
    assert canned.seen_body == {}
