import importlib.util
import json

import httpx
import pytest
from aiohttp.test_utils import TestServer
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples
from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app


def normalized(payload):
    if isinstance(payload, dict):
        return {key: normalized(value) for key, value in payload.items() if key not in {"id", "created"}}
    if isinstance(payload, list):
        return [normalized(value) for value in payload]
    return payload


def response_payload(response):
    if response.headers["content-type"].startswith("text/event-stream"):
        return [
            normalized(json.loads(line[6:]))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    return normalized(response.json())


def test_client_install_has_no_miles_dependency():
    assert importlib.util.find_spec("miles") is None, "run the installed client tests with python -I"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,extra",
    [
        ("chat/completions", {}),
        ("chat/completions", {"stream": True}),
        ("chat/completions", {"stop": ["END"]}),
        ("chat/completions", {"stop": [42]}),
        ("chat/completions", {"response_format": {"type": "json_object"}}),
        ("messages", {}),
        ("messages", {"stream": True, "thinking": {"type": "enabled", "budget_tokens": 16}}),
    ],
)
async def test_capture_preserves_official_proxy_behavior(endpoint, extra, renderer, sampler_factory, capture_store):
    bare, backend = sampler_factory(), sampler_factory()
    official = TestServer(make_app(ProxyDeps(renderer, bare, "policy", default_max_tokens=32)))
    await official.start_server()
    try:
        async with serve_proxy(renderer, backend, model_label="policy", default_max_tokens=32) as port:
            async with httpx.AsyncClient() as http:
                body = {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 17,
                    "temperature": 0.7,
                    **extra,
                }
                path = f"/r/run/test/traj/0/v1/{endpoint}"
                expected = await http.post(
                    str(official.make_url(path.replace("/run/test/", "/run/official/"))), json=body
                )
                actual = await http.post(f"http://127.0.0.1:{port}{path}", json=body)
                assert actual.status_code == expected.status_code
                assert response_payload(actual) == response_payload(expected)
                assert backend.calls == bare.calls
        await flush_capture()
        rows = await read_samples(capture_store.base_url, "test", traj_idx=0)
        assert len(rows) == len(backend.calls)
        if rows:
            assert rows[0]["prompt_tokens"] == backend.calls[0][0]
            assert rows[0]["sampled_tokens"] == [1000]
    finally:
        await official.close()


@pytest.mark.asyncio
async def test_parse_failure_keeps_the_official_capture(monkeypatch, renderer, sampler_factory, capture_store):
    backend = sampler_factory()

    def fail_parse(tokens):
        raise ValueError("broken output parser")

    monkeypatch.setattr(renderer, "parse_response", fail_parse)
    async with serve_proxy(renderer, backend, model_label="policy", default_max_tokens=32) as port:
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"http://127.0.0.1:{port}/r/run/test/traj/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 500
    await flush_capture()
    rows = await read_samples(capture_store.base_url, "test", traj_idx=0)
    assert len(rows) == 1
    assert rows[0]["sampled_tokens"] == [1000]
