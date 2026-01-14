import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

from miles.utils.http_utils import post
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, start_mock_server


def create_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode = lambda ids, **kwargs: f"decoded:{','.join(map(str, ids))}"
    tokenizer.encode = lambda text, **kwargs: [ord(c) % 1000 for c in text[:10]]
    return tokenizer


def test_basic_server_start_stop():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()
        assert server.port > 0
        assert f"http://{server.host}:{server.port}" == server.url
    finally:
        server.stop()


def test_generate_endpoint_basic():
    tokenizer = create_mock_tokenizer()

    def process_fn(prompt: str) -> str:
        return f"Response to: {prompt[:20]}"

    server = MockSGLangServer(tokenizer=tokenizer, process_fn=process_fn, finish_reason="stop", cached_tokens=2)
    try:
        server.start()

        input_ids = [1, 2, 3, 4, 5]
        response = httpx.post(
            f"{server.url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert "Response to:" in data["text"]
        assert "meta_info" in data
        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert data["meta_info"]["prompt_tokens"] == len(input_ids)
        assert data["meta_info"]["cached_tokens"] == 2
        assert data["meta_info"]["completion_tokens"] > 0

        assert len(server.requests) == 1
        assert server.requests[0]["input_ids"] == input_ids
    finally:
        server.stop()


def test_finish_reason_stop():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "stop"
        assert "length" not in data["meta_info"]["finish_reason"]
    finally:
        server.stop()


def test_finish_reason_length():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="length")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "length"
        assert "length" in data["meta_info"]["finish_reason"]
        assert data["meta_info"]["finish_reason"]["length"] == data["meta_info"]["completion_tokens"]
    finally:
        server.stop()


def test_finish_reason_abort():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="abort")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["finish_reason"]["type"] == "abort"
    finally:
        server.stop()


def test_return_logprob():
    tokenizer = create_mock_tokenizer()
    tokenizer.encode = lambda text, **kwargs: [100, 200, 300]

    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "output_token_logprobs" in data["meta_info"]
        logprobs = data["meta_info"]["output_token_logprobs"]
        assert isinstance(logprobs, list)
        assert len(logprobs) == 3
        assert isinstance(logprobs[0], list)
        assert len(logprobs[0]) == 2
        assert isinstance(logprobs[0][0], float)
        assert logprobs[0][1] == 100
        assert logprobs[1][1] == 200
        assert logprobs[2][1] == 300
    finally:
        server.stop()


def test_return_routed_experts():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3, 4, 5], "sampling_params": {}, "return_routed_experts": True},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert "routed_experts" in data["meta_info"]
        routed_experts_b64 = data["meta_info"]["routed_experts"]
        assert isinstance(routed_experts_b64, str)
    finally:
        server.stop()


def test_request_recording():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        request1 = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.7}}
        request2 = {"input_ids": [4, 5, 6], "sampling_params": {"temperature": 0.9}, "return_logprob": True}

        httpx.post(f"{server.url}/generate", json=request1, timeout=5.0)
        httpx.post(f"{server.url}/generate", json=request2, timeout=5.0)

        assert len(server.requests) == 2
        assert server.requests[0] == request1
        assert server.requests[1] == request2

        server.clear_requests()
        assert len(server.requests) == 0
    finally:
        server.stop()


def test_weight_version():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop", weight_version="v1.0")
    try:
        server.start()

        response = httpx.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["weight_version"] == "v1.0"
    finally:
        server.stop()


def test_context_manager():
    tokenizer = create_mock_tokenizer()

    def process_fn(prompt: str) -> str:
        return "Context test response"

    with start_mock_server(tokenizer=tokenizer, process_fn=process_fn, finish_reason="stop") as server:
        response = httpx.post(f"{server.url}/generate", json={"input_ids": [1, 2, 3], "sampling_params": {}}, timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Context test response"


def test_prompt_tokens_calculated_from_input_ids():
    tokenizer = create_mock_tokenizer()
    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        input_ids = [10, 20, 30, 40, 50, 60, 70]
        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": input_ids, "sampling_params": {}},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["prompt_tokens"] == len(input_ids)
    finally:
        server.stop()


def test_completion_tokens_calculated_from_output():
    tokenizer = create_mock_tokenizer()
    tokenizer.encode = lambda text, **kwargs: [1, 2, 3, 4, 5]

    server = MockSGLangServer(tokenizer=tokenizer, finish_reason="stop")
    try:
        server.start()

        response = httpx.post(
            f"{server.url}/generate",
            json={"input_ids": [1, 2, 3], "sampling_params": {}},
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["meta_info"]["completion_tokens"] == 5
    finally:
        server.stop()


def test_process_fn_receives_decoded_prompt():
    tokenizer = create_mock_tokenizer()
    received_prompts = []

    def process_fn(prompt: str) -> str:
        received_prompts.append(prompt)
        return "response"

    server = MockSGLangServer(tokenizer=tokenizer, process_fn=process_fn, finish_reason="stop")
    try:
        server.start()

        input_ids = [1, 2, 3]
        httpx.post(f"{server.url}/generate", json={"input_ids": input_ids, "sampling_params": {}}, timeout=5.0)

        assert len(received_prompts) == 1
        assert received_prompts[0] == "decoded:1,2,3"
    finally:
        server.stop()


def test_async_post():
    tokenizer = create_mock_tokenizer()

    def process_fn(prompt: str) -> str:
        return "Async test response"

    async def _run():
        response = await post(url, payload)
        assert response["text"] == "Async test response"
        assert response["meta_info"]["finish_reason"]["type"] == "stop"
        assert len(server.requests) == 1

    with start_mock_server(tokenizer=tokenizer, process_fn=process_fn, finish_reason="stop") as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3], "sampling_params": {}}
        asyncio.run(_run())


def test_async_with_logprob():
    tokenizer = create_mock_tokenizer()
    tokenizer.encode = lambda text, **kwargs: [100, 200]

    async def _run():
        response = await post(url, payload)
        assert "output_token_logprobs" in response["meta_info"]
        logprobs = response["meta_info"]["output_token_logprobs"]
        assert len(logprobs) == 2
        assert logprobs[0][1] == 100
        assert logprobs[1][1] == 200

    with start_mock_server(tokenizer=tokenizer, finish_reason="stop") as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True}
        asyncio.run(_run())


def test_async_with_routed_experts():
    tokenizer = create_mock_tokenizer()

    async def _run():
        response = await post(url, payload)
        assert "routed_experts" in response["meta_info"]
        routed_experts_b64 = response["meta_info"]["routed_experts"]
        assert isinstance(routed_experts_b64, str)

    with start_mock_server(tokenizer=tokenizer, finish_reason="stop") as server:
        url = f"{server.url}/generate"
        payload = {"input_ids": [1, 2, 3, 4, 5], "sampling_params": {}, "return_routed_experts": True}
        asyncio.run(_run())
