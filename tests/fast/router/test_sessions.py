"""Integration tests for session HTTP routes (create / get / delete / proxy)."""

import asyncio
import json
import re
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import requests
from fastapi.responses import JSONResponse

from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.openai_stream_client import stream_chat_completions
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_chat(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/chat/completions", json=payload, timeout=10.0)


def _parse_sse(body: str) -> list[str]:
    """Return the data payload of each SSE event, in order."""
    return [block[len("data: ") :] for block in body.split("\n\n") if block.startswith("data: ")]


@pytest.fixture(scope="class")
def router_env():
    """Create a standalone SessionServer with session routes and a mock backend."""

    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
            # R3 replay payloads: must reach the session record but never the
            # client-facing chat response (see _strip_replay_payloads).
            "routed_experts": [[0, 1], [2, 3]],
            "indexer_topk": [[4], [5]],
        }
        return response

    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response):
        with with_mock_server(process_fn=process_fn) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint="Qwen/Qwen3-0.6B",
                chat_template_path=None,
                apply_chat_template_kwargs={"enable_thinking": False},
                tito_model="default",
                tito_allowed_append_roles=["tool"],
                trajectory_manager="linear_trajectory",
                session_server_instance_id=uuid.uuid4().hex,
            )
            server_obj = SessionServer(args, backend_url=backend.url)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()

            url = f"http://127.0.0.1:{port}"

            try:
                yield SimpleNamespace(url=url, backend=backend)
            finally:
                server.stop()


class TestSessionRoutes:
    def test_health_reports_stable_instance_id(self, router_env):
        first = requests.get(f"{router_env.url}/health", timeout=5.0)
        second = requests.get(f"{router_env.url}/health", timeout=5.0)

        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert first_body["status"] == "ok"
        assert second_body["status"] == "ok"
        assert re.fullmatch(r"[0-9a-f]{32}", first_body["session_server_instance_id"])
        assert second_body["session_server_instance_id"] == first_body["session_server_instance_id"]

    def test_create_session(self, router_env):
        response = requests.post(f"{router_env.url}/sessions", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_get_session_initial_state(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["session_id"] == session_id
        assert data["records"] == []

    def test_get_session_not_found(self, router_env):
        response = requests.get(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"

    def test_delete_session(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        delete_resp = requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        assert delete_resp.status_code == 204
        assert delete_resp.text == ""

        assert requests.delete(f"{router_env.url}/sessions/{session_id}", timeout=5.0).status_code == 404

    def test_delete_session_not_found(self, router_env):
        response = requests.delete(f"{router_env.url}/sessions/nonexistent", timeout=5.0)
        assert response.status_code == 404
        assert response.json()["error"] == "session not found: session_id=nonexistent"


class TestSessionProxy:
    def test_proxy_chat_appends_record(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 1+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body
        assert body["choices"]

        get_resp = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0)
        records = get_resp.json()["records"]

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["status_code"] == 200

    def test_chat_malformed_json_body_returns_400(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"].startswith("invalid JSON body:")

    def test_chat_upstream_null_message_returns_502(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        fixture_response = MockSGLangServer._compute_chat_completions_response

        def null_message_response(self, payload: dict) -> dict:
            response = fixture_response(self, payload)
            response["choices"][0]["message"] = None
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=null_message_response):
            resp = requests.post(
                f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            )
        assert resp.status_code == 502
        assert "assistant message content is None" in resp.json()["error"]

    def test_chat_response_strips_replay_payloads_but_record_keeps_them(self, router_env):
        session_id = requests.post(f"{router_env.url}/sessions", timeout=5.0).json()["session_id"]

        payload = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "return_logprob": True,
        }
        resp = requests.post(
            f"{router_env.url}/sessions/{session_id}/v1/chat/completions",
            json=payload,
            timeout=10.0,
        )
        assert resp.status_code == 200
        client_meta = resp.json()["choices"][0]["meta_info"]
        assert "routed_experts" not in client_meta
        assert "indexer_topk" not in client_meta
        # Stripping must not swallow the rest of meta_info.
        assert "output_token_logprobs" in client_meta

        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        record_meta = record["response"]["choices"][0]["meta_info"]
        assert record_meta["routed_experts"] == [[0, 1], [2, 3]]
        assert record_meta["indexer_topk"] == [[4], [5]]


class TestChatFakeStreaming:
    """``stream: true`` is fake streaming: non-streaming backend call, single SSE chunk."""

    MESSAGES = [{"role": "user", "content": "What is 1+2?"}]

    def test_stream_single_chunk_matches_non_stream(self, router_env):
        non_stream_sid = _create_session(router_env.url)
        stream_sid = _create_session(router_env.url)

        non_stream = _post_chat(router_env.url, non_stream_sid, {"messages": self.MESSAGES}).json()

        resp = _post_chat(
            router_env.url,
            stream_sid,
            {"messages": self.MESSAGES, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert len(events) == 2
        assert events[1] == "[DONE]"

        chunk = json.loads(events[0])
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "mock-model"
        [chunk_choice] = chunk["choices"]
        expected_choice = non_stream["choices"][0]
        assert chunk_choice["delta"]["role"] == "assistant"
        assert chunk_choice["delta"]["content"] == expected_choice["message"]["content"]
        assert chunk_choice["finish_reason"] == expected_choice["finish_reason"] == "stop"
        # Training payloads never reach the chunk; they live on the session record.
        assert "meta_info" not in chunk_choice
        assert "routed_experts" not in resp.text
        assert "indexer_topk" not in resp.text

        # Neither the backend nor the record sees the stream flags.
        backend_payload = router_env.backend.request_log[-1]
        assert "stream" not in backend_payload
        assert "stream_options" not in backend_payload

        stream_record = requests.get(f"{router_env.url}/sessions/{stream_sid}", timeout=5.0).json()["records"][0]
        non_stream_record = requests.get(f"{router_env.url}/sessions/{non_stream_sid}", timeout=5.0).json()["records"][
            0
        ]
        assert stream_record["request"] == non_stream_record["request"]
        assert (
            stream_record["response"]["choices"][0]["message"]
            == non_stream_record["response"]["choices"][0]["message"]
        )
        assert (
            stream_record["response"]["choices"][0]["meta_info"]
            == non_stream_record["response"]["choices"][0]["meta_info"]
        )

    def test_stream_tool_calls_single_chunk_with_index(self, router_env):
        session_id = _create_session(router_env.url)

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text=(
                    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
                )
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            },
        ]

        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "tools": tools, "stream": True})

        assert resp.status_code == 200
        chunk = json.loads(_parse_sse(resp.text)[0])
        [chunk_choice] = chunk["choices"]
        assert chunk_choice["finish_reason"] == "tool_calls"
        tool_calls = chunk_choice["delta"]["tool_calls"]
        assert [tool_call["index"] for tool_call in tool_calls] == [0, 1]
        assert [tool_call["function"]["name"] for tool_call in tool_calls] == ["get_weather", "get_time"]
        for tool_call in tool_calls:
            json.loads(tool_call["function"]["arguments"])

    def test_stream_passes_through_usage_and_length_finish_reason(self, router_env):
        session_id = _create_session(router_env.url)
        fixture_response = MockSGLangServer._compute_chat_completions_response

        def length_with_usage(self, payload: dict) -> dict:
            response = fixture_response(self, payload)
            response["choices"][0]["finish_reason"] = "length"
            response["usage"] = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=length_with_usage):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": True})

        assert resp.status_code == 200
        chunk = json.loads(_parse_sse(resp.text)[0])
        assert chunk["choices"][0]["finish_reason"] == "length"
        assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

    def test_stream_backend_error_passes_through_json(self, router_env):
        session_id = _create_session(router_env.url)

        async def reject(self, request, compute_fn):
            return JSONResponse(content={"error": "context too long"}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": True})

        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"error": "context too long"}
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert records == []

    def test_stream_false_keeps_json_response(self, router_env):
        session_id = _create_session(router_env.url)
        resp = _post_chat(router_env.url, session_id, {"messages": self.MESSAGES, "stream": False})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["choices"][0]["message"]["content"]

    def test_streaming_client_multi_turn_passes_prefix_check(self, router_env):
        """The rebuilt assistant message must survive the next turn's TITO prefix check."""
        session_id = _create_session(router_env.url)
        url = f"{router_env.url}/sessions/{session_id}/v1/chat/completions"

        async def run():
            async with httpx.AsyncClient(timeout=10) as client:
                first = await stream_chat_completions(client, url, {"messages": self.MESSAGES}, label="turn 1")
                messages = [
                    *self.MESSAGES,
                    first["choices"][0]["message"],
                    {"role": "tool", "content": "ok", "tool_call_id": "t0"},
                ]
                second = await stream_chat_completions(client, url, {"messages": messages}, label="turn 2")
                return first, second

        first, second = asyncio.run(run())
        assert first["choices"][0]["message"]["content"]
        assert second["choices"][0]["message"]["content"]
        # Turn 2 extended (not rolled back) the session: both records are kept.
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2

    def test_streaming_client_rebuilds_tool_calls_exactly(self, router_env):
        """Rebuilt tool_calls must dict-equal the record's stored assistant tool_calls."""
        session_id = _create_session(router_env.url)
        url = f"{router_env.url}/sessions/{session_id}/v1/chat/completions"

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text=(
                    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n'
                    '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
                )
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            },
        ]

        async def run():
            async with httpx.AsyncClient(timeout=10) as client:
                return await stream_chat_completions(
                    client, url, {"messages": self.MESSAGES, "tools": tools}, label="tool turn"
                )

        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            response = asyncio.run(run())

        rebuilt = response["choices"][0]["message"]["tool_calls"]
        record = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"][0]
        stored = record["response"]["choices"][0]["message"]["tool_calls"]
        assert rebuilt == stored

    def test_openai_sdk_accumulates_fake_stream(self, router_env):
        openai = pytest.importorskip("openai")

        non_stream_sid = _create_session(router_env.url)
        stream_sid = _create_session(router_env.url)

        expected = _post_chat(router_env.url, non_stream_sid, {"messages": self.MESSAGES}).json()
        expected_content = expected["choices"][0]["message"]["content"]

        client = openai.OpenAI(
            base_url=f"{router_env.url}/sessions/{stream_sid}/v1", api_key="not-used", max_retries=0
        )
        stream = client.chat.completions.create(model="mock-model", messages=self.MESSAGES, stream=True)
        content = ""
        finish_reason = None
        for chunk in stream:
            [choice] = chunk.choices
            if choice.delta.content:
                content += choice.delta.content
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        assert content == expected_content
        assert finish_reason == "stop"
