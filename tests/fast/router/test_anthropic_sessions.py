import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import router_env  # noqa: F401

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult
from miles.utils.types import Sample

# ruff: noqa: F811 -- imported pytest fixture names are injected as test arguments.


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_messages(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/messages", json=payload, timeout=10.0)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event_line, data_line = frame.splitlines()
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def _request(**overrides: object) -> dict:
    request = {
        "model": "mock-model",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hello"}],
    }
    request.update(overrides)
    return request


class TestAnthropicSessionRoute:
    def test_concurrent_turns_are_serialized_per_session(self, router_env, monkeypatch) -> None:
        session_id = _create_session(router_env.url)
        barrier = Barrier(4)
        monkeypatch.setattr(router_env.backend, "latency", 0.2)
        router_env.backend.reset_stats()

        def post_turn() -> requests.Response:
            barrier.wait()
            return _post_messages(router_env.url, session_id, _request())

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(post_turn) for _ in range(4)]
            responses = [future.result() for future in futures]

        assert all(response.status_code == 200 for response in responses)
        assert len(router_env.backend.request_log) == 4
        assert router_env.backend.max_concurrent == 1

    def test_different_sessions_remain_parallel(self, router_env, monkeypatch) -> None:
        session_ids = [_create_session(router_env.url) for _ in range(4)]
        barrier = Barrier(len(session_ids))
        monkeypatch.setattr(router_env.backend, "latency", 0.2)
        router_env.backend.reset_stats()

        def post_turn(session_id: str) -> requests.Response:
            barrier.wait()
            return _post_messages(router_env.url, session_id, _request())

        with ThreadPoolExecutor(max_workers=len(session_ids)) as pool:
            futures = [pool.submit(post_turn, session_id) for session_id in session_ids]
            responses = [future.result() for future in futures]

        assert all(response.status_code == 200 for response in responses)
        assert len(router_env.backend.request_log) == len(session_ids)
        assert router_env.backend.max_concurrent >= 2

    def test_session_sampling_overrides_win_over_claude_request(self, router_env) -> None:
        create = requests.post(
            f"{router_env.url}/sessions",
            json={"request_overrides": {"max_tokens": 128, "temperature": 0.4, "top_p": 0.8}},
            timeout=5.0,
        )
        session_id = create.json()["session_id"]

        response = _post_messages(
            router_env.url,
            session_id,
            _request(max_tokens=4096, temperature=1.0, top_p=1.0),
        )

        assert response.status_code == 200
        request = router_env.backend.request_log[-1]
        assert request["max_tokens"] == 128
        assert request["temperature"] == 0.4
        assert request["top_p"] == 0.8

    def test_session_rejects_non_sampling_overrides(self, router_env) -> None:
        response = requests.post(
            f"{router_env.url}/sessions",
            json={"request_overrides": {"messages": []}},
            timeout=5.0,
        )

        assert response.status_code == 400
        assert "unsupported session request overrides" in response.json()["error"]

        response = requests.post(
            f"{router_env.url}/sessions",
            json={"request_overrides": []},
            timeout=5.0,
        )
        assert response.status_code == 400
        assert response.json()["error"] == "request_overrides must be an object"

    def test_non_streaming_request_is_recorded_in_openai_form(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(
            router_env.url,
            session_id,
            _request(
                system="Be concise.",
                temperature=0.7,
                top_p=0.9,
                top_k=20,
            ),
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"][0]["type"] == "text"

        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"][:2] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
        assert record["request"]["temperature"] == 0.7
        assert record["request"]["top_p"] == 0.9
        assert record["request"]["top_k"] == 20
        assert record["request"]["logprobs"] is True
        assert record["request"]["return_meta_info"] is True
        assert "input_ids" in record["request"]
        assert "stream" not in record["request"]

    def test_streaming_response_has_anthropic_events_and_one_record(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(router_env.url, session_id, _request(stream=True))

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.text)
        assert [event_type for event_type, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2][1]["delta"]["type"] == "text_delta"
        assert events[-2][1]["delta"]["stop_reason"] == "end_turn"

        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 1
        assert "stream" not in router_env.backend.request_log[-1]

    def test_multi_turn_response_replay_passes_tito_prefix_check(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        first_request = _request()
        first = _post_messages(router_env.url, session_id, first_request)
        assert first.status_code == 200

        second_messages = [
            *first_request["messages"],
            {"role": "assistant", "content": first.json()["content"]},
            {"role": "user", "content": "continue"},
        ]
        second = _post_messages(router_env.url, session_id, _request(messages=second_messages))

        assert second.status_code == 200
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2

    def test_anthropic_turn_collects_a_nonempty_training_sample(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        original_chat_response = MockSGLangServer._compute_chat_completions_response

        def response_without_replay_buffers(self: MockSGLangServer, payload: dict) -> dict:
            response = original_chat_response(self, payload)
            meta_info = response["choices"][0]["meta_info"]
            meta_info.pop("routed_experts", None)
            meta_info.pop("indexer_topk", None)
            return response

        with patch.object(
            MockSGLangServer,
            "_compute_chat_completions_response",
            new=response_without_replay_buffers,
        ):
            response = _post_messages(router_env.url, session_id, _request())
            assert response.status_code == 200

            samples = requests.post(
                f"{router_env.url}/sessions/{session_id}/samples",
                json={},
                timeout=10.0,
            )

        assert samples.status_code == 200
        reply = decode_samples_and_merge_input_sample(samples.content, Sample())
        assert reply.empty_reason is None
        assert len(reply.samples) == 1
        [sample] = reply.samples
        assert sample.response_length > 0
        assert len(sample.rollout_log_probs) == sample.response_length

    def test_tool_use_and_result_round_trip_passes_tito_prefix_check(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        tools = [
            {
                "name": "bash",
                "description": "Run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ]

        def tool_call_process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text='<tool_call>\n{"name":"bash","arguments":{"command":"pwd"}}\n</tool_call>')

        first_request = _request(messages=[{"role": "user", "content": "show the directory"}], tools=tools)
        with patch.object(router_env.backend, "process_fn", new=tool_call_process_fn):
            first = _post_messages(router_env.url, session_id, first_request)
        assert first.status_code == 200
        [tool_use] = [block for block in first.json()["content"] if block["type"] == "tool_use"]

        second_messages = [
            *first_request["messages"],
            {"role": "assistant", "content": first.json()["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "/workspace"}],
            },
        ]
        with patch.object(router_env.backend, "process_fn", new=lambda prompt: ProcessResult(text="done")):
            second = _post_messages(
                router_env.url,
                session_id,
                _request(messages=second_messages, tools=tools),
            )

        assert second.status_code == 200
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2
        assert records[1]["request"]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": tool_use["id"],
            "content": "/workspace",
        }

    def test_invalid_request_uses_anthropic_error_envelope(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        response = _post_messages(router_env.url, session_id, _request(max_tokens=0))

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "max_tokens must be a positive integer"},
        }

    def test_upstream_error_uses_anthropic_envelope_and_is_not_recorded(self, router_env) -> None:
        session_id = _create_session(router_env.url)

        async def reject(self: MockSGLangServer, request: object, compute_fn: object) -> JSONResponse:
            return JSONResponse(content={"error": {"message": "context too long"}}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            response = _post_messages(router_env.url, session_id, _request())

        assert response.status_code == 400
        assert response.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "context too long"},
        }
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert records == []

    @pytest.mark.parametrize("stream", [False, True])
    def test_aborted_generation_returns_retryable_error(self, router_env, stream: bool) -> None:
        session_id = _create_session(router_env.url)
        original_chat_response = MockSGLangServer._compute_chat_completions_response

        def aborted_response(self: MockSGLangServer, payload: dict) -> dict:
            response = original_chat_response(self, payload)
            response["choices"][0]["finish_reason"] = "abort"
            return response

        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=aborted_response):
            response = _post_messages(router_env.url, session_id, _request(stream=stream))

        assert response.status_code == 529
        assert response.json() == {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "upstream generation was aborted"},
        }

    def test_retry_after_aborted_generation_passes_tito_prefix_check(self, router_env) -> None:
        session_id = _create_session(router_env.url)
        original_chat_response = MockSGLangServer._compute_chat_completions_response
        attempts = 0

        def abort_once(self: MockSGLangServer, payload: dict) -> dict:
            nonlocal attempts
            response = original_chat_response(self, payload)
            if attempts == 0:
                response["choices"][0]["finish_reason"] = "abort"
            attempts += 1
            return response

        first_request = _request()
        with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=abort_once):
            aborted = _post_messages(router_env.url, session_id, first_request)
            assert aborted.status_code == 529

            retried = _post_messages(router_env.url, session_id, first_request)
            assert retried.status_code == 200

            next_messages = [
                *first_request["messages"],
                {"role": "assistant", "content": retried.json()["content"]},
                {"role": "user", "content": "continue"},
            ]
            continued = _post_messages(
                router_env.url,
                session_id,
                _request(messages=next_messages),
            )

        assert continued.status_code == 200
        records = requests.get(f"{router_env.url}/sessions/{session_id}", timeout=5.0).json()["records"]
        assert len(records) == 2
        assert records[0]["response"]["choices"][0]["finish_reason"] == "stop"
