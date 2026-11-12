import copy
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import responses_adapter
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_WEATHER_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}\n</tool_call>'
_PATCH_TEXT = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"
_PATCH_CALL = (
    "<tool_call>\n" + json.dumps({"name": "apply_patch", "arguments": {"input": _PATCH_TEXT}}) + "\n</tool_call>"
)
_WEATHER_TOOL = {"type": "function", "name": "get_weather", "description": "weather", "parameters": {"type": "object"}}
_PATCH_TOOL = {"type": "custom", "name": "apply_patch", "description": "edit files", "format": {"type": "text"}}
_HOSTED_TOOLS = [
    {"type": "web_search"},
    {"type": "namespace", "name": "agents", "tools": [{"type": "function", "name": "spawn"}]},
]
_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


def _process_fn(prompt: str) -> ProcessResult:
    if "sunny" in prompt or "patched" in prompt:
        return ProcessResult(text="final-answer", finish_reason="stop")
    if "use the weather tool" in prompt:
        return ProcessResult(text=_WEATHER_CALL, finish_reason="stop")
    if "apply a patch" in prompt:
        return ProcessResult(text=_PATCH_CALL, finish_reason="stop")
    return ProcessResult(text="responses-echo", finish_reason="stop")


@contextmanager
def _responses_env(extra_args: dict | None):
    with with_mock_server(process_fn=_process_fn) as backend:
        config = make_session_server_config(
            backend_url=backend.url,
            timeout=30,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            instance_id=uuid.uuid4().hex,
            **(extra_args or {}),
        )
        port = find_available_port(32000)
        server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


@pytest.fixture(scope="module", params=["v1", "v2"])
def env(request):
    with _responses_env(_V2_ARGS if request.param == "v2" else None) as env:
        yield env


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", json={}, timeout=5.0).json()["session_id"]


def _post(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/responses", json=payload, timeout=30.0)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _sse_events(body: str) -> list[dict]:
    events = []
    for block in filter(str.strip, body.split("\n\n")):
        event_line, data_line = block.split("\n", 1)
        event = json.loads(data_line.removeprefix("data: "))
        assert event_line == f"event: {event['type']}"
        events.append(event)
    return events


def _assert_prefix_reused(records: list[dict]) -> None:
    assert len(records) == 2
    first_ids = records[0]["request"]["input_ids"]
    assert records[1]["request"]["input_ids"][: len(first_ids)] == first_ids


class TestResponsesRoute:
    def test_text_turn_records_canonical_chat_request(self, env):
        session_id = _create_session(env.url)
        developer = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]}
        resp = _post(env.url, session_id, {"model": "m", "instructions": "sys", "input": [developer, _user("hello")]})

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response" and body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "responses-echo"
        assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0
        [record] = _records(env.url, session_id)
        assert body["id"] == record["response"]["id"]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "Developer instructions:\ndev"},
            {"role": "user", "content": "hello"},
        ]

    def test_string_input_and_stream_end_with_response_completed(self, env):
        session_id = _create_session(env.url)
        resp = _post(
            env.url, session_id, {"model": "m", "input": "hello", "stream": True, "text": {"format": {"type": "text"}}}
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.text)
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert [event["sequence_number"] for event in events] == list(range(len(events)))
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == "responses-echo"
        assert _records(env.url, session_id)[0]["request"]["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.parametrize("prompt", ["hello", "please use the weather tool"])
    def test_stream_deltas_accumulate_to_the_final_output(self, env, prompt):
        session_id = _create_session(env.url)
        resp = _post(env.url, session_id, {"model": "m", "input": prompt, "tools": [_WEATHER_TOOL], "stream": True})

        events = _sse_events(resp.text)
        items = {}
        for event in events:
            index = event.get("output_index")
            if event["type"] == "response.output_item.added":
                items[index] = copy.deepcopy(event["item"])
            elif event["type"] == "response.content_part.added":
                items[index]["content"].append(copy.deepcopy(event["part"]))
            elif event["type"] == "response.output_text.delta":
                items[index]["content"][event["content_index"]]["text"] += event["delta"]
            elif event["type"] == "response.function_call_arguments.delta":
                items[index]["arguments"] += event["delta"]
        final = events[-1]["response"]["output"]
        assert [items[index] | {"status": item["status"]} for index, item in enumerate(final)] == final

    def test_function_call_round_trip_reuses_prefix_under_strict_matcher(self, env):
        session_id = _create_session(env.url)
        tools = [_WEATHER_TOOL, *_HOSTED_TOOLS]
        first = _post(
            env.url, session_id, {"model": "m", "input": [_user("please use the weather tool")], "tools": tools}
        )

        assert first.status_code == 200
        [call] = first.json()["output"]
        assert call["type"] == "function_call" and call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"city": "Paris", "unit": "C"}
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": "sunny"}
        replay = [_user("please use the weather tool"), call, output]
        second = _post(env.url, session_id, {"model": "m", "input": replay, "tools": tools})

        assert second.status_code == 200
        assert second.json()["output"][0]["content"][0]["text"] == "final-answer"
        records = _records(env.url, session_id)
        _assert_prefix_reused(records)
        assert [tool["function"]["name"] for tool in records[1]["request"]["tools"]] == ["get_weather"]
        assert records[1]["request"]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": call["call_id"],
            "content": "sunny",
        }

    def test_custom_tool_maps_to_one_string_function_and_back(self, env):
        session_id = _create_session(env.url)
        payload = {"model": "m", "input": [_user("apply a patch")], "tools": [_PATCH_TOOL], "tool_choice": "required"}
        first = _post(env.url, session_id, {**payload, "stream": True})

        assert first.status_code == 200
        [call] = _sse_events(first.text)[-1]["response"]["output"]
        assert call["type"] == "custom_tool_call" and call["input"] == _PATCH_TEXT
        output = {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "patched"}
        second = _post(env.url, session_id, {**payload, "input": [*payload["input"], call, output]})

        assert second.status_code == 200
        records = _records(env.url, session_id)
        _assert_prefix_reused(records)
        [tool] = records[0]["request"]["tools"]
        assert tool["function"]["parameters"]["required"] == ["input"]
        assert records[0]["request"]["tool_choice"] == "required"
        replayed = records[1]["request"]["messages"][1]["tool_calls"][0]["function"]
        assert json.loads(replayed["arguments"]) == {"input": _PATCH_TEXT}

    def test_reasoning_round_trips_through_encrypted_content(self, env, monkeypatch):
        original = env.backend._compute_chat_completions_response

        def with_reasoning(payload):
            response = original(payload)
            response["choices"][0]["message"]["reasoning_content"] = "ponder\n"
            return response

        monkeypatch.setattr(env.backend, "_compute_chat_completions_response", with_reasoning)
        session_id = _create_session(env.url)
        first = _post(env.url, session_id, {"model": "m", "input": [_user("first")]})

        assert first.status_code == 200
        reasoning, message = first.json()["output"]
        assert reasoning["type"] == "reasoning" and reasoning["encrypted_content"]
        replayed_reasoning = {**reasoning, "summary": []}
        second = _post(
            env.url, session_id, {"model": "m", "input": [_user("first"), replayed_reasoning, message, _user("again")]}
        )

        assert second.status_code == 200
        records = _records(env.url, session_id)
        _assert_prefix_reused(records)
        assert records[1]["request"]["messages"][1] == {
            "role": "assistant",
            "content": "responses-echo",
            "reasoning_content": "ponder\n",
        }

    @pytest.mark.parametrize(
        "payload",
        [
            {"input": "hello", "previous_response_id": "resp_1"},
            {"input": "hello", "text": {"format": {"type": "json_schema", "name": "x", "schema": {}, "strict": True}}},
            {"input": "hello", "text": {"format": {"type": "json_object"}}},
            {"input": [{"type": "item_reference", "id": "x"}]},
            {"input": [{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "u"}]}]},
        ],
    )
    def test_unsupported_request_is_400_without_record(self, env, payload):
        session_id = _create_session(env.url)
        resp = _post(env.url, session_id, {"model": "m", **payload})

        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert _records(env.url, session_id) == []

    def test_unavailable_sglang_helpers_return_501(self, env):
        session_id = _create_session(env.url)
        with patch.object(responses_adapter, "encode_reasoning_state", None):
            resp = _post(env.url, session_id, {"model": "m", "input": "hello"})

        assert resp.status_code == 501
        assert _records(env.url, session_id) == []

    def test_sglang_error_body_keeps_its_message(self, env):
        session_id = _create_session(env.url)
        error = {"object": "error", "message": "Requested token count exceeds", "type": "BadRequestError", "code": 400}

        async def reject(self, request, compute_fn):
            return JSONResponse(content={**error, "param": None}, status_code=400)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            resp = _post(env.url, session_id, {"model": "m", "input": "hello"})

        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Requested token count exceeds"
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_unknown_session_wears_responses_error(self, env):
        resp = _post(env.url, "missing", {"model": "m", "input": "hello"})

        assert resp.status_code == 404
        assert resp.json()["error"]["message"]
