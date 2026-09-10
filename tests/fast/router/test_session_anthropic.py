"""Integration tests for ``POST /sessions/{session_id}/v1/messages``.

The Anthropic route must reuse the OpenAI session path end to end: canonical
OpenAI ``SessionRecord``s, TITO ``input_ids``, matcher semantics, and the
same commit/skip decisions — clients only ever see Anthropic wire shapes.
"""

import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import convert_to_chat_completion_request
from tests.fast.fixtures.session_fixtures import make_session_server_config

import miles.rollout.session.core as core_module
import miles.rollout.session.v2.core as v2_core_module
from miles.rollout.session import sessions as sessions_module
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils.message_matcher_hub import resolve_session_message_matcher, strict_message_matches
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# Two-key arguments: the qwen25 parser re-serializes them in this key order,
# so a replay whose input object uses the REVERSED key order re-serializes to
# a different spelling — the matcher-gate scenario from the design doc.
_TOOL_CALL_TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}\n</tool_call>'
_TOOLS = [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {}}}]


def _process_fn(prompt: str) -> ProcessResult:
    if "sunny" in prompt:
        return ProcessResult(text="final-answer", finish_reason="stop")
    if "use the weather tool" in prompt:
        return ProcessResult(text=_TOOL_CALL_TEXT, finish_reason="stop")
    return ProcessResult(text="anthropic-echo", finish_reason="stop")


@contextmanager
def _anthropic_env(extra_args: dict | None = None, *, latency: float = 0.0):
    # The mock backend already emits choice.meta_info with
    # output_token_logprobs/completion_tokens in the session-server format.
    with with_mock_server(process_fn=_process_fn, latency=latency) as backend:
        config = make_session_server_config(
            backend_url=backend.url,
            timeout=30,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            instance_id=uuid.uuid4().hex,
            **({"pause_generation_mode": "retract"} | (extra_args or {})),
        )
        server_obj = SessionServer(config)
        port = find_available_port(31000)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env(request):
    with _anthropic_env(_V2_ARGS if request.param == "v2" else None) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env_loose(request):
    extra = {"session_message_matcher": "loose_tool_call", **(_V2_ARGS if request.param == "v2" else {})}
    with _anthropic_env(extra) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_messages(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/messages", json=payload, timeout=30.0)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _payload(messages, **extra) -> dict:
    return {"model": "claude-test", "max_tokens": 64, "messages": messages, **extra}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n", 1)
        assert event_line.startswith("event: ") and data_line.startswith("data: ")
        events.append((event_line[len("event: ") :], json.loads(data_line[len("data: ") :])))
    return events


@pytest.mark.parametrize(
    "missing_module",
    [
        "sglang.srt.entrypoints.anthropic",
        "sglang.srt.entrypoints.anthropic.utils",
        "sglang.srt.entrypoints.anthropic.serving",
        "sglang.srt.entrypoints.anthropic.protocol",
    ],
)
def test_sessions_module_imports_without_optional_sglang_anthropic_helpers(missing_module: str) -> None:
    script = (
        "import sys\n"
        f"missing = {missing_module!r}\n"
        "sys.modules[missing] = None\n"
        "from miles.rollout.session import anthropic_adapter, sessions\n"
        "if missing.endswith(('.utils', 'anthropic')):\n"
        "    assert sessions.anthropic_utils is None\n"
        "if missing.endswith(('.serving', 'anthropic')):\n"
        "    assert sessions.convert_response is None\n"
        "    assert sessions.convert_to_chat_completion_request is None\n"
        "if missing.endswith(('.protocol', 'anthropic')):\n"
        "    assert anthropic_adapter.anthropic_adapter_available() is False\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


class TestAnthropicRoute:
    def test_unavailable_sglang_adapter_returns_501_without_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        with patch.object(sessions_module, "anthropic_utils", None):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))

        assert resp.status_code == 501
        assert resp.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The installed SGLang does not support the Anthropic Messages adapter",
            },
        }
        assert _records(anthropic_env.url, session_id) == []

    def test_health_reports_live_intermediate_system_capability(self, anthropic_env):
        body = requests.get(f"{anthropic_env.url}/health", timeout=5.0).json()
        assert body["anthropic_intermediate_system_supported"] is True

    def test_non_stream_text_creates_canonical_openai_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        # Non-stream JSON keeps the backend response model (frozen behavior).
        assert body["model"] == "mock-model"
        assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0

        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        records = snapshot["records"]
        assert len(records) == 1
        record = records[0]
        assert body["id"] == record["response"]["id"]
        if anthropic_env.version == "v2":
            assert body["id"] == snapshot["metadata"]["tree"]["nodes"][0]["response_id"]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert isinstance(record["request"]["input_ids"], list) and record["request"]["input_ids"]
        assert record["response"]["object"] == "chat.completion"

    def test_assistant_thinking_history_becomes_canonical_reasoning_content(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "ponder\n"},
                            {"type": "text", "text": "hello"},
                        ],
                    },
                    {"role": "user", "content": "again"},
                ]
            ),
        )
        assert resp.status_code == 200

        [record] = _records(anthropic_env.url, session_id)
        assistant = record["request"]["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "hello"
        assert assistant["reasoning_content"] == "ponder\n"

    @pytest.mark.parametrize(
        ("remaining_block", "expected_content"),
        [
            (None, ""),
            (
                {"type": "tool_use", "id": "call1", "name": "get_weather", "input": {"city": "Paris"}},
                None,
            ),
        ],
    )
    def test_assistant_thinking_strip_preserves_the_rest_of_the_message(self, remaining_block, expected_content):
        content = [{"type": "thinking", "thinking": "first"}, {"type": "thinking", "thinking": "second"}]
        if remaining_block is not None:
            content.append(remaining_block)
        request = AnthropicMessagesRequest.model_validate(_payload([{"role": "assistant", "content": content}]))

        conversion_request, reasoning_history = sessions_module._strip_anthropic_reasoning_history(request)
        converted = convert_to_chat_completion_request(conversion_request, merge_inline_system=True)
        dumped = converted.model_dump(mode="json", exclude_none=True, exclude_unset=True, by_alias=True)
        sessions_module._restore_anthropic_reasoning_history(dumped, reasoning_history)

        [assistant] = dumped["messages"]
        assert assistant["reasoning_content"] == "first\nsecond"
        if expected_content is not None:
            assert assistant["content"] == expected_content
        else:
            assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_assistant_thinking_response_replays_as_the_same_canonical_history(self, anthropic_env, monkeypatch):
        original = anthropic_env.backend._compute_chat_completions_response

        def with_reasoning(payload):
            response = original(payload)
            response["choices"][0]["message"]["reasoning_content"] = "ponder\n"
            return response

        monkeypatch.setattr(anthropic_env.backend, "_compute_chat_completions_response", with_reasoning)
        session_id = _create_session(anthropic_env.url)
        first = _post_messages(
            anthropic_env.url,
            session_id,
            _payload([{"role": "user", "content": "first"}]),
        )
        assert first.status_code == 200
        assert first.json()["content"][0] == {"type": "thinking", "thinking": "ponder\n"}

        second = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": first.json()["content"]},
                    {"role": "user", "content": "again"},
                ]
            ),
        )
        assert second.status_code == 200

        records = _records(anthropic_env.url, session_id)
        assert len(records) == 2
        stored = records[0]["response"]["choices"][0]["message"]
        replayed = records[1]["request"]["messages"][1]
        assert strict_message_matches(stored, replayed)
        assert replayed["reasoning_content"] == "ponder\n"

    def test_intermediate_system_stays_in_place(self, anthropic_env):
        leading_system = "initial policy"
        intermediate_system = "answer briefly"
        session_id = _create_session(anthropic_env.url)
        first = _post_messages(
            anthropic_env.url,
            session_id,
            _payload([{"role": "user", "content": "first"}], system=leading_system),
        )
        assert first.status_code == 200

        second = _post_messages(
            anthropic_env.url,
            session_id,
            _payload(
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": first.json()["content"]},
                    {"role": "system", "content": intermediate_system},
                    {"role": "user", "content": "again"},
                ],
                system=leading_system,
            ),
        )
        assert second.status_code == 200

        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        first_record, second_record = snapshot["records"]
        messages = second_record["request"]["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "system",
            "user",
        ]
        assert [message for message in messages if message["role"] == "system"] == [
            {"role": "system", "content": leading_system},
            {"role": "system", "content": intermediate_system},
        ]

        completion_ids = [
            item[1] for item in first_record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]
        ]
        expected_prefix = first_record["request"]["input_ids"] + completion_ids
        check_len = max(0, len(expected_prefix) - snapshot["metadata"]["max_trim_tokens"])
        assert second_record["request"]["input_ids"][:check_len] == expected_prefix[:check_len]
        decoded_suffix = load_tokenizer("Qwen/Qwen3-0.6B", trust_remote_code=True).decode(
            second_record["request"]["input_ids"][check_len:]
        )
        assert decoded_suffix.index(intermediate_system) < decoded_suffix.index("again")
        if anthropic_env.version == "v2":
            assert [node["parent"] for node in snapshot["metadata"]["tree"]["nodes"]] == [None, 0]

    def test_stream_returns_eager_fake_sse(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        message_start = events[0][1]["message"]
        # The stream's message model is the original Anthropic request model,
        # never the backend's served-model alias.
        assert message_start["model"] == "claude-test"
        assert message_start["usage"]["output_tokens"] == 0 and message_start["usage"]["input_tokens"] > 0
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "anthropic-echo"}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert events[4][1]["usage"]["output_tokens"] > 0
        snapshot = requests.get(f"{anthropic_env.url}/sessions/{session_id}", timeout=5.0).json()
        [record] = snapshot["records"]
        assert message_start["id"] == record["response"]["id"]
        if anthropic_env.version == "v2":
            assert message_start["id"] == snapshot["metadata"]["tree"]["nodes"][0]["response_id"]

    def test_parity_with_equivalent_openai_request(self, anthropic_env):
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]
        assert (
            anthropic_record["response"]["choices"][0]["message"] == openai_record["response"]["choices"][0]["message"]
        )

    def test_parity_with_equivalent_openai_request_with_tools(self, anthropic_env):
        """Tool-bearing parity pins the canonical tools spelling: the chat
        template renders tools in dict key order, so byte-identical TITO
        input_ids require the OpenAI client to write the codec's field order
        (function.description before function.name, no extra defaults)."""
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hi"}], tools=_TOOLS)
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "description": "weather",
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["tools"] == openai_record["request"]["tools"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]

    def test_unexpected_core_exception_wears_anthropic_envelope(self, anthropic_env):
        """Unexpected (non-SessionError) processing failures keep the frozen
        wire behavior: a scrubbed generic Anthropic api_error envelope, not
        the framework's text/plain 500 page."""
        session_id = _create_session(anthropic_env.url)
        # patch.object on pre-imported modules: a string-target patch would
        # import v2.core lazily INSIDE the first patch's window and bake the
        # first mock into its from-import binding, leaking it after restore.
        with (
            patch.object(core_module, "extract_completion", side_effect=RuntimeError("boom")),
            patch.object(v2_core_module, "extract_completion", side_effect=RuntimeError("boom")),
        ):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        assert _records(anthropic_env.url, session_id) == []

    def test_nan_sampling_param_maps_to_invalid_request_error(self, anthropic_env):
        """json.loads admits the non-standard NaN literal and the wire models
        accept it, but ``_render_json`` (allow_nan=False) rejects it — that
        failure must stay a 400 request error, never a plain-text 500."""
        session_id = _create_session(anthropic_env.url)
        raw = (
            b'{"model": "claude-test", "max_tokens": 64, "temperature": NaN,'
            b' "messages": [{"role": "user", "content": "x"}]}'
        )
        resp = requests.post(
            f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
            data=raw,
            headers={"content-type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert _records(anthropic_env.url, session_id) == []

    def test_error_envelopes_and_no_record_on_validation_failure(self, anthropic_env):
        payload = _payload([{"role": "user", "content": "hello"}])
        resp = _post_messages(anthropic_env.url, "nonexistent", payload)
        assert resp.status_code == 404
        assert resp.json() == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "session not found: session_id=nonexistent"},
        }

        session_id = _create_session(anthropic_env.url)
        cases = [
            (
                b"{not json",
                "invalid JSON body: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            (
                json.dumps({"model": "claude-test", "messages": [{"role": "user", "content": "x"}]}).encode(),
                None,
            ),
        ]
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            for raw, exact_message in cases:
                resp = requests.post(
                    f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
                    data=raw,
                    headers={"content-type": "application/json"},
                    timeout=10.0,
                )
                assert resp.status_code == 400, raw
                error = resp.json()["error"]
                assert error["type"] == "invalid_request_error"
                if exact_message is not None:
                    assert error["message"] == exact_message
                else:
                    assert "max_tokens" in error["message"]
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_fixed_feature_policy_rejects_before_core(self, anthropic_env):
        messages = [{"role": "user", "content": "q"}]
        cases = [
            (
                _payload(messages, thinking={"type": "disabled"}),
                "thinking is not supported by this endpoint",
            ),
            (
                _payload(messages, output_config={"effort": "high"}),
                "output_config is not enabled for this deployment",
            ),
            (_payload(messages, betas=["b-1"]), "betas is not enabled for this deployment"),
            (
                _payload(messages, tools=[{"type": "web_search_20250305", "name": "web_search"}]),
                "server tool 'web_search' (type='web_search_20250305') is not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "thinking", "thinking": "t"}]}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload([{"role": "system", "content": [{"type": "thinking", "thinking": "t"}]}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload(messages, system=[{"type": "thinking", "thinking": "t"}]),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": [{"type": "thinking", "thinking": "t"}],
                                }
                            ],
                        }
                    ]
                ),
                "thinking content blocks are only supported in assistant history",
            ),
            (
                _payload([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
                "redacted_thinking content blocks are not supported by this endpoint",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "data": "eA=="},
                                }
                            ],
                        }
                    ]
                ),
                "image content blocks are not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "tool_reference", "tool_name": "f"}]}]),
                "tool_reference content blocks are not enabled for this deployment",
            ),
            (
                _payload([{"role": "user", "content": [{"type": "search_result", "title": "t"}]}]),
                "search_result content blocks are not enabled for this deployment",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": "failed",
                                    "is_error": True,
                                }
                            ],
                        }
                    ]
                ),
                "tool_result is_error=true is not supported by this endpoint",
            ),
            (
                _payload(
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t",
                                    "content": [{"type": "tool_reference", "tool_name": "f"}],
                                }
                            ],
                        }
                    ]
                ),
                "tool_reference content blocks are not enabled for this deployment",
            ),
            (
                _payload(messages, system=[{"type": "image", "source": {"url": "https://x"}}]),
                "image content blocks are not enabled for this deployment",
            ),
        ]
        session_id = _create_session(anthropic_env.url)
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            for payload, message in cases:
                resp = _post_messages(anthropic_env.url, session_id, payload)
                assert resp.status_code == 400, payload
                assert resp.json() == {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": message},
                }
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_converter_exception_is_400_before_core(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        backend_requests = len(anthropic_env.backend.request_log)
        records = _records(anthropic_env.url, session_id)
        with (
            patch.object(
                sessions_module,
                "convert_to_chat_completion_request",
                side_effect=RuntimeError("conversion boom"),
            ),
            patch.object(core_module.SessionCore, "chat_completions", autospec=True) as v1_chat,
            patch.object(v2_core_module.SessionCoreV2, "chat_completions", autospec=True) as v2_chat,
        ):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "q"}]))
        assert resp.status_code == 400
        assert resp.json() == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "conversion boom"},
        }
        assert v1_chat.await_count == 0
        assert v2_chat.await_count == 0
        assert len(anthropic_env.backend.request_log) == backend_requests
        assert _records(anthropic_env.url, session_id) == records

    def test_feature_priority_is_stable(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_reference", "tool_name": "f"}]},
        ]
        payloads = [
            (
                _payload(
                    messages,
                    thinking={"type": "disabled"},
                    output_config={"effort": "high"},
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "thinking is not supported by this endpoint",
            ),
            (
                _payload(
                    messages,
                    output_config={"effort": "high"},
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "output_config is not enabled for this deployment",
            ),
            (
                _payload(
                    messages,
                    betas=["b-1"],
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "betas is not enabled for this deployment",
            ),
            (
                _payload(
                    messages,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    system=[{"type": "image", "source": {"url": "https://x"}}],
                ),
                "server tool 'web_search' (type='web_search_20250305') is not enabled for this deployment",
            ),
            (
                _payload(messages, system=[{"type": "image", "source": {"url": "https://x"}}]),
                "image content blocks are not enabled for this deployment",
            ),
        ]
        for payload, message in payloads:
            request = AnthropicMessagesRequest.model_validate(payload)
            with pytest.raises(ValueError) as exc_info:
                sessions_module._validate_anthropic_features(request)
            assert str(exc_info.value) == message

    def test_policy_traversal_boundaries_match_converter(self):
        system_nested = _payload(
            [{"role": "user", "content": "q"}],
            betas=[],
            system=[
                {
                    "type": "tool_result",
                    "tool_use_id": "system-tool",
                    "content": [{"type": "image", "source": {"url": "https://x"}}],
                }
            ],
        )
        request = AnthropicMessagesRequest.model_validate(system_nested)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        assert converted.model_dump(mode="json", exclude_none=True)["messages"] == [{"role": "user", "content": "q"}]

        depth_two = _payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "outer",
                            "content": [
                                {"type": "text", "text": "depth1"},
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "inner",
                                    "content": [{"type": "image", "source": {"type": "base64", "data": "eA=="}}],
                                },
                            ],
                        }
                    ],
                }
            ]
        )
        request = AnthropicMessagesRequest.model_validate(depth_two)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        dump = converted.model_dump(mode="json", exclude_none=True)
        assert dump["messages"] == [{"role": "tool", "content": "depth1", "tool_call_id": "outer"}]
        assert "image_url" not in json.dumps(dump)

        arbitrary_json = _payload(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call1",
                            "name": "f",
                            "input": {"type": "image", "nested": {"type": "thinking"}},
                        }
                    ],
                }
            ],
            metadata={"type": "search_result"},
            tools=[
                {
                    "name": "f",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "image"}}},
                }
            ],
        )
        request = AnthropicMessagesRequest.model_validate(arbitrary_json)
        sessions_module._validate_anthropic_features(request)
        converted = convert_to_chat_completion_request(request, merge_inline_system=True)
        dump = converted.model_dump(mode="json", exclude_none=True)
        assert json.loads(dump["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {
            "type": "image",
            "nested": {"type": "thinking"},
        }
        assert dump["tools"][0]["function"]["parameters"] == {
            "type": "object",
            "properties": {"x": {"type": "image"}},
        }

    def test_backend_failure_maps_to_anthropic_error_without_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)

        async def reject(self, request, compute_fn):
            return JSONResponse(content={"error": "mock backend failure"}, status_code=500)

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=reject):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error" and body["error"]["type"] == "api_error"
        assert body["error"]["message"] == "Internal server error"
        assert _records(anthropic_env.url, session_id) == []

    def test_post_commit_conversion_failure_returns_500_and_keeps_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        with patch.object(sessions_module, "convert_response", side_effect=RuntimeError("boom")):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        # The accepted first-version boundary: core already committed.
        assert len(_records(anthropic_env.url, session_id)) == 1

    def test_sse_build_failure_returns_json_500_not_partial_stream(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        with patch.object(
            sessions_module.anthropic_utils, "to_anthropic_fake_sse_events", side_effect=RuntimeError("boom")
        ):
            resp = _post_messages(
                anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
            )
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["type"] == "api_error"
        # Same post-commit boundary as the non-stream twin: the record stays.
        assert len(_records(anthropic_env.url, session_id)) == 1


def _tool_turn1(url: str, session_id: str) -> dict:
    """Run the tool-eliciting first turn; returns the tool_use block."""
    turn1 = _post_messages(
        url, session_id, _payload([{"role": "user", "content": "please use the weather tool"}], tools=_TOOLS)
    )
    assert turn1.status_code == 200
    body = turn1.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = next(block for block in body["content"] if block["type"] == "tool_use")
    assert tool_use["name"] == "get_weather" and tool_use["input"] == {"city": "Paris", "unit": "C"}
    return tool_use


def _tool_turn2_payload(tool_use: dict) -> dict:
    """Replay with the tool_use input keys REVERSED: json.dumps preserves the
    object's key order, so the re-serialized arguments spelling differs from
    the stored one — accepted only by ``loose_tool_call``."""
    respelled_input = dict(reversed(list(tool_use["input"].items())))
    return _payload(
        [
            {"role": "user", "content": "please use the weather tool"},
            {"role": "assistant", "content": [{**tool_use, "input": respelled_input}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "sunny"}]},
        ],
        tools=_TOOLS,
    )


class TestAnthropicToolFlow:
    def test_multi_turn_tool_flow_reuses_stored_prefix(self, anthropic_env_loose):
        url = anthropic_env_loose.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200
        assert turn2.json()["content"] == [{"type": "text", "text": "final-answer"}]

        records = _records(url, session_id)
        assert len(records) == 2
        first_ids = records[0]["request"]["input_ids"]
        second_ids = records[1]["request"]["input_ids"]
        # The stored TITO prefix is reused: turn 2's prompt extends turn 1's.
        assert second_ids[: len(first_ids)] == first_ids
        replayed_assistant = next(m for m in records[1]["request"]["messages"] if m["role"] == "assistant")
        assert replayed_assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(replayed_assistant["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Paris",
            "unit": "C",
        }

    def test_strict_matcher_rejects_respelled_tool_arguments(self, anthropic_env):
        """Design matcher gate, end to end: under the default ``strict``
        matcher the re-serialized tool arguments diverge from the stored
        assistant message, so v1 rolls back to the empty checkpoint —
        discarding turn 1's record — and re-renders from scratch instead of
        reusing the stored TITO prefix (contrast with the loose test above)."""
        if anthropic_env.version != "v1":
            pytest.skip("v2 branches to a new lineage instead of rolling back; the loose fixture covers v2")
        url = anthropic_env.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)
        first_ids = _records(url, session_id)[0]["request"]["input_ids"]

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200

        records = _records(url, session_id)
        assert len(records) == 1
        assert [m["role"] for m in records[0]["request"]["messages"]] == ["user", "assistant", "tool"]
        second_ids = records[0]["request"]["input_ids"]
        assert second_ids[: len(first_ids)] != first_ids


class TestAnthropicCloseRace:
    def test_delete_during_inflight_chat_skips_update_gracefully(self):
        """Split-lock close race through the Anthropic route: DELETE lands
        while the chat is mid-proxy; Phase 3 sees closing=True and skips the
        commit, but the client still gets a well-formed Anthropic response —
        the same 200-with-skip outcome as the OpenAI route."""
        with _anthropic_env(latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = _payload([{"role": "user", "content": "hello"}])

            with ThreadPoolExecutor(max_workers=2) as pool:
                inflight = pool.submit(_post_messages, env.url, session_id, payload)

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                inflight_resp = inflight.result(timeout=30.0)

            assert delete_resp.status_code == 204
            assert inflight_resp.status_code == 200
            body = inflight_resp.json()
            assert body["type"] == "message"
            assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]

            post_delete = _post_messages(env.url, session_id, payload)
            assert post_delete.status_code == 404
            assert post_delete.json()["error"]["type"] == "not_found_error"


class TestMatcherGate:
    """Design matcher gate: object → string re-serialization may change tool
    argument spelling; ``strict`` must reject it, ``loose_tool_call`` must
    accept it, so tool launch profiles default to ``loose_tool_call``."""

    def _replayed_assistant(self, arguments_object: dict) -> dict:
        request = AnthropicMessagesRequest.model_validate(
            _payload(
                [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call1", "name": "f", "input": arguments_object}],
                    },
                ]
            )
        )
        openai_request = convert_to_chat_completion_request(request, merge_inline_system=True)
        return openai_request.model_dump(mode="json", exclude_none=True, by_alias=True)["messages"][-1]

    def test_strict_rejects_and_loose_accepts_respelled_arguments(self):
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert replayed["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # json.dumps spelling
        assert strict_message_matches(stored, replayed) is False
        loose = resolve_session_message_matcher("loose_tool_call")
        assert loose(stored, replayed) is True

    def test_strict_accepts_identical_spelling(self):
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert strict_message_matches(stored, replayed) is True
