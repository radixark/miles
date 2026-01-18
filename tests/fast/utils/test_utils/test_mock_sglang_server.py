import asyncio
import concurrent.futures
import time

import pytest
import requests

from miles.utils.test_utils.mock_sglang_server import (
    Counter,
    ProcessResult,
    ProcessResultMetaInfo,
    default_process_fn,
    with_mock_server,
)
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, TwoTurnStub


def expected_logprobs(tokenizer, text: str) -> list[dict]:
    output_ids = tokenizer.encode(text, add_special_tokens=False)
    return [{"token": tokenizer.convert_ids_to_tokens(tid), "logprob": -i / 128} for i, tid in enumerate(output_ids)]


@pytest.fixture(scope="module")
def mock_server():
    with with_mock_server() as server:
        yield server


class TestProcessResultMetaInfo:
    def test_to_dict_empty(self):
        assert ProcessResultMetaInfo().to_dict() == {}

    def test_to_dict_single_field(self):
        assert ProcessResultMetaInfo(weight_version="v1").to_dict() == {"weight_version": "v1"}

    def test_to_dict_partial_fields(self):
        assert ProcessResultMetaInfo(weight_version="v1", spec_accept_token_num=10).to_dict() == {
            "weight_version": "v1",
            "spec_accept_token_num": 10,
        }

    def test_to_dict_all_fields(self):
        assert ProcessResultMetaInfo(
            weight_version="v1",
            routed_experts="abc",
            spec_accept_token_num=10,
            spec_draft_token_num=15,
            spec_verify_ct=3,
        ).to_dict() == {
            "weight_version": "v1",
            "routed_experts": "abc",
            "spec_accept_token_num": 10,
            "spec_draft_token_num": 15,
            "spec_verify_ct": 3,
        }


class TestDefaultProcessFn:
    def test_math_question(self):
        assert default_process_fn("What is 1+5?") == ProcessResult(text="\\boxed{6}", finish_reason="stop")
        assert default_process_fn("What is 1+10?") == ProcessResult(text="\\boxed{11}", finish_reason="stop")

    def test_unknown_question(self):
        assert default_process_fn("Hello") == ProcessResult(text="I don't understand.", finish_reason="stop")


class TestCounter:
    def test_tracks_max(self):
        counter = Counter()
        assert counter.max_value == 0

        with counter.track():
            assert counter.max_value == 1
            with counter.track():
                assert counter.max_value == 2

        counter.reset()
        assert counter.max_value == 0

    def test_concurrent_tasks(self):
        counter = Counter()

        async def task():
            with counter.track():
                await asyncio.sleep(0.1)

        async def run_all():
            await asyncio.gather(task(), task(), task())

        asyncio.run(run_all())
        assert counter.max_value == 3


class TestMockServerBasic:
    def test_start_stop(self, mock_server):
        assert mock_server.port > 0
        assert f"http://{mock_server.host}:{mock_server.port}" == mock_server.url

    def test_request_log_and_reset_stats(self, mock_server):
        mock_server.reset_stats()
        assert len(mock_server.request_log) == 0

        payload = {"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0.5}, "return_logprob": True}
        requests.post(f"{mock_server.url}/generate", json=payload, timeout=5.0)
        assert len(mock_server.request_log) == 1
        assert mock_server.request_log[0] == payload

        mock_server.reset_stats()
        assert len(mock_server.request_log) == 0
        assert mock_server.max_concurrent == 0

    @pytest.mark.parametrize("latency,min_time,max_time", [(0.0, 0.0, 0.3), (0.5, 0.5, 1.0)])
    def test_latency(self, latency, min_time, max_time):
        with with_mock_server(latency=latency) as server:
            start = time.time()
            requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)
            elapsed = time.time() - start
            assert min_time <= elapsed < max_time

    def test_max_concurrent_with_latency(self):
        with with_mock_server(latency=0.1) as server:

            def send_request():
                requests.post(f"{server.url}/generate", json={"input_ids": [1], "sampling_params": {}}, timeout=5.0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(send_request) for _ in range(3)]
                concurrent.futures.wait(futures)

            assert server.max_concurrent == 3

    def test_health_endpoint(self, mock_server):
        response = requests.get(f"{mock_server.url}/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_abort_request_endpoint(self, mock_server):
        response = requests.post(f"{mock_server.url}/abort_request", json={}, timeout=5.0)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestGenerateEndpoint:
    def test_basic(self, mock_server):
        prompt = "What is 1+7?"
        input_ids = mock_server.tokenizer.encode(prompt, add_special_tokens=False)
        assert input_ids == [3838, 374, 220, 16, 10, 22, 30]

        response = requests.post(
            f"{mock_server.url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 10},
                "return_logprob": True,
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        assert response.json() == {
            "text": "\\boxed{8}",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(input_ids),
                "cached_tokens": 0,
                "completion_tokens": 5,
                "output_token_logprobs": [
                    [-0.0, 59],
                    [-0.0078125, 79075],
                    [-0.015625, 90],
                    [-0.0234375, 23],
                    [-0.03125, 92],
                ],
            },
        }

    def test_with_meta_info(self):
        def process_fn(_: str) -> ProcessResult:
            return ProcessResult(
                text="ok",
                finish_reason="stop",
                cached_tokens=5,
                meta_info=ProcessResultMetaInfo(
                    weight_version="v2.0",
                    routed_experts="encoded_data",
                    spec_accept_token_num=10,
                    spec_draft_token_num=15,
                    spec_verify_ct=3,
                ),
            )

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )

            assert response.json() == {
                "text": "ok",
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 3,
                    "cached_tokens": 5,
                    "completion_tokens": 1,
                    "output_token_logprobs": [[-0.0, 562]],
                    "weight_version": "v2.0",
                    "routed_experts": "encoded_data",
                    "spec_accept_token_num": 10,
                    "spec_draft_token_num": 15,
                    "spec_verify_ct": 3,
                },
            }

    def test_finish_reason_length(self):
        def process_fn(_: str) -> ProcessResult:
            return ProcessResult(text="truncated output", finish_reason="length")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": [1, 2, 3], "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )
            data = response.json()

        finish_reason = data["meta_info"]["finish_reason"]
        assert finish_reason["type"] == "length"
        assert finish_reason["length"] == data["meta_info"]["completion_tokens"]


class TestChatCompletionsEndpoint:
    def test_basic(self, mock_server):
        response = requests.post(
            f"{mock_server.url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is 1+5?"}],
            },
            timeout=5.0,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["id"].startswith("chatcmpl-")
        assert isinstance(data["created"], int)
        assert data == {
            "id": data["id"],
            "object": "chat.completion",
            "created": data["created"],
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "\\boxed{6}", "tool_calls": None},
                    "logprobs": {"content": expected_logprobs(mock_server.tokenizer, "\\boxed{6}")},
                    "finish_reason": "stop",
                }
            ],
        }

    def test_with_tool_calls(self):
        tool_call_response = 'Let me check for you.\n<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>'

        def process_fn(_: str) -> ProcessResult:
            return ProcessResult(text=tool_call_response, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What year is it?"}],
                    "tools": SAMPLE_TOOLS,
                },
                timeout=5.0,
            )
            data = response.json()

            assert data["choices"][0] == {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check for you.",
                    "tool_calls": [
                        {"id": "call00000", "type": "function", "function": {"name": "get_year", "arguments": "{}"}}
                    ],
                },
                "logprobs": {"content": expected_logprobs(server.tokenizer, tool_call_response)},
                "finish_reason": "tool_calls",
            }

    def test_with_tools_but_no_tool_call(self):
        response_text = "The weather is sunny today."

        def process_fn(_: str) -> ProcessResult:
            return ProcessResult(text=response_text, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What's the weather?"}],
                    "tools": SAMPLE_TOOLS,
                },
                timeout=5.0,
            )
            data = response.json()

            assert data["choices"][0] == {
                "index": 0,
                "message": {"role": "assistant", "content": response_text, "tool_calls": None},
                "logprobs": {"content": expected_logprobs(server.tokenizer, response_text)},
                "finish_reason": "stop",
            }

    def test_with_multiple_tool_calls(self):
        multi_tool_response = (
            "I will get year and temperature.\n"
            '<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_temperature", "arguments": {"location": "Shanghai"}}\n</tool_call>'
        )

        def process_fn(_: str) -> ProcessResult:
            return ProcessResult(text=multi_tool_response, finish_reason="stop")

        with with_mock_server(process_fn=process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "What year and temperature?"}],
                    "tools": SAMPLE_TOOLS,
                },
                timeout=5.0,
            )
            data = response.json()

            assert data["choices"][0] == {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will get year and temperature.",
                    "tool_calls": [
                        {"id": "call00000", "type": "function", "function": {"name": "get_year", "arguments": "{}"}},
                        {
                            "id": "call00001",
                            "type": "function",
                            "function": {"name": "get_temperature", "arguments": '{"location": "Shanghai"}'},
                        },
                    ],
                },
                "logprobs": {"content": expected_logprobs(server.tokenizer, multi_tool_response)},
                "finish_reason": "tool_calls",
            }


class TestMultiTurnToolCallProcessFn:
    @pytest.mark.parametrize(
        "prompt,expected_response",
        [
            pytest.param(TwoTurnStub.FIRST_PROMPT, TwoTurnStub.FIRST_RESPONSE, id="first_turn"),
            pytest.param(TwoTurnStub.SECOND_PROMPT, TwoTurnStub.SECOND_RESPONSE, id="second_turn"),
        ],
    )
    def test_generate_endpoint(self, prompt, expected_response):
        with with_mock_server(process_fn=TwoTurnStub.process_fn) as server:
            input_ids = server.tokenizer.encode(prompt, add_special_tokens=False)
            response = requests.post(
                f"{server.url}/generate",
                json={"input_ids": input_ids, "sampling_params": {}, "return_logprob": True},
                timeout=5.0,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == expected_response
            assert data["meta_info"]["finish_reason"] == {"type": "stop"}

    @pytest.mark.parametrize(
        "messages,expected_content,expected_tool_calls,expected_finish_reason",
        [
            pytest.param(
                TwoTurnStub.OPENAI_MESSAGES_FIRST_TURN,
                TwoTurnStub.FIRST_RESPONSE_CONTENT,
                TwoTurnStub.FIRST_TOOL_CALLS_OPENAI_FORMAT,
                "tool_calls",
                id="first_turn",
            ),
            pytest.param(
                TwoTurnStub.OPENAI_MESSAGES_SECOND_TURN_FROM_CLIENT,
                TwoTurnStub.SECOND_RESPONSE,
                None,
                "stop",
                id="second_turn",
            ),
        ],
    )
    def test_chat_completions_endpoint(self, messages, expected_content, expected_tool_calls, expected_finish_reason):
        with with_mock_server(process_fn=TwoTurnStub.process_fn) as server:
            response = requests.post(
                f"{server.url}/v1/chat/completions",
                json={"model": "test", "messages": messages, "tools": SAMPLE_TOOLS},
                timeout=5.0,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == expected_content
            assert data["choices"][0]["message"]["tool_calls"] == expected_tool_calls
            assert data["choices"][0]["finish_reason"] == expected_finish_reason
