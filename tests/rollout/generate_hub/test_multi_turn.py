import json

import pytest
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from tests.fixtures.tool_fixtures import SAMPLE_TOOLS, execute_tool_call

from miles.utils.test_utils.mock_sglang_server import ProcessResult

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def make_first_turn_response() -> str:
    return (
        "Let me get the year and temperature first.\n"
        "<tool_call>\n"
        '{"name": "get_year", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_temperature", "arguments": {"location": "Mars"}}\n'
        "</tool_call>"
    )


def make_second_turn_response(i: int) -> str:
    result = i + 2026 + 25
    return (
        "Now I have the information I need.\n"
        "The current year is 2025, and the temperature on Mars is 25 degrees.\n"
        f"So the calculation is: {i} + 2025 + 25 = {result}.\n"
        f"The answer is {result}."
    )


def make_multi_turn_process_fn(i: int):
    turn_count = {"value": 0}

    def process_fn(prompt: str) -> ProcessResult:
        turn = turn_count["value"]
        turn_count["value"] += 1

        if turn == 0:
            return ProcessResult(text=make_first_turn_response(), finish_reason="stop")
        else:
            return ProcessResult(text=make_second_turn_response(i), finish_reason="stop")

    return process_fn


class TestToolExecution:
    def test_execute_get_year(self):
        result = execute_tool_call("get_year", {})
        assert result == {"year": 2025}

    def test_execute_get_temperature(self):
        result = execute_tool_call("get_temperature", {"location": "Mars"})
        assert result == {"temperature": -60}


class TestToolCallParsing:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_parse_multi_tool_calls(self, parser):
        response = make_first_turn_response()
        normal_text, calls = parser.parse_non_stream(response)

        assert normal_text == "Let me get the year and temperature first."
        assert len(calls) == 2
        assert calls[0].name == "get_year"
        assert calls[0].parameters == "{}"
        assert calls[1].name == "get_temperature"
        assert json.loads(calls[1].parameters) == {"location": "Mars"}

    def test_parse_no_tool_calls(self, parser):
        response = make_second_turn_response(10)
        normal_text, calls = parser.parse_non_stream(response)

        assert len(calls) == 0
        assert "The answer is 2060" in normal_text


class TestMultiTurnProcessFn:
    def test_first_turn_returns_tool_calls(self):
        process_fn = make_multi_turn_process_fn(i=10)
        result = process_fn("What is 10 + year + temperature?")

        assert result.finish_reason == "stop"
        assert "<tool_call>" in result.text
        assert "get_year" in result.text
        assert "get_temperature" in result.text

    def test_second_turn_returns_answer(self):
        process_fn = make_multi_turn_process_fn(i=10)
        process_fn("What is 10 + year + temperature?")
        result = process_fn("Tool results...")

        assert result.finish_reason == "stop"
        assert "The answer is 2060" in result.text
        assert "<tool_call>" not in result.text

    def test_answer_calculation(self):
        for i in [0, 5, 100]:
            process_fn = make_multi_turn_process_fn(i=i)
            process_fn("first turn")
            result = process_fn("second turn")
            expected = i + 2025 + 25
            assert f"The answer is {expected}" in result.text


class TestEndToEndToolFlow:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_full_multi_turn_flow(self, parser):
        i = 42
        process_fn = make_multi_turn_process_fn(i=i)

        first_response = process_fn(f"What is {i} + year + temperature?")
        normal_text, calls = parser.parse_non_stream(first_response.text)

        assert len(calls) == 2
        tool_results = []
        for call in calls:
            params = json.loads(call.parameters) if call.parameters else {}
            result = execute_tool_call(call.name, params)
            tool_results.append({"name": call.name, "result": result})

        assert tool_results[0] == {"name": "get_year", "result": {"year": 2025}}
        assert tool_results[1] == {"name": "get_temperature", "result": {"temperature": 25}}

        tool_response_str = "\n".join(json.dumps(r["result"]) for r in tool_results)
        second_response = process_fn(tool_response_str)

        expected_answer = i + 2025 + 25
        assert f"The answer is {expected_answer}" in second_response.text
