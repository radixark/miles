import json

import pytest
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from miles.utils.test_utils.mock_sglang_server import make_multi_turn_process_fn
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, execute_tool_call

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
YEAR = 2026
TEMPERATURE = -60


class TestToolExecution:
    def test_execute_get_year(self):
        result = execute_tool_call("get_year", {})
        assert result == {"year": YEAR}

    def test_execute_get_temperature(self):
        result = execute_tool_call("get_temperature", {"location": "Mars"})
        assert result == {"temperature": TEMPERATURE}


class TestToolCallParsing:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_parse_multi_tool_calls(self, parser):
        process_fn = make_multi_turn_process_fn(i=0, year=YEAR, temperature=TEMPERATURE)
        response = process_fn("first turn").text
        normal_text, calls = parser.parse_non_stream(response)

        assert normal_text == "Let me get the year and temperature first."
        assert len(calls) == 2
        assert calls[0].name == "get_year"
        assert calls[0].parameters == "{}"
        assert calls[1].name == "get_temperature"
        assert json.loads(calls[1].parameters) == {"location": "Mars"}

    def test_parse_no_tool_calls(self, parser):
        process_fn = make_multi_turn_process_fn(i=10, year=YEAR, temperature=TEMPERATURE)
        process_fn("first turn")
        response = process_fn("second turn").text
        normal_text, calls = parser.parse_non_stream(response)

        assert len(calls) == 0
        expected = 10 + YEAR + TEMPERATURE
        assert f"The answer is: 10 + {YEAR} + {TEMPERATURE} = {expected}" in normal_text


class TestMultiTurnProcessFn:
    def test_first_turn_returns_tool_calls(self):
        process_fn = make_multi_turn_process_fn(i=10, year=YEAR, temperature=TEMPERATURE)
        result = process_fn("What is 10 + year + temperature?")

        assert result.finish_reason == "stop"
        assert "<tool_call>" in result.text
        assert "get_year" in result.text
        assert "get_temperature" in result.text

    def test_second_turn_returns_answer(self):
        process_fn = make_multi_turn_process_fn(i=10, year=YEAR, temperature=TEMPERATURE)
        process_fn("What is 10 + year + temperature?")
        result = process_fn("Tool results...")

        assert result.finish_reason == "stop"
        expected = 10 + YEAR + TEMPERATURE
        assert f"The answer is: 10 + {YEAR} + {TEMPERATURE} = {expected}" in result.text
        assert "<tool_call>" not in result.text

    def test_answer_calculation(self):
        for i in [0, 5, 100]:
            process_fn = make_multi_turn_process_fn(i=i, year=YEAR, temperature=TEMPERATURE)
            process_fn("first turn")
            result = process_fn("second turn")
            expected = i + YEAR + TEMPERATURE
            assert f"The answer is: {i} + {YEAR} + {TEMPERATURE} = {expected}" in result.text


class TestEndToEndToolFlow:
    @pytest.fixture
    def parser(self):
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        return FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    def test_full_multi_turn_flow(self, parser):
        i = 42
        process_fn = make_multi_turn_process_fn(i=i, year=YEAR, temperature=TEMPERATURE)

        first_response = process_fn(f"What is {i} + year + temperature?")
        normal_text, calls = parser.parse_non_stream(first_response.text)

        assert len(calls) == 2
        tool_results = []
        for call in calls:
            params = json.loads(call.parameters) if call.parameters else {}
            result = execute_tool_call(call.name, params)
            tool_results.append({"name": call.name, "result": result})

        assert tool_results[0] == {"name": "get_year", "result": {"year": YEAR}}
        assert tool_results[1] == {"name": "get_temperature", "result": {"temperature": TEMPERATURE}}

        tool_response_str = "\n".join(json.dumps(r["result"]) for r in tool_results)
        second_response = process_fn(tool_response_str)

        expected_answer = i + YEAR + TEMPERATURE
        assert f"The answer is: {i} + {YEAR} + {TEMPERATURE} = {expected_answer}" in second_response.text
