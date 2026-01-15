import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser


SAMPLE_TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        ),
    ),
    Tool(
        type="function",
        function=Function(
            name="search",
            description="Search for information",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ),
]


def test_deepseekv3_parse_single_tool_call():
    """
    DeepSeek V3 format:
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name
        ```json
        {"arg": "value"}
        ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    detector = DeepSeekV3Detector()
    model_output = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Beijing", "unit": "celsius"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert detector.has_tool_call(model_output)
    assert detector.detect_and_parse(model_output, SAMPLE_TOOLS) == StreamingParseResult(
        normal_text="",
        calls=[
            ToolCallItem(
                tool_index=0,
                name="get_weather",
                parameters='{"city": "Beijing", "unit": "celsius"}',
            )
        ],
    )


def test_deepseekv3_parse_multiple_tool_calls():
    detector = DeepSeekV3Detector()
    model_output = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Shanghai"}\n```'
        "<｜tool▁call▁end｜>\n"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
        '```json\n{"query": "restaurants"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert detector.detect_and_parse(model_output, SAMPLE_TOOLS) == StreamingParseResult(
        normal_text="",
        calls=[
            ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Shanghai"}'),
            ToolCallItem(tool_index=1, name="search", parameters='{"query": "restaurants"}'),
        ],
    )


def test_deepseekv3_text_before_tool_call():
    detector = DeepSeekV3Detector()
    model_output = (
        "Let me check the weather.\n"
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Tokyo"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert detector.detect_and_parse(model_output, SAMPLE_TOOLS) == StreamingParseResult(
        normal_text="Let me check the weather.",
        calls=[ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Tokyo"}')],
    )


def test_deepseekv3_no_tool_call():
    detector = DeepSeekV3Detector()
    model_output = "The weather is sunny today."

    assert not detector.has_tool_call(model_output)
    assert detector.detect_and_parse(model_output, SAMPLE_TOOLS) == StreamingParseResult(
        normal_text="The weather is sunny today.",
        calls=[],
    )


def test_function_call_parser_wrapper():
    """FunctionCallParser supports: deepseekv3, qwen25, llama3, mistral, pythonic, etc."""
    parser = FunctionCallParser(tools=SAMPLE_TOOLS, tool_call_parser="deepseekv3")
    model_output = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"city": "Paris"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    assert parser.has_tool_call(model_output)

    normal_text, tool_calls = parser.parse_non_stream(model_output)

    assert normal_text == ""
    assert tool_calls == [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')]
