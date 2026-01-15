import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
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


class TestSGLangFunctionCallParser:
    """Test to demonstrate and ensure SGLang function call parser have features we need without breaking changes."""

    @pytest.mark.parametrize(
        "model_output,expected",
        [
            (
                'Let me check the weather for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
                (
                    "Let me check the weather for you.",
                    [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')],
                ),
            ),
            (
                "I will search for weather and restaurants.\n"
                '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n</tool_call>\n'
                '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>',
                (
                    "I will search for weather and restaurants.",
                    [
                        ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Shanghai"}'),
                        ToolCallItem(tool_index=1, name="search", parameters='{"query": "restaurants"}'),
                    ],
                ),
            ),
            (
                "The weather is sunny today.",
                ("The weather is sunny today.", []),
            ),
        ],
        ids=["single_tool_call", "multi_tool_calls", "no_tool_call"],
    )
    def test_parse_non_stream(self, model_output, expected):
        parser = FunctionCallParser(tools=SAMPLE_TOOLS, tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected
