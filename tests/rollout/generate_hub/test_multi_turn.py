import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


class TestSGLangFunctionCallParser:
    """Test to ensure SGLang function call parser have features we need without breaking changes."""

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

    @pytest.mark.parametrize(
        "model_output,expected",
        [
            (
                '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
                ("", [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')]),
            ),
            (
                '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n</tool_call>\n'
                '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>',
                (
                    "",
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
        parser = FunctionCallParser(tools=self.SAMPLE_TOOLS, tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected
