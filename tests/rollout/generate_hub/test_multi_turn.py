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

DEEPSEEKV3_SINGLE_TOOL_CALL = (
    "<｜tool▁calls▁begin｜>"
    "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
    '```json\n{"city": "Paris"}\n```'
    "<｜tool▁call▁end｜>"
    "<｜tool▁calls▁end｜>"
)

DEEPSEEKV3_MULTI_TOOL_CALLS = (
    "<｜tool▁calls▁begin｜>"
    "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
    '```json\n{"city": "Shanghai"}\n```'
    "<｜tool▁call▁end｜>\n"
    "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n"
    '```json\n{"query": "restaurants"}\n```'
    "<｜tool▁call▁end｜>"
    "<｜tool▁calls▁end｜>"
)


def test_function_call_parser_single_tool_call():
    """FunctionCallParser supports: deepseekv3, qwen25, llama3, mistral, pythonic, etc."""
    parser = FunctionCallParser(tools=SAMPLE_TOOLS, tool_call_parser="deepseekv3")

    assert parser.has_tool_call(DEEPSEEKV3_SINGLE_TOOL_CALL)

    normal_text, tool_calls = parser.parse_non_stream(DEEPSEEKV3_SINGLE_TOOL_CALL)

    assert (normal_text, tool_calls) == (
        "",
        [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')],
    )


def test_function_call_parser_multi_tool_calls():
    parser = FunctionCallParser(tools=SAMPLE_TOOLS, tool_call_parser="deepseekv3")

    normal_text, tool_calls = parser.parse_non_stream(DEEPSEEKV3_MULTI_TOOL_CALLS)

    assert (normal_text, tool_calls) == (
        "",
        [
            ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Shanghai"}'),
            ToolCallItem(tool_index=1, name="search", parameters='{"query": "restaurants"}'),
        ],
    )


def test_function_call_parser_no_tool_call():
    parser = FunctionCallParser(tools=SAMPLE_TOOLS, tool_call_parser="deepseekv3")
    model_output = "The weather is sunny today."

    assert not parser.has_tool_call(model_output)

    normal_text, tool_calls = parser.parse_non_stream(model_output)

    assert (normal_text, tool_calls) == ("The weather is sunny today.", [])
