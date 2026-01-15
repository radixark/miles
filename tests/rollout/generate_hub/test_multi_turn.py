import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def to_pydantic_tools(tools: list[dict]) -> list[Tool]:
    return [Tool(type=t["type"], function=Function(**t["function"])) for t in tools]


class TestApplyChatTemplateWithTools:
    """
    Demonstrates how to use apply_chat_template with tools parameter.

    When generating prompts for tool-calling models:
    1. Pass tools to apply_chat_template() so the model knows available tools
    2. Model generates output with tool calls in a specific format
    3. Use FunctionCallParser to parse the generated tool calls
    """

    def test_apply_chat_template_includes_tools(self):
        """Verify that apply_chat_template with tools produces a prompt containing tool info."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        prompt_without_tools = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_with_tools = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
        )

        assert "get_weather" not in prompt_without_tools
        assert "get_weather" in prompt_with_tools
        assert "city" in prompt_with_tools


class TestSGLangFunctionCallParser:
    """Test to demonstrate and ensure SGLang function call parser have features we need without breaking changes."""

    @pytest.mark.parametrize(
        "model_output,expected",
        [
            pytest.param(
                'Let me check the weather for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
                (
                    "Let me check the weather for you.",
                    [ToolCallItem(tool_index=0, name="get_weather", parameters='{"city": "Paris"}')],
                ),
                id="single_tool_call",
            ),
            pytest.param(
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
                id="multi_tool_calls",
            ),
            pytest.param(
                "The weather is sunny today.",
                ("The weather is sunny today.", []),
                id="no_tool_call",
            ),
        ],
    )
    def test_parse_non_stream(self, model_output, expected):
        parser = FunctionCallParser(tools=to_pydantic_tools(SAMPLE_TOOLS), tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected
