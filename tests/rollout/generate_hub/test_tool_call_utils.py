import pytest
from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS

from miles.rollout.generate_hub.tool_call_utils import _DUMMY_USER, _build_dummy_assistant, tokenize_tool_responses

TOOL_CALL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V3",
    "stepfun-ai/step3",
    "MiniMaxAI/MiniMax-M2",
    "internlm/internlm3-8b-instruct",
    "THUDM/glm-4-9b-chat",
    "moonshotai/Kimi-K2-Instruct",
    "XiaomiMiMo/MiMo-7B-RL",
]

SINGLE_TOOL_CALL_ONLY_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
]

SAMPLE_TOOL_RESPONSES = [
    {
        "role": "tool",
        "tool_call_id": "call00000",
        "content": '{"year": 2026}',
        "name": "get_year",
    },
    {
        "role": "tool",
        "tool_call_id": "call00001",
        "content": '{"temperature": 25}',
        "name": "get_temperature",
    },
]


class TestTokenizeToolResponses:
    @pytest.mark.parametrize("num_tools", [1, 2])
    @pytest.mark.parametrize("model_name", TOOL_CALL_TEST_MODELS)
    def test_tokenize_tool_responses(self, model_name, num_tools):
        if num_tools > 1 and model_name in SINGLE_TOOL_CALL_ONLY_MODELS:
            pytest.skip(f"{model_name} only supports single tool call")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        tool_responses = SAMPLE_TOOL_RESPONSES[:num_tools]
        assert len(tool_responses) == num_tools

        actual_token_ids = tokenize_tool_responses(tool_responses, tokenizer)
        actual_str = tokenizer.decode(actual_token_ids)

        dummy_assistant = _build_dummy_assistant(tool_responses)
        base_messages = [_DUMMY_USER, dummy_assistant]
        expected_str = self._compute_chat_template_diff(base_messages, tool_responses, tokenizer)

        assert actual_str == expected_str, f"{model_name=}"


    @staticmethod
    def _compute_chat_template_diff(base_messages, extra_messages, tokenizer) -> str:
        text_with = tokenizer.apply_chat_template(
            base_messages + extra_messages, tokenize=False, add_generation_prompt=False
        )
        text_without = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=False)
        return text_with[len(text_without) :]


class TestApplyChatTemplateWithTools:
    EXPECTED_PROMPT_WITHOUT_TOOLS = (
        "<|im_start|>user\n" "What's the weather in Paris?<|im_end|>\n" "<|im_start|>assistant\n"
    )

    EXPECTED_PROMPT_WITH_TOOLS = (
        "<|im_start|>system\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        '{"type": "function", "function": {"name": "get_year", "description": "Get current year", "parameters": {"type": "object", "properties": {}, "required": []}}}\n'
        '{"type": "function", "function": {"name": "get_temperature", "description": "Get temperature for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}\n'
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        "<|im_start|>user\n"
        "What's the weather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    @pytest.mark.parametrize(
        "tools,expected",
        [
            pytest.param(None, EXPECTED_PROMPT_WITHOUT_TOOLS, id="without_tools"),
            pytest.param(SAMPLE_TOOLS, EXPECTED_PROMPT_WITH_TOOLS, id="with_tools"),
        ],
    )
    def test_apply_chat_template(self, tools, expected):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)

        assert prompt == expected


class TestSGLangFunctionCallParser:
    """Test to demonstrate and ensure SGLang function call parser have features we need without breaking changes."""

    @pytest.mark.parametrize(
        "model_output,expected",
        [
            pytest.param(
                'Let me check for you.\n<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>',
                (
                    "Let me check for you.",
                    [ToolCallItem(tool_index=0, name="get_year", parameters="{}")],
                ),
                id="single_tool_call",
            ),
            pytest.param(
                "I will get year and temperature.\n"
                '<tool_call>\n{"name": "get_year", "arguments": {}}\n</tool_call>\n'
                '<tool_call>\n{"name": "get_temperature", "arguments": {"location": "Shanghai"}}\n</tool_call>',
                (
                    "I will get year and temperature.",
                    [
                        ToolCallItem(tool_index=0, name="get_year", parameters="{}"),
                        ToolCallItem(tool_index=1, name="get_temperature", parameters='{"location": "Shanghai"}'),
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
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected
