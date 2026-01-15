import pytest
from pydantic import TypeAdapter

from miles.rollout.generate_hub.tool_call_utils import tokenize_tool_response
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


# Typical models that support tool calling, mapped from sglang tool call parsers.
TOOL_CALL_MODELS = [
    # qwen/qwen25: Qwen2.5 family
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # qwen3_coder: Qwen3 family
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
    # llama3: Llama-3.2 family
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # mistral: Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3",
    # deepseekv3/v31/v32: DeepSeek-V3 family
    "deepseek-ai/DeepSeek-V3",
    # glm/glm45/glm47: GLM-4 family
    "THUDM/glm-4-9b-chat",
    # kimi_k2: Kimi-K2
    "moonshotai/Kimi-K2-Instruct",
    # mimo: MiMo
    "XiaomiMiMo/MiMo-7B-RL",
    # step3: Step-3
    # "StepFun/Step-3",  # TODO: find correct HF repo
    # minimax-m2: MiniMax-M2
    # "MiniMaxAI/MiniMax-M2",  # TODO: find correct HF repo
    # interns1: InternLM
    "internlm/internlm3-8b-instruct",
]


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


class TestApplyChatTemplateWithTools:
    EXPECTED_PROMPT_WITHOUT_TOOLS = (
        "<|im_start|>user\n"
        "What's the weather in Paris?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    EXPECTED_PROMPT_WITH_TOOLS = (
        "<|im_start|>system\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        '{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}\n'
        '{"type": "function", "function": {"name": "search", "description": "Search for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}\n'
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

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        assert prompt == expected


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
        tools = TypeAdapter(list[Tool]).validate_python(SAMPLE_TOOLS)
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        assert parser.parse_non_stream(model_output) == expected


class TestTokenizeToolResponse:
    """Test tokenize_tool_response across different models and tool call counts."""

    @pytest.fixture
    def single_tool_response(self):
        return {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": '{"temperature": 25, "condition": "sunny"}',
            "name": "get_weather",
        }

    @pytest.fixture
    def double_tool_responses(self):
        return [
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": '{"temperature": 25}',
                "name": "get_weather",
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": '{"results": ["restaurant A", "restaurant B"]}',
                "name": "search",
            },
        ]

    @pytest.mark.parametrize("model_name", TOOL_CALL_MODELS)
    def test_single_tool_response(self, model_name, single_tool_response):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        token_ids = tokenize_tool_response(single_tool_response, tokenizer)

        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(t, int) for t in token_ids)

        decoded = tokenizer.decode(token_ids)
        assert single_tool_response["content"] in decoded or "temperature" in decoded

    @pytest.mark.parametrize("model_name", TOOL_CALL_MODELS)
    def test_double_tool_responses(self, model_name, double_tool_responses):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        all_token_ids = []
        for tool_response in double_tool_responses:
            token_ids = tokenize_tool_response(tool_response, tokenizer)

            assert isinstance(token_ids, list)
            assert len(token_ids) > 0
            assert all(isinstance(t, int) for t in token_ids)

            all_token_ids.append(token_ids)

        assert len(all_token_ids) == 2
        assert all_token_ids[0] != all_token_ids[1]

    @pytest.mark.parametrize("model_name", TOOL_CALL_MODELS)
    def test_token_consistency(self, model_name, single_tool_response):
        """Verify that tokenizing the same message twice gives consistent results."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        token_ids_1 = tokenize_tool_response(single_tool_response, tokenizer)
        token_ids_2 = tokenize_tool_response(single_tool_response, tokenizer)

        assert token_ids_1 == token_ids_2
