import pytest
from pydantic import TypeAdapter

from miles.rollout.generate_hub.tool_call_utils import tokenize_tool_response
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


# TODO add more models
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
    "stepfun-ai/step3",
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


SINGLE_TOOL_RESPONSE = {
    "role": "tool",
    "tool_call_id": "call_001",
    "content": '{"temperature": 25}',
    "name": "get_weather",
}

DOUBLE_TOOL_RESPONSES = [
    {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": '{"temperature": 25}',
        "name": "get_weather",
    },
    {
        "role": "tool",
        "tool_call_id": "call_002",
        "content": '{"results": ["A", "B"]}',
        "name": "search",
    },
]

# Expected values for each (model, num_tools) combination
# Format: (model_name, num_tools) -> (tool_response, expected_token_ids, expected_decoded_str)
EXPECTED_TOKENIZE_RESULTS = {
    # qwen/qwen25: Qwen2.5 family
    ("Qwen/Qwen2.5-0.5B-Instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("Qwen/Qwen2.5-0.5B-Instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    ("Qwen/Qwen2.5-7B-Instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("Qwen/Qwen2.5-7B-Instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # qwen3_coder: Qwen3 family
    ("Qwen/Qwen3-0.6B", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("Qwen/Qwen3-0.6B", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    ("Qwen/Qwen3-8B", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("Qwen/Qwen3-8B", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # llama3: Llama-3.2 family
    ("meta-llama/Llama-3.2-1B-Instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("meta-llama/Llama-3.2-1B-Instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    ("meta-llama/Llama-3.2-3B-Instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("meta-llama/Llama-3.2-3B-Instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # mistral: Mistral family
    ("mistralai/Mistral-7B-Instruct-v0.3", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("mistralai/Mistral-7B-Instruct-v0.3", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # deepseekv3: DeepSeek-V3 family
    ("deepseek-ai/DeepSeek-V3", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("deepseek-ai/DeepSeek-V3", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # glm: GLM-4 family
    ("THUDM/glm-4-9b-chat", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("THUDM/glm-4-9b-chat", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # kimi_k2: Kimi-K2
    ("moonshotai/Kimi-K2-Instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("moonshotai/Kimi-K2-Instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # mimo: MiMo
    ("XiaomiMiMo/MiMo-7B-RL", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("XiaomiMiMo/MiMo-7B-RL", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
    # interns1: InternLM
    ("internlm/internlm3-8b-instruct", 1): (
        SINGLE_TOOL_RESPONSE,
        [],  # TODO: fill after first run
        "",  # TODO: fill after first run
    ),
    ("internlm/internlm3-8b-instruct", 2): (
        DOUBLE_TOOL_RESPONSES,
        [[], []],  # TODO: fill after first run
        ["", ""],  # TODO: fill after first run
    ),
}


def _get_test_params():
    """Generate pytest parameters from EXPECTED_TOKENIZE_RESULTS."""
    params = []
    for (model_name, num_tools), (tool_resp, expected_ids, expected_str) in EXPECTED_TOKENIZE_RESULTS.items():
        params.append(
            pytest.param(
                model_name, num_tools, tool_resp, expected_ids, expected_str,
                id=f"{model_name.split('/')[-1]}-{num_tools}tool",
            )
        )
    return params


class TestTokenizeToolResponse:
    """Test tokenize_tool_response across different models and tool call counts."""

    @pytest.mark.parametrize(
        "model_name,num_tools,tool_response,expected_token_ids,expected_decoded_str",
        _get_test_params(),
    )
    def test_tokenize_tool_response(
        self, model_name, num_tools, tool_response, expected_token_ids, expected_decoded_str
    ):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if num_tools == 1:
            token_ids = tokenize_tool_response(tool_response, tokenizer)
            decoded_str = tokenizer.decode(token_ids)

            if expected_token_ids:
                assert token_ids == expected_token_ids
            if expected_decoded_str:
                assert decoded_str == expected_decoded_str

            print(f"\n[{model_name}] single tool response:")
            print(f"  token_ids = {token_ids}")
            print(f"  decoded   = {repr(decoded_str)}")

        else:
            all_token_ids = []
            all_decoded_strs = []
            for i, resp in enumerate(tool_response):
                token_ids = tokenize_tool_response(resp, tokenizer)
                decoded_str = tokenizer.decode(token_ids)
                all_token_ids.append(token_ids)
                all_decoded_strs.append(decoded_str)

                if expected_token_ids and expected_token_ids[i]:
                    assert token_ids == expected_token_ids[i]
                if expected_decoded_str and expected_decoded_str[i]:
                    assert decoded_str == expected_decoded_str[i]

            print(f"\n[{model_name}] double tool responses:")
            for i, (tids, dstr) in enumerate(zip(all_token_ids, all_decoded_strs)):
                print(f"  [{i}] token_ids = {tids}")
                print(f"  [{i}] decoded   = {repr(dstr)}")
