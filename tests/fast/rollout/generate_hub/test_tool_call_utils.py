from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionMessageToolCall
from sglang.srt.function_call.core_types import ToolCallItem

from miles.rollout.generate_utils.tool_call_utils import (
    _DUMMY_USER,
    _build_dummy_assistant,
    _execute_tool_call,
    tokenize_tool_responses,
)
from miles.utils.processing_utils import load_tokenizer

TOOL_CALL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-Coder-Next",
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo, requires HF_TOKEN in CI
    "mistralai/Mistral-7B-Instruct-v0.3",
    "MiniMaxAI/MiniMax-M2.5",
    "MiniMaxAI/MiniMax-M2.7",
    "internlm/internlm3-8b-instruct",
    "zai-org/GLM-4.7-Flash",
    # "stepfun-ai/Step-3.5-Flash",  # Disabled: config num_hidden_layers/layer_types mismatch breaks huggingface_hub strict validation
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "XiaomiMiMo/MiMo-7B-RL",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]

# Models that fail decode round-trip under transformers>=5.x due to upstream tokenizer issues.
# These are excluded from TOOL_CALL_TEST_MODELS but listed here for tracking.
# - DeepSeek-V3, step3: transformers v5 unified LlamaTokenizer overwrites their ByteLevel
#   pre_tokenizer/decoder with Metaspace, causing decode(encode(text)) != text.
#   See https://github.com/huggingface/transformers/issues/43066
# - DeepSeek-V3.1: its tool-call chat template concatenates function.arguments as a string,
#   but our dummy tool-call shape provides a dict, raising TypeError before the round-trip check.
# - glm-4-9b-chat: v5 removed the legacy _decode special-token segmentation, exposing a bug in
#   the model's custom convert_tokens_to_string (doesn't handle str-type special tokens).
TOOL_CALL_KNOWN_FAILURES = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.1",
    "stepfun-ai/step3",
    "THUDM/glm-4-9b-chat",
]

SINGLE_TOOL_CALL_ONLY_MODELS = [
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo
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


def _make_tool_call(call_type: str, arguments: str) -> ToolCallItem | ChatCompletionMessageToolCall:
    if call_type == "sglang":
        return ToolCallItem(tool_index=0, name="bash", parameters=arguments)
    return ChatCompletionMessageToolCall(
        id="call_test",
        type="function",
        function={"name": "bash", "arguments": arguments},
    )


class TestExecuteToolCall:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["sglang", "openai"])
    @pytest.mark.parametrize(
        ("arguments", "expected_params"),
        [
            ('{"cmd": "pytest"}', {"cmd": "pytest"}),
            ("", {}),
        ],
    )
    async def test_valid_object_arguments_reach_executor(self, call_type, arguments, expected_params):
        execute_one = AsyncMock(return_value="ok")

        message = await _execute_tool_call(_make_tool_call(call_type, arguments), execute_one)

        execute_one.assert_awaited_once_with("bash", expected_params)
        assert message["role"] == "tool"
        assert message["content"] == "ok"
        assert message["name"] == "bash"
        assert message["tool_call_id"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", ["sglang", "openai"])
    @pytest.mark.parametrize(
        "arguments",
        [
            pytest.param('{"cmd": "pytest"', id="malformed"),
            pytest.param("[1, 2]", id="list"),
            pytest.param("42", id="number"),
        ],
    )
    async def test_invalid_arguments_return_tool_error_without_execution(self, call_type, arguments):
        execute_one = AsyncMock(return_value="must not run")

        message = await _execute_tool_call(_make_tool_call(call_type, arguments), execute_one)

        execute_one.assert_not_awaited()
        assert message["role"] == "tool"
        assert message["content"] == "Error: Tool arguments must be a valid JSON object."
        assert message["name"] == "bash"
        assert message["tool_call_id"]


class TestTokenizeToolResponses:
    @pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
    def test_snapshot(self, model_name):
        tokenizer = load_tokenizer(model_name, trust_remote_code=True)
        token_ids = tokenize_tool_responses(SAMPLE_TOOL_RESPONSES, tokenizer)
        decoded = tokenizer.decode(token_ids)

        assert decoded == (
            "<|im_start|>user\n"
            "<tool_response>\n"
            '{"year": 2026}\n'
            "</tool_response>\n"
            "<tool_response>\n"
            '{"temperature": 25}\n'
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @pytest.mark.parametrize("num_tools", [1, 2])
    @pytest.mark.parametrize("model_name", TOOL_CALL_TEST_MODELS)
    def test_tokenize_tool_responses(self, model_name, num_tools):
        if num_tools > 1 and model_name in SINGLE_TOOL_CALL_ONLY_MODELS:
            pytest.skip(f"{model_name} only supports single tool call")

        tokenizer = load_tokenizer(model_name, trust_remote_code=True)

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
            base_messages + extra_messages, tokenize=False, add_generation_prompt=True
        )
        text_without = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=False)
        return text_with[len(text_without) :]
