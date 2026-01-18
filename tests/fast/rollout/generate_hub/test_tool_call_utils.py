import pytest

from miles.rollout.generate_utils.tool_call_utils import _DUMMY_USER, _build_dummy_assistant, tokenize_tool_responses

TOOL_CALL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo, requires HF_TOKEN in CI
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
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo
]

# Models where tokenize->decode produces extra whitespace vs direct string diff
TOKENIZE_DECODE_WHITESPACE_DIFF_MODELS = [
    "THUDM/glm-4-9b-chat",
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
    @pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
    def test_snapshot(self, model_name):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        tool_responses = SAMPLE_TOOL_RESPONSES[:num_tools]
        assert len(tool_responses) == num_tools

        actual_token_ids = tokenize_tool_responses(tool_responses, tokenizer)
        actual_str = tokenizer.decode(actual_token_ids)

        dummy_assistant = _build_dummy_assistant(tool_responses)
        base_messages = [_DUMMY_USER, dummy_assistant]
        expected_str = self._compute_chat_template_diff(base_messages, tool_responses, tokenizer)

        if model_name in TOKENIZE_DECODE_WHITESPACE_DIFF_MODELS:
            # Some models produce whitespace differences between tokenize->decode and direct string diff
            actual_str = actual_str.replace(" ", "")
            expected_str = expected_str.replace(" ", "")

        assert actual_str == expected_str, f"{model_name=}"

    @staticmethod
    def _compute_chat_template_diff(base_messages, extra_messages, tokenizer) -> str:
        text_with = tokenizer.apply_chat_template(
            base_messages + extra_messages, tokenize=False, add_generation_prompt=True
        )
        text_without = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=False)
        return text_with[len(text_without) :]
