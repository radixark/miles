from typing import Any
from transformers import AutoTokenizer
from miles.rollout.generate_utils.tool_call_utils import tokenize_tool_responses


_DUMMY_SYSTEM = {"role": "system", "content": "FOR CALCULATING ADDITIONAL TOKENS ONLY"}
_DUMMY_USER = {"role": "user", "content": "FOR CALCULATING ADDITIONAL TOKENS ONLY"}
_DUMMY_ASSISTANT = {"role": "assistant", "content": "FOR CALCULATING ADDITIONAL TOKENS ONLY"}


def calc_generation_prompt_tokens(tokenizer: AutoTokenizer) -> list[int]:
    messages = [_DUMMY_SYSTEM, _DUMMY_USER, _DUMMY_ASSISTANT]
    with_generation_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    without_generation_prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    assert with_generation_prompt[: len(without_generation_prompt)] == without_generation_prompt
    return with_generation_prompt[len(without_generation_prompt) :]


# TODO(jiajun): need e2e test to validate. According to https://zhuanlan.zhihu.com/p/1917126584806139373
# Notice: This function will automatically trim think tokens if the model's chat template trim thinking parts. Like Qwen3.
def _naive_calc_additional_tokens(
    message: dict[str, Any], tokenizer: AutoTokenizer, add_generation_prompt: bool = True
) -> list[int]:
    prefix = [_DUMMY_SYSTEM, _DUMMY_USER, _DUMMY_ASSISTANT, _DUMMY_USER]
    suffix = [_DUMMY_SYSTEM, _DUMMY_USER]
    prefix_tokens = tokenizer.apply_chat_template(prefix, tokenize=True)
    messages_tokens = tokenizer.apply_chat_template(prefix + [message] + suffix, tokenize=True)
    suffix_tokens = tokenizer.apply_chat_template(suffix, tokenize=True)

    response_tokens = messages_tokens[len(prefix_tokens) : -len(suffix_tokens)]
    generation_prompt_tokens = calc_generation_prompt_tokens(tokenizer)
    return response_tokens + generation_prompt_tokens


# TODO(jiajun): need e2e test to validate.
def tokenize_messages(
    messages: list[dict[str, Any]],
    tokenizer,
    add_generation_prompt: bool = True,
) -> list[int]:
    token_ids = []
    for message in messages:
        if message["role"] == "assistant" or message["role"] == "user" or message["role"] == "system":
            token_ids.extend(_naive_calc_additional_tokens(message, tokenizer, add_generation_prompt))
        elif message["role"] == "tool":
            token_ids.extend(tokenize_tool_responses([message], tokenizer))
        else:
            raise ValueError(f"Unsupported message role: {message['role']}")

    return token_ids
