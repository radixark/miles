import pytest


from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


def test_loss_mask_qwen3_simple(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 1, f"Expected 1 text, got {len(selected_texts)}"

    print(f"==== Single Turn Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_loss_mask_qwen3_tools(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {
            "role": "assistant",
            "content": "I WILL CALL terminal",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
            ],
        },
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Perform operations from the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute as `bash -c <command>`",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the command for the user.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file given its path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be read.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]

    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages, tools)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 2, f"Expected 2 texts, got {len(selected_texts)}"

    print(f"==== Multi-turn with Tools Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")


def _distill_mask_generator(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") -> MultiTurnLossMaskGenerator:
    # The default `qwen` type routes DeepSeek-style tokenizers to the distill path.
    return MultiTurnLossMaskGenerator(load_tokenizer(model_name), tokenizer_type="qwen")


def test_loss_mask_distill_keeps_context_and_trains_eos():
    mask_generator = _distill_mask_generator()
    tokenizer = mask_generator.tokenizer
    messages = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "WHAT IS 2+2"},
        {"role": "assistant", "content": "FIRST ANSWER"},
        {"role": "user", "content": "AND 3+3"},
        {"role": "assistant", "content": "<think>\nadd them\n</think>\n\nSIX"},
    ]
    token_ids, loss_mask = mask_generator.get_loss_mask(messages)
    assert len(token_ids) == len(loss_mask)

    assert tokenizer.decode(token_ids).count("<think>") == 1

    prompt_text = tokenizer.decode([t for t, m in zip(token_ids, loss_mask, strict=True) if m == 0])
    for content in ("You are a math tutor.", "WHAT IS 2+2", "FIRST ANSWER", "AND 3+3"):
        assert content in prompt_text

    trained_ids = [t for t, m in zip(token_ids, loss_mask, strict=True) if m == 1]
    assert trained_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(trained_ids[:-1]) == "add them\n</think>\n\nSIX"


def test_loss_mask_distill_trims_untrainable_trailing_messages():
    mask_generator = _distill_mask_generator()
    tokenizer = mask_generator.tokenizer
    tool_call = {"function": {"name": "ls", "arguments": "{}"}, "id": "call_0", "type": "function"}
    messages = [
        {"role": "user", "content": "LIST FILES"},
        {"role": "assistant", "content": "SURE"},
        {"role": "user", "content": "NOW RUN IT"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "TOOLOUT"},
    ]
    token_ids, loss_mask = mask_generator.get_loss_mask(messages)
    assert "TOOLOUT" not in tokenizer.decode(token_ids)
    trained_ids = [t for t, m in zip(token_ids, loss_mask, strict=True) if m == 1]
    assert tokenizer.decode(trained_ids) == "SURE" + tokenizer.eos_token


def test_loss_mask_distill_rejects_sample_without_plain_assistant_answer():
    mask_generator = _distill_mask_generator()
    messages = [
        {"role": "user", "content": "LIST FILES"},
        {"role": "tool", "content": "TOOLOUT"},
    ]
    with pytest.raises(ValueError, match="assistant message without tool_calls"):
        mask_generator.get_loss_mask(messages)


def test_loss_mask_distill_skips_masked_answers():
    mask_generator = _distill_mask_generator()
    tokenizer = mask_generator.tokenizer
    tool_call = {"function": {"name": "ls", "arguments": "{}"}, "id": "call_0", "type": "function"}
    tail = [
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "OUT"},
    ]

    # The masked answer is trimmed like any other untrainable tail; the earlier answer is trained.
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "KEEP"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "OLD", "step_loss_mask": 0},
        *tail,
    ]
    token_ids, loss_mask = mask_generator.get_loss_mask(messages)
    trained_ids = [t for t, m in zip(token_ids, loss_mask, strict=True) if m == 1]
    assert tokenizer.decode(trained_ids) == "KEEP" + tokenizer.eos_token

    # With no unmasked answer left, the row is rejected instead of yielding an all-zero mask.
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "OLD", "step_loss_mask": 0}, *tail]
    with pytest.raises(ValueError, match="step_loss_mask=1"):
        mask_generator.get_loss_mask(messages)
