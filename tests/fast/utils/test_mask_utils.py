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
    assert all_token_ids == tokenizer.apply_chat_template(messages, tools=tools, preserve_thinking=True, return_dict=False)

    print(f"==== Multi-turn with Tools Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_qwen3_grouped_tools_preserve_thinking(model_name: str = "Qwen/Qwen3-8B") -> None:
    tokenizer = load_tokenizer(model_name)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    for count in (1, 2, 3):
        messages = [
            {"role": "user", "content": "Inspect the files."},
            {"role": "assistant", "reasoning_content": "I should inspect first.", "content": "Checking."},
            *[{"role": "tool", "content": f"TOOL_RESULT_{i}"} for i in range(count)],
            {"role": "assistant", "reasoning_content": "The inspection is complete.", "content": "Done."},
            {"role": "user", "content": "Explain."},
            {"role": "assistant", "content": "MASKED_ANSWER", "step_loss_mask": 0},
        ]
        tokens, mask = generator.get_loss_mask(messages)
        assert tokens == tokenizer.apply_chat_template(messages, preserve_thinking=True, return_dict=False)
        assert len(tokens) == len(mask)
        selected = "".join(generator.get_text_from_loss_mask(tokens, mask))
        assert "I should inspect first." in selected
        assert "The inspection is complete." in selected
        assert "Done." in selected
        assert "TOOL_RESULT_" not in selected
        assert "Inspect the files." not in selected
        assert "MASKED_ANSWER" not in selected


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
