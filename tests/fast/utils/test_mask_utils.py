from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


class _CharOffsetTokenizer:
    name_or_path = ""
    eos_token = "<｜end▁of▁sentence｜>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        output = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return output

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


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


def test_loss_mask_deepseek_v4_masks_full_assistant_content(monkeypatch):
    assistant_content = '<function_calls>{"name":"terminal"}</function_calls>'
    rendered = f"<user>show tool calls</user><assistant>{assistant_content}<｜end▁of▁sentence｜>"

    def fake_apply_chat_template(messages, tokenizer, tools=None, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return rendered

    monkeypatch.setattr(
        "miles.utils.mask_utils.chat_template_utils.apply_chat_template",
        fake_apply_chat_template,
    )

    mask_generator = MultiTurnLossMaskGenerator(_CharOffsetTokenizer(), tokenizer_type="deepseek_v4")
    token_ids, loss_mask = mask_generator.get_loss_mask(
        [
            {"role": "user", "content": "show tool calls"},
            {"role": "assistant", "content": assistant_content},
        ]
    )

    assert len(token_ids) == len(loss_mask)
    assert mask_generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        assistant_content + "<｜end▁of▁sentence｜>"
    ]


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
