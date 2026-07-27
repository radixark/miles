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


class _BridgeJinjaTokenizer:
    name_or_path = ""
    chat_template = "bridge-jinja"
    eos_token = "<EOS>"

    rendered = "<BOS><USER>question<ASSISTANT></think>answer<EOS>"
    token_spans = [
        (0, "<BOS>"),
        (100, "<USER>"),
        (10, "question"),
        (101, "<ASSISTANT>"),
        (128822, "</think>"),
        (20, "answer"),
        (1, "<EOS>"),
    ]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        assert messages == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        assert add_generation_prompt is False
        assert tools is None
        if tokenize:
            assert return_dict is True
            assert return_assistant_tokens_mask is True
            return {
                "input_ids": [token_id for token_id, _ in self.token_spans],
                "assistant_masks": [0, 0, 0, 0, 0, 1, 1],
            }
        assert return_dict is False
        assert return_assistant_tokens_mask is False
        return self.rendered

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text == self.rendered
        assert add_special_tokens is False
        cursor = 0
        input_ids = []
        offset_mapping = []
        for token_id, token_text in self.token_spans:
            start = self.rendered.index(token_text, cursor)
            end = start + len(token_text)
            input_ids.append(token_id)
            offset_mapping.append((start, end))
            cursor = end
        output = {"input_ids": input_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offset_mapping
        return output

    def decode(self, token_ids):
        pieces = {token_id: token_text for token_id, token_text in self.token_spans}
        return "".join(pieces[token_id] for token_id in token_ids)


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


def test_loss_mask_deepseek_v4_jinja_matches_bridge_shifted_semantics(monkeypatch):
    def fail_official_encoder(*args, **kwargs):
        raise AssertionError("deepseek_v4_jinja must bypass the official encoder")

    monkeypatch.setattr(
        "miles.utils.mask_utils.chat_template_utils.apply_chat_template",
        fail_official_encoder,
    )

    mask_generator = MultiTurnLossMaskGenerator(_BridgeJinjaTokenizer(), tokenizer_type="deepseek_v4_jinja")
    token_ids, target_mask = mask_generator.get_loss_mask(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert token_ids == [0, 100, 10, 101, 128822, 20, 1]
    assert target_mask == [0, 0, 0, 0, 0, 1, 1]

    response_length = mask_generator.get_response_lengths([target_mask])[0]
    prompt_length = len(token_ids) - response_length
    effective_loss_mask = [0] * (prompt_length - 1) + target_mask[-response_length:] + [0]
    assert effective_loss_mask == [0, 0, 0, 0, 1, 1, 0]
    assert token_ids[4] == 128822  # </think> predicts the first assistant token.
    assert token_ids[-1] == 1 and effective_loss_mask[-1] == 0  # EOS is a target, not an input loss position.


def test_loss_mask_deepseek_v4_jinja_supervises_reasoning_and_empty_think_close():
    class ThinkingTokenizer(_BridgeJinjaTokenizer):
        chat_template = "bridge-thinking-jinja {% generation %}"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            tools=None,
            return_dict=False,
            return_assistant_tokens_mask=False,
        ):
            assert tokenize is True
            assert add_generation_prompt is False
            assert tools is None
            assert return_dict is True
            assert return_assistant_tokens_mask is True
            reasoning = messages[-1].get("reasoning_content") or ""
            token_ids = [0, 100, 10, 101, 128821]
            assistant_masks = [0, 0, 0, 0, 0]
            if reasoning:
                token_ids.append(30)
                assistant_masks.append(1)
            token_ids.extend([128822, 20, 1])
            assistant_masks.extend([1, 1, 1])
            return {"input_ids": token_ids, "assistant_masks": assistant_masks}

    mask_generator = MultiTurnLossMaskGenerator(ThinkingTokenizer(), tokenizer_type="deepseek_v4_jinja")
    reasoning_ids, reasoning_mask = mask_generator.get_loss_mask(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "reasoning_content": "reason", "content": "answer"},
        ]
    )
    assert reasoning_ids == [0, 100, 10, 101, 128821, 30, 128822, 20, 1]
    assert reasoning_mask == [0, 0, 0, 0, 0, 1, 1, 1, 1]

    empty_ids, empty_mask = mask_generator.get_loss_mask(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "reasoning_content": "", "content": "answer"},
        ]
    )
    assert empty_ids == [0, 100, 10, 101, 128821, 128822, 20, 1]
    assert empty_mask == [0, 0, 0, 0, 0, 1, 1, 1]


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
