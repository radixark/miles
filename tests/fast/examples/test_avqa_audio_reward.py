import asyncio
from types import SimpleNamespace

import pytest

from examples.avqa_audio import reward
from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.types import Sample

CONTROL = {
    "<|message_model|>": 1,
    "<|content_thinking|>": 2,
    "<|content_text|>": 3,
    "<|content_invoke_tool_json|>": 4,
    "<|content_invoke_tool_text|>": 5,
    "<|end_message|>": 6,
    "<|content_model_end_sampling|>": 7,
}


class FakeTokenizer:
    """One id per character; the Inkling control tokens take the ids below 32."""

    all_special_ids = list(CONTROL.values())
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return CONTROL[token]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def completion(*blocks: tuple[str, str], ended: bool = True) -> list[int]:
    ids = []
    for index, (kind, text) in enumerate(blocks):
        if index:
            ids.append(CONTROL["<|message_model|>"])
        ids.append(CONTROL[f"<|content_{kind}|>"])
        ids.extend(ord(c) for c in text)
        if ended or index < len(blocks) - 1:
            ids.append(CONTROL["<|end_message|>"])
    if ended:
        ids.append(CONTROL["<|content_model_end_sampling|>"])
    return ids


def sample(response_tokens: list[int], label: str) -> Sample:
    prompt = [ord(c) for c in "question"]
    return Sample(tokens=prompt + response_tokens, response_length=len(response_tokens), label=label)


@pytest.fixture
def parser(monkeypatch):
    fake = InklingResponseParser(FakeTokenizer())
    monkeypatch.setattr(reward, "_parser", lambda hf_checkpoint, chat_template_path: fake)
    return fake


ARGS = SimpleNamespace(hf_checkpoint="inkling", chat_template_path=None)


def grade(response_tokens, label):
    return asyncio.run(reward.reward_func(ARGS, sample(response_tokens, label)))


def test_grades_the_visible_letter_not_the_thinking(parser):
    tokens = completion(("thinking", "The correct option is (B)... no, a siren means a fire truck."), ("text", "A"))
    assert grade(tokens, "A") == 1.0
    assert grade(tokens, "B") == 0.0


def test_truncated_thinking_has_no_answer(parser):
    tokens = completion(("thinking", "Answer: A. Listening to the clip, I hear"), ended=False)
    assert reward.visible_content(parser, sample(tokens, "A")) == ""
    assert grade(tokens, "A") == 0.0


def test_only_the_response_tokens_are_parsed(parser):
    tokens = completion(("text", "C"))
    assert reward.visible_content(parser, sample(tokens, "C")) == "C"
