from argparse import Namespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from miles.rollout.diffusion_gemma_sft import generate_rollout, tokenize_final_response
from miles.utils.types import Sample


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}</turn>" for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_real_fast_tokenizer_offsets_preserve_final_answer_boundary():
    backend = Tokenizer(WordLevel({"<unk>": 0, "question": 1, "answer": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<assistant>", "</turn>"]})
    tokenizer.chat_template = "{% for m in messages %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endfor %}"
    tokens, length = tokenize_final_response(
        tokenizer,
        messages=[{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
        template_kwargs={},
    )
    assert tokenizer.convert_ids_to_tokens(tokens[-length:]) == ["answer", "</turn>"]


def test_keeps_history_clean_and_supervises_only_final_answer_and_closure():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "answer"},
    ]
    tokens, length = tokenize_final_response(tokenizer, messages=messages, template_kwargs={})
    assert "".join(map(chr, tokens[-length:])) == "answer</turn>"
    assert "".join(map(chr, tokens)).endswith("<assistant>answer</turn>")


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "unfinished"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}],
        [{"role": "user", "content": [{"type": "image"}]}, {"role": "assistant", "content": "x"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "tool_calls": [{}]}],
    ],
)
def test_rejects_unsupported_training_rows(messages):
    with pytest.raises(ValueError):
        tokenize_final_response(CharacterTokenizer(), messages=messages, template_kwargs={})


def test_adapter_returns_existing_miles_sample_groups_without_generation(monkeypatch):
    sample = Sample(prompt=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])

    class Buffer:
        def get_samples(self, count):
            assert count == 1
            return [[sample]]

    monkeypatch.setattr("miles.rollout.diffusion_gemma_sft.load_tokenizer", lambda *a, **k: CharacterTokenizer())
    result = generate_rollout(
        Namespace(
            rollout_global_dataset=True,
            debug_train_only=True,
            rollout_batch_size=1,
            hf_checkpoint="tiny",
            chat_template_path=None,
            apply_chat_template_kwargs={},
        ),
        0,
        Buffer(),
    )
    assert result == [[sample]]
    assert sample.loss_mask == [1] * sample.response_length
    assert sample.reward == 0


def test_text_only_processor_placeholders_are_accepted(monkeypatch):
    sample = Sample(
        prompt=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        multimodal_inputs={"images": None, "videos": None},
    )
    buffer = type("Buffer", (), {"get_samples": lambda self, count: [[sample]]})()
    monkeypatch.setattr("miles.rollout.diffusion_gemma_sft.load_tokenizer", lambda *a, **k: CharacterTokenizer())
    args = Namespace(
        rollout_global_dataset=True,
        debug_train_only=True,
        rollout_batch_size=1,
        hf_checkpoint="tiny",
        chat_template_path=None,
        apply_chat_template_kwargs={},
    )
    assert generate_rollout(args, 0, buffer) == [[sample]]
