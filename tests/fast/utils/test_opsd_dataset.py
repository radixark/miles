import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.on_policy_self_distillation import build_teacher_prompt
from miles.utils.data import Dataset

register_cpu_ci(est_time=30, suite="stage-a-cpu")


class _Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def __call__(self, prompts, add_special_tokens):
        return {"input_ids": [self.encode(prompt, add_special_tokens) for prompt in prompts]}


def test_dataset_materializes_and_length_filters_privileged_teacher_prompt(monkeypatch):
    rows = [{"text": "2 + 2?", "label": "4", "metadata": {"source": "unit"}}]
    monkeypatch.setattr("miles.utils.data.read_file", lambda _: iter(rows))

    dataset = Dataset(
        "unused.jsonl",
        tokenizer=_Tokenizer(),
        processor=None,
        max_length=1000,
        label_key="label",
        teacher_prompt_builder=build_teacher_prompt,
    )

    sample = dataset[0]
    expected_prompt = build_teacher_prompt("2 + 2?", "4", {"source": "unit"})
    assert sample.privileged_prompt_tokens == _Tokenizer().encode(expected_prompt, add_special_tokens=False)

    filtered_dataset = Dataset(
        "unused.jsonl",
        tokenizer=_Tokenizer(),
        processor=None,
        max_length=len("2 + 2?"),
        label_key="label",
        teacher_prompt_builder=build_teacher_prompt,
    )
    assert len(filtered_dataset) == 0


def test_dataset_calls_custom_teacher_prompt_builder_by_keyword(monkeypatch):
    rows = [{"text": "2 + 2?", "label": "4", "metadata": {"source": "unit"}}]
    monkeypatch.setattr("miles.utils.data.read_file", lambda _: iter(rows))
    received = {}

    def teacher_prompt_builder(*, prompt, label, metadata):
        received.update(prompt=prompt, label=label, metadata=metadata)
        return "privileged"

    dataset = Dataset(
        "unused.jsonl",
        tokenizer=_Tokenizer(),
        processor=None,
        max_length=None,
        label_key="label",
        teacher_prompt_builder=teacher_prompt_builder,
    )

    assert len(dataset) == 1
    assert received == {
        "prompt": "2 + 2?",
        "label": "4",
        "metadata": {"source": "unit"},
    }


def test_dataset_rejects_tool_enabled_opsd_samples(monkeypatch):
    rows = [
        {
            "text": "Use the tool.",
            "label": "done",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ]
    monkeypatch.setattr("miles.utils.data.read_file", lambda _: iter(rows))

    with pytest.raises(ValueError, match="tool-enabled"):
        Dataset(
            "unused.jsonl",
            tokenizer=_Tokenizer(),
            processor=None,
            max_length=None,
            label_key="label",
            tool_key="tools",
            teacher_prompt_builder=build_teacher_prompt,
        )


def test_dataset_renders_teacher_conversation_with_independent_template_kwargs(monkeypatch):
    rows = [{"text": "2 + 2?", "label": "4"}]
    monkeypatch.setattr("miles.utils.data.read_file", lambda _: iter(rows))
    captured = {}

    def render(messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return "teacher-rendered"

    monkeypatch.setattr("miles.utils.data.chat_template_utils.apply_chat_template", render)

    dataset = Dataset(
        "unused.jsonl",
        tokenizer=_Tokenizer(),
        processor=None,
        max_length=None,
        label_key="label",
        teacher_prompt_builder=lambda **_: [{"role": "user", "content": "privileged"}],
        teacher_chat_template_kwargs={"enable_thinking": False},
    )

    assert len(dataset) == 1
    assert captured["messages"] == [{"role": "user", "content": "privileged"}]
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["kwargs"]["add_generation_prompt"] is True
    assert dataset[0].privileged_prompt_tokens == _Tokenizer().encode(
        "teacher-rendered",
        add_special_tokens=False,
    )
