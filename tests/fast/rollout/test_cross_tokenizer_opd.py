"""Unit tests for cross-tokenizer OPD reward-path helpers.

Regression coverage for `_teacher_prompt_ids`: ``apply_chat_template(tokenize=True)``
returns a BatchEncoding (dict-like) for some tokenizers, not a flat ``list[int]`` —
iterating it would yield string keys and the teacher SGLang server rejects the
request. We pass ``return_dict=True`` and read ``input_ids`` instead.
"""

from tests.ci.ci_register import register_cpu_ci

from miles.rollout.cross_tokenizer_opd import _teacher_prompt_ids

register_cpu_ci(est_time=20, suite="stage-a-cpu")


class _DictTemplateTokenizer:
    """Mimics a tokenizer whose apply_chat_template returns a BatchEncoding-like dict."""

    def __init__(self, input_ids):
        self._input_ids = input_ids

    def apply_chat_template(self, conversation, **kwargs):
        # The reward path must request a dict and the generation prompt.
        assert kwargs.get("return_dict") is True
        assert kwargs.get("tokenize") is True
        assert kwargs.get("add_generation_prompt") is True
        return {"input_ids": self._input_ids, "attention_mask": [1, 1, 1]}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, 102]


def test_reads_input_ids_from_batchencoding_dict():
    tok = _DictTemplateTokenizer([5, 6, 7])
    assert _teacher_prompt_ids(tok, [{"role": "user", "content": "hi"}], None) == [5, 6, 7]


def test_flattens_batched_input_ids():
    tok = _DictTemplateTokenizer([[8, 9, 10]])
    assert _teacher_prompt_ids(tok, [{"role": "user", "content": "hi"}], None) == [8, 9, 10]


def test_string_prompt_uses_encode():
    tok = _DictTemplateTokenizer([1, 2])
    assert _teacher_prompt_ids(tok, "raw prompt text", None) == [101, 102]
