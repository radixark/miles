"""DeepSeek V4.1 detection and TITO binding; rendering is covered with V4 in
``test_deepseek_v4.py``."""

from __future__ import annotations

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="stage-a-cpu", labels=[])

import json

from miles.utils.chat_template_utils import deepseek
from miles.utils.chat_template_utils.tito_tokenizer import (
    DeepSeekV4TITOTokenizer,
    DeepSeekV41TITOTokenizer,
    TITOTokenizerType,
)


class _FakeTokenizer:
    def __init__(self, name_or_path: str):
        self.name_or_path = name_or_path

    def convert_tokens_to_ids(self, token):
        return hash(token) % 1000


def _tok_with_model_type(tmp_path, model_type: str) -> _FakeTokenizer:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return _FakeTokenizer(str(tmp_path))


def test_detect_by_config(tmp_path):
    assert deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v41")) == "deepseek_v41"
    assert deepseek._FAMILIES["deepseek_v41"] is deepseek.V41


def test_v4_is_not_v41(tmp_path):
    assert deepseek._FAMILIES[deepseek.model_type(_tok_with_model_type(tmp_path, "deepseek_v4"))] is deepseek.V4


def test_tito_binding(tmp_path):
    assert TITOTokenizerType.get_tokenizer_class(TITOTokenizerType.DEEPSEEKV41) is DeepSeekV41TITOTokenizer
    assert issubclass(DeepSeekV41TITOTokenizer, DeepSeekV4TITOTokenizer)
    assert (DeepSeekV41TITOTokenizer.reasoning_parser, DeepSeekV41TITOTokenizer.tool_call_parser) == (
        "deepseek-v41",
        "deepseekv41",
    )
    tok = _tok_with_model_type(tmp_path, "deepseek_v41")
    tito = DeepSeekV41TITOTokenizer(tok, chat_template_kwargs={"enable_thinking": False})
    assert tito.chat_template_kwargs["thinking"] is False
