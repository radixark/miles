"""load_processor falls back to the engine's processor for a model transformers cannot resolve."""

from __future__ import annotations

import json
import sys
import types

import pytest

from miles.utils import processing_utils


class _Tokenizer:
    pass


def _checkpoint(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(tmp_path)


@pytest.fixture
def bare_tokenizer_from_auto(monkeypatch):
    """transformers without the model's processor: AutoProcessor hands back a tokenizer."""
    monkeypatch.setattr(processing_utils, "PreTrainedTokenizerBase", _Tokenizer)
    monkeypatch.setattr(
        processing_utils.AutoProcessor, "from_pretrained", staticmethod(lambda path, **kw: _Tokenizer())
    )


def _install_fake_sglang(monkeypatch, loaded):
    class Glm5NextProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loaded.append((path, kwargs))
            return cls()

    mod = types.ModuleType("sglang.srt.configs.glm5_next_processing")
    mod.Glm5NextProcessor = Glm5NextProcessor
    for name in ("sglang", "sglang.srt", "sglang.srt.configs"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.glm5_next_processing", mod)
    return Glm5NextProcessor


def test_glm5_next_gets_the_engine_processor_when_transformers_returns_a_tokenizer(
    tmp_path, monkeypatch, bare_tokenizer_from_auto
):
    loaded = []
    cls = _install_fake_sglang(monkeypatch, loaded)
    proc = processing_utils.load_processor(_checkpoint(tmp_path, "glm5_next"), trust_remote_code=True)
    assert isinstance(proc, cls)
    assert loaded == [(str(tmp_path), {"trust_remote_code": True})]


def test_other_models_still_get_none_for_a_tokenizer(tmp_path, monkeypatch, bare_tokenizer_from_auto):
    loaded = []
    _install_fake_sglang(monkeypatch, loaded)
    assert processing_utils.load_processor(_checkpoint(tmp_path, "qwen3_5")) is None
    assert loaded == []


def test_glm5_next_without_sglang_gets_none(tmp_path, monkeypatch, bare_tokenizer_from_auto):
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.glm5_next_processing", None)  # import raises ImportError
    assert processing_utils.load_processor(_checkpoint(tmp_path, "glm5_next")) is None


def test_a_real_processor_from_transformers_is_kept(tmp_path, monkeypatch):
    class Proc(processing_utils.ProcessorMixin):
        def __init__(self):
            pass

    monkeypatch.setattr(processing_utils.AutoProcessor, "from_pretrained", staticmethod(lambda path, **kw: Proc()))
    assert isinstance(processing_utils.load_processor(_checkpoint(tmp_path, "glm5_next")), Proc)
