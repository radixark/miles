"""--lora-alpha is the source adapter's when there is one; the flag may only repeat it."""

import json
from types import SimpleNamespace

import pytest

from miles.utils.lora.arguments import _resolve_lora_alpha


def _args(path, alpha):
    return SimpleNamespace(lora_adapter_path=None if path is None else str(path), lora_alpha=alpha)


def test_without_a_source_the_default_is_16():
    assert _resolve_lora_alpha(_args(None, None)) == 16.0
    assert _resolve_lora_alpha(_args(None, 8.0)) == 8.0


def test_a_source_alpha_is_adopted_and_a_different_flag_refused(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32}))
    assert _resolve_lora_alpha(_args(tmp_path, None)) == 32
    assert _resolve_lora_alpha(_args(tmp_path, 32.0)) == 32
    with pytest.raises(AssertionError, match="differs"):
        _resolve_lora_alpha(_args(tmp_path, 16.0))


def test_a_source_without_alpha_needs_the_flag(tmp_path):
    with pytest.raises(AssertionError, match="pass --lora-alpha"):
        _resolve_lora_alpha(_args(tmp_path, None))
    assert _resolve_lora_alpha(_args(tmp_path, 4.0)) == 4.0
