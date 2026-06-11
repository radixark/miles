import importlib.util
import sys
from pathlib import Path

import pytest


def load_backend_module():
    module_path = Path(__file__).resolve().parents[2] / "miles_plugins" / "models" / "qwen_gdn_backend.py"
    spec = importlib.util.spec_from_file_location("test_qwen_gdn_backend_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unknown_backend_raises_value_error():
    module = load_backend_module()
    with pytest.raises(ValueError, match="Unsupported Qwen GDN backend"):
        module.get_chunk_gated_delta_rule("nope")


def test_fla_backend_returns_fla_kernel():
    pytest.importorskip("fla.ops.gated_delta_rule")
    module = load_backend_module()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    assert module.get_chunk_gated_delta_rule("fla") is chunk_gated_delta_rule


def test_flashqla_backend_errors_clearly_when_missing():
    module = load_backend_module()
    if "flash_qla" in sys.modules or importlib.util.find_spec("flash_qla") is not None:
        pytest.skip("flash_qla installed; missing-package error path not applicable")
    with pytest.raises(ImportError, match="FlashQLA"):
        module.get_chunk_gated_delta_rule("flashqla")


def test_version_parser_handles_local_suffixes():
    module = load_backend_module()
    assert module._parse_version("2.11.0+cu130") == (2, 11)
    assert module._parse_version("13.0") == (13, 0)
    assert module._parse_version("2") == (2, 0)
