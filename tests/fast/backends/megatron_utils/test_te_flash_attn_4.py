"""CPU tests for --te-disable-flash-attn-4 (hide FlashAttention 4 from Transformer Engine)."""

import argparse
import ast
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_TE_UTILS = "transformer_engine.pytorch.attention.dot_product_attention.utils"


@pytest.fixture(scope="module")
def maybe_hide_te_flash_attn_4() -> Callable:
    # Execute the production helper without importing miles' GPU-only packages.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/misc_utils.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "maybe_hide_te_flash_attn_4"
    )
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["maybe_hide_te_flash_attn_4"]


@pytest.fixture
def flash_attention_utils(monkeypatch) -> type:
    # Stand in for TE's backend-selection module, so the test needs neither TE nor a GPU.
    flash_attention_utils = type("FlashAttentionUtils", (), {"v4_is_installed": True})
    module = types.ModuleType(_TE_UTILS)
    module.FlashAttentionUtils = flash_attention_utils
    monkeypatch.setitem(sys.modules, _TE_UTILS, module)
    return flash_attention_utils


def test_flag_hides_flash_attention_4(maybe_hide_te_flash_attn_4, flash_attention_utils):
    maybe_hide_te_flash_attn_4(argparse.Namespace(te_disable_flash_attn_4=True))
    assert flash_attention_utils.v4_is_installed is False


def test_flag_off_leaves_te_untouched(maybe_hide_te_flash_attn_4, flash_attention_utils):
    maybe_hide_te_flash_attn_4(argparse.Namespace(te_disable_flash_attn_4=False))
    assert flash_attention_utils.v4_is_installed is True


def test_missing_flag_leaves_te_untouched(maybe_hide_te_flash_attn_4, flash_attention_utils):
    maybe_hide_te_flash_attn_4(argparse.Namespace())
    assert flash_attention_utils.v4_is_installed is True
