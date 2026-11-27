"""TOP DeepSeek-V4 attention parameters under Megatron's stock names export to the HF names serving loads.

The stock converter takes these parameters directly. Only serving's fused and processed forms (wqkv_a, the
compressor's wkv_gate and APE, wo_a) go through the export hook.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf, convert_to_hf_stock
from miles.backends.megatron_utils.megatron_to_hf.export_hooks import export_hooks, register_export_hooks
from miles_plugins.top import dsv4_export

_PREFIX = "module.module.decoder.layers.3.self_attention."
_ARGS = SimpleNamespace(backend=Namespace(vocab_size=129_280, q_lora_rank=8))
# The stock name a TOP parameter carries -> the HF tensor serving loads.
_STOCK = [
    ("linear_q_up_proj.weight", "self_attn.wq_b.weight"),
    ("linear_proj.weight", "self_attn.wo_b.weight"),
    ("q_layernorm.weight", "self_attn.q_norm.weight"),
    ("kv_layernorm.weight", "self_attn.kv_norm.weight"),
    ("core_attention.attn_sink", "self_attn.attn_sink"),
    ("core_attention.compressor.norm.weight", "self_attn.compressor.norm.weight"),
]


@pytest.fixture
def ds4_export_hook():
    saved = vars(export_hooks()).copy()
    dsv4_export.register()
    yield
    register_export_hooks(**saved)


def _export(name, param):
    return [(hf, tensor) for hf, tensor in convert_to_hf(_ARGS, "deepseekv4", _PREFIX + name, param)]


@pytest.mark.parametrize(("stock", "hf"), _STOCK, ids=[stock for stock, _ in _STOCK])
def test_a_stock_named_parameter_exports_to_the_hf_name_serving_loads(ds4_export_hook, stock, hf):
    param = torch.randn(16, 8)
    assert dsv4_export.export_dsv4(_ARGS, "deepseekv4", _PREFIX + stock, param, None, None) is None
    (name, tensor), *rest = _export(stock, param)
    assert not rest
    assert name == f"model.layers.3.{hf}"
    assert tensor is param


def test_the_compressor_pair_exports_as_its_checkpoint_halves(ds4_export_hook):
    param = torch.randn(16, 8)
    (wkv_name, wkv), (wgate_name, wgate) = _export("core_attention.compressor.wkv_gate.weight", param)
    assert (wkv_name, wgate_name) == (
        "model.layers.3.self_attn.compressor.wkv.weight",
        "model.layers.3.self_attn.compressor.wgate.weight",
    )
    assert torch.equal(wkv, param[:8]) and torch.equal(wgate, param[8:])


@pytest.mark.parametrize("ratio", [4, 128])
def test_the_compressor_ape_exports_in_checkpoint_layout_without_recursing(ds4_export_hook, ratio):
    """The APE keeps its stock name in and out of the hook, so the hook must hand it to the stock converter."""
    param = torch.randn(ratio, 32)
    ((name, tensor),) = _export("core_attention.compressor.ape", param)
    assert name == "model.layers.3.self_attn.compressor.ape"
    assert torch.equal(tensor, dsv4_export.checkpoint_ape(param))


def test_wqkv_a_splits_into_wq_a_and_wkv(ds4_export_hook):
    param = torch.randn(_ARGS.backend.q_lora_rank + 4, 8)
    out = _export("wqkv_a_linear.weight", param)
    assert [name for name, _ in out] == ["model.layers.3.self_attn.wq_a.weight", "model.layers.3.self_attn.wkv.weight"]
    assert torch.equal(torch.cat([t for _, t in out]), param)


def test_without_the_hook_the_stock_converter_gives_the_same_tensors(ds4_export_hook):
    param = torch.randn(16, 8)
    for stock, _ in _STOCK:
        hooked = _export(stock, param)
        stock_only = convert_to_hf_stock(_ARGS, "deepseekv4", _PREFIX + stock, param)
        assert [name for name, _ in hooked] == [name for name, _ in stock_only]
        assert all(torch.equal(a, b) for (_, a), (_, b) in zip(hooked, stock_only, strict=True))


def test_the_vjp_adapter_maps_the_stock_names_to_the_reference_block():
    reference = pytest.importorskip("miles.utils.debug_utils.top_reference_dsv4")
    grads = {f"decoder.layers.3.self_attention.{stock}": torch.zeros(1) for stock, _ in _STOCK}
    out = reference.to_reference_param_grads(
        grads, SimpleNamespace(q_lora_rank=8, vocab_size=_ARGS.backend.vocab_size)
    )
    assert set(out) == {
        "model.layers.3.attn.wq_b.weight",
        "model.layers.3.attn.wo_b.weight",
        "model.layers.3.attn.q_norm.weight",
        "model.layers.3.attn.kv_norm.weight",
        "model.layers.3.attn.attn_sink",
        "model.layers.3.attn.compressor.norm.weight",
    }
