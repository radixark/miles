import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[4]
_INDEXER_WEIGHTS = [
    ("wq_b.weight", "linear_wq_b.weight", (256, 8)),
    ("wk.weight", "linear_wk.weight", (128, 8)),
    ("weights_proj.weight", "linear_weights_proj.weight", (2, 8)),
    ("k_norm.weight", "k_norm.weight", (128,)),
    ("k_norm.bias", "k_norm.bias", (128,)),
]


def _load_module(relative_path, name):
    # Load only the converter under test, without unrelated GPU model plugins.
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raw_converter():
    return _load_module(
        "miles/backends/megatron_utils/megatron_to_hf/deepseekv3.py", "dsa_raw_export_under_test"
    ).convert_deepseekv3_to_hf


@pytest.fixture(scope="module")
def bridge_module():
    pytest.importorskip("mbridge")
    return _load_module("miles_plugins/mbridge/deepseek_v32.py", "dsa_mbridge_under_test")


def _weight(shape):
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def _expected_hf_weight(weight, hf_suffix, impl, interleave):
    if impl == "megatron" or not interleave or hf_suffix == "weights_proj.weight":
        return weight
    # The legacy Miles layout stores the 64 RoPE channels after the 64
    # non-RoPE channels, independently for every indexer query head.
    rows = torch.arange(weight.shape[0])
    hf_rows = rows // 128 * 128 + (rows % 128 + 64) % 128
    return weight[hf_rows]


def _mcore_name(hf_suffix, native_suffix, impl):
    suffix = f"core_attention.indexer.{native_suffix}" if impl == "megatron" else hf_suffix
    return f"decoder.layers.3.self_attention.{suffix}"


@pytest.mark.parametrize("impl", ["miles", "megatron"])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("hf_suffix,native_suffix,shape", _INDEXER_WEIGHTS)
def test_raw_indexer_export_preserves_each_implementation_layout(
    raw_converter, impl, interleave, hf_suffix, native_suffix, shape
):
    args = SimpleNamespace(
        hidden_size=8, num_attention_heads=2, num_query_groups=1, indexer_rope_interleave=interleave
    )
    weight = _weight(shape)
    name = "module.module." + _mcore_name(hf_suffix, native_suffix, impl)
    [(hf_name, exported)] = raw_converter(args, name, weight)
    assert hf_name == f"model.layers.3.self_attn.indexer.{hf_suffix}"
    torch.testing.assert_close(exported, _expected_hf_weight(weight, hf_suffix, impl, interleave), rtol=0, atol=0)


@pytest.mark.parametrize(("bridge_class", "interleave"), [("DeepseekV32Bridge", False), ("GlmMoeDsaBridge", True)])
@pytest.mark.parametrize("impl", ["miles", "megatron"])
@pytest.mark.parametrize("hf_suffix,native_suffix,shape", _INDEXER_WEIGHTS)
def test_mbridge_indexer_import_export_matches_raw_layout(
    bridge_module, bridge_class, impl, interleave, hf_suffix, native_suffix, shape
):
    bridge = object.__new__(getattr(bridge_module, bridge_class))
    bridge.hf_config = SimpleNamespace(indexer_rope_interleave=interleave)
    bridge.config = SimpleNamespace(mtp_num_layers=None)
    bridge.make_vocab_size_divisible_by = None
    weight = _weight(shape)
    name = _mcore_name(hf_suffix, native_suffix, impl)
    hf_names, [exported] = bridge._weight_to_hf_format(name, weight)
    assert hf_names == [f"model.layers.3.self_attn.indexer.{hf_suffix}"]
    torch.testing.assert_close(exported, _expected_hf_weight(weight, hf_suffix, impl, interleave), rtol=0, atol=0)
    imported = bridge._weight_to_mcore_format(name, [exported])
    torch.testing.assert_close(imported, weight, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["q", "kv"])
@pytest.mark.parametrize("layout", ["fused", "unfused"])
def test_mla_norm_raw_export_keeps_hf_order(raw_converter, norm, layout):
    suffix = f"linear_{norm}_up_proj.layer_norm_weight" if layout == "fused" else f"{norm}_layernorm.weight"
    name = f"decoder.layers.3.self_attention.{suffix}"
    weight = _weight((128,))
    args = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_query_groups=1)
    [(hf_name, exported)] = raw_converter(args, "module.module." + name, weight)
    assert hf_name == f"model.layers.3.self_attn.{norm}_a_layernorm.weight"
    torch.testing.assert_close(exported, weight, rtol=0, atol=0)


@pytest.mark.parametrize("norm", ["q", "kv"])
@pytest.mark.parametrize("layout", ["fused", "unfused"])
def test_mla_norm_mbridge_round_trip_keeps_hf_order(bridge_module, norm, layout):
    bridge = object.__new__(bridge_module.DeepseekV32Bridge)
    bridge.hf_config = SimpleNamespace(indexer_rope_interleave=True)
    bridge.config = SimpleNamespace(mtp_num_layers=None)
    bridge.make_vocab_size_divisible_by = None
    suffix = f"linear_{norm}_up_proj.layer_norm_weight" if layout == "fused" else f"{norm}_layernorm.weight"
    name = f"decoder.layers.3.self_attention.{suffix}"
    weight = _weight((128,))
    names, [exported] = bridge._weight_to_hf_format(name, weight)
    assert names == [f"model.layers.3.self_attn.{norm}_a_layernorm.weight"]
    torch.testing.assert_close(exported, weight, rtol=0, atol=0)
    torch.testing.assert_close(bridge._weight_to_mcore_format(name, [exported]), weight, rtol=0, atol=0)
