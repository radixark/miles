import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch


@pytest.fixture
def quantizer(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    package_paths = {
        "miles": root / "miles",
        "miles.utils": root / "miles" / "utils",
        "miles.backends": root / "miles" / "backends",
        "miles.backends.megatron_utils": root / "miles" / "backends" / "megatron_utils",
        "miles.backends.megatron_utils.megatron_to_hf": (
            root / "miles" / "backends" / "megatron_utils" / "megatron_to_hf"
        ),
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    fp8_kernel = types.ModuleType("miles.utils.fp8_kernel")
    fp8_kernel.blockwise_cast_to_fp8_triton = lambda *args, **kwargs: pytest.fail(
        "tests must stub _quantize_param before invoking the FP8 kernel"
    )
    monkeypatch.setitem(sys.modules, fp8_kernel.__name__, fp8_kernel)

    sglang_module_name = "miles.backends.megatron_utils.sglang"
    sglang_module = types.ModuleType(sglang_module_name)
    sglang_module.per_block_cast_to_fp8 = None
    sglang_module.quant_weight_ue8m0 = None
    sglang_module.should_deepgemm_weight_requant_ue8m0 = None
    sglang_module.transform_scale_ue8m0 = None
    monkeypatch.setitem(sys.modules, sglang_module_name, sglang_module)

    package_name = "miles.backends.megatron_utils.megatron_to_hf.processors"
    module_name = "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8"
    package_path = root / "miles" / "backends" / "megatron_utils" / "megatron_to_hf" / "processors"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_path = package_path / "quantizer_fp8.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    quantizer_fp8 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, quantizer_fp8)
    assert spec.loader is not None
    spec.loader.exec_module(quantizer_fp8)

    def passthrough(_args, _name, converted_named_params, _config):
        return converted_named_params

    processor_stubs = {
        "padding_remover": ("remove_padding", lambda _name, param, _vocab_size: param),
        "quantizer_compressed_tensors": ("quantize_params_compressed_tensors", passthrough),
        "quantizer_mxfp8": ("quantize_params_mxfp8", passthrough),
        "quantizer_nvfp4": ("quantize_params_nvfp4", passthrough),
    }
    for child_name, (function_name, function) in processor_stubs.items():
        child = types.ModuleType(f"{package_name}.{child_name}")
        setattr(child, function_name, function)
        monkeypatch.setitem(sys.modules, child.__name__, child)

    package_spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    processors = importlib.util.module_from_spec(package_spec)
    monkeypatch.setitem(sys.modules, package_name, processors)
    assert package_spec.loader is not None
    package_spec.loader.exec_module(processors)

    yield quantizer_fp8, processors.quantize_params


def _config(modules_to_not_convert):
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": modules_to_not_convert,
    }


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.decoder.layers.3.self_attention.wq_b.weight",
            "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
        ),
        (
            "module.module.decoder.layers.3.self_attention.linear_kv_up_proj.weight",
            "model.language_model.layers.3.self_attn.kv_b_proj.weight",
        ),
    ],
)
def test_fp8_quantizer_preserves_excluded_glm53_dsa_weights(quantizer, monkeypatch, megatron_name, hf_name):
    quantizer, _ = quantizer
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        quantizer,
        "_quantize_param",
        lambda *args, **kwargs: pytest.fail("excluded weight must not be quantized"),
    )

    output = quantizer.quantize_params_fp8(
        Namespace(indexer_rope_interleave=True),
        megatron_name,
        [(hf_name, weight)],
        _config([hf_name.replace("model.language_model.", "model.").removesuffix(".weight")]),
    )

    assert len(output) == 1
    assert output[0][0] == hf_name
    assert output[0][1] is weight
    assert not any(name.endswith("weight_scale_inv") for name, _ in output)


def test_fp8_quantizer_still_quantizes_non_excluded_weight(quantizer, monkeypatch):
    quantizer, _ = quantizer
    megatron_name = "module.module.decoder.layers.3.self_attention.linear_q_up_proj.weight"
    hf_name = "model.language_model.layers.3.self_attn.q_b_proj.weight"
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    quantized = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    monkeypatch.setattr(
        quantizer,
        "_quantize_param",
        lambda args, name, value, weight_block_size: [
            (name, quantized),
            (name.replace(".weight", ".weight_scale_inv"), scale),
        ],
    )

    output = quantizer.quantize_params_fp8(
        Namespace(indexer_rope_interleave=True),
        megatron_name,
        [(hf_name, weight)],
        _config(["model.layers.3.self_attn.indexer"]),
    )

    assert output == [
        (hf_name, quantized),
        ("model.language_model.layers.3.self_attn.q_b_proj.weight_scale_inv", scale),
    ]


@pytest.mark.parametrize(
    "rule",
    [
        "model.layers.3.self_attn.indexer.wq_b.weight",
        "layers.3.self_attn.indexer.wq_b",
    ],
)
def test_fp8_exclusion_accepts_weight_suffix_and_missing_model_prefix(quantizer, monkeypatch, rule):
    quantizer, _ = quantizer
    megatron_name = "module.module.decoder.layers.3.self_attention.wq_b.weight"
    hf_name = "model.language_model.layers.3.self_attn.indexer.wq_b.weight"
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        quantizer,
        "_quantize_param",
        lambda *args, **kwargs: pytest.fail("excluded weight must not be quantized"),
    )

    output = quantizer.quantize_params_fp8(
        Namespace(indexer_rope_interleave=True),
        megatron_name,
        [(hf_name, weight)],
        _config([rule]),
    )

    assert len(output) == 1
    assert output[0][0] == hf_name
    assert output[0][1] is weight


def test_fp8_exclusion_respects_dotted_module_boundaries(quantizer, monkeypatch):
    quantizer, _ = quantizer
    megatron_name = "module.module.decoder.layers.3.mlp.linear_fc1.weight"
    hf_name = "model.language_model.layers.3.mlp.gate_up_proj.weight"
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    quantized = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1)
    monkeypatch.setattr(
        quantizer,
        "_quantize_param",
        lambda args, name, value, weight_block_size: [
            (name, quantized),
            (name.replace(".weight", ".weight_scale_inv"), scale),
        ],
    )

    output = quantizer.quantize_params_fp8(
        Namespace(indexer_rope_interleave=True),
        megatron_name,
        [(hf_name, weight)],
        _config(["model.layers.3.mlp.gate"]),
    )

    assert output == [
        (hf_name, quantized),
        ("model.language_model.layers.3.mlp.gate_up_proj.weight_scale_inv", scale),
    ]


def test_none_quantization_config_passes_through(quantizer):
    _, quantize_params = quantizer
    converted_named_params = [("model.layers.0.self_attn.q_proj.weight", object())]

    output = quantize_params(None, "unused", converted_named_params, None)

    assert output is converted_named_params
