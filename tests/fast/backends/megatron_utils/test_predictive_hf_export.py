import sys
import types
from argparse import Namespace

import torch


def _install_export_test_stubs():
    if "sglang.srt.layers.quantization.fp8_utils" not in sys.modules:
        fp8_utils_module = types.ModuleType("sglang.srt.layers.quantization.fp8_utils")
        fp8_utils_module.quant_weight_ue8m0 = None
        fp8_utils_module.transform_scale_ue8m0 = None
        fp8_utils_module.mxfp8_group_quantize = None

        quantization_module = types.ModuleType("sglang.srt.layers.quantization")
        quantization_module.fp8_utils = fp8_utils_module
        layers_module = types.ModuleType("sglang.srt.layers")
        layers_module.quantization = quantization_module

        model_loader_utils_module = types.ModuleType("sglang.srt.model_loader.utils")
        model_loader_utils_module.should_deepgemm_weight_requant_ue8m0 = None
        model_loader_module = types.ModuleType("sglang.srt.model_loader")
        model_loader_module.utils = model_loader_utils_module

        patch_torch_module = types.ModuleType("sglang.srt.utils.patch_torch")
        patch_torch_module.monkey_patch_torch_reductions = lambda: None
        utils_module = types.ModuleType("sglang.srt.utils")
        utils_module.patch_torch = patch_torch_module
        utils_module.MultiprocessingSerializer = object

        tensor_bucket_module = types.ModuleType("sglang.srt.weight_sync.tensor_bucket")
        tensor_bucket_module.FlattenedTensorBucket = object
        weight_sync_module = types.ModuleType("sglang.srt.weight_sync")
        weight_sync_module.tensor_bucket = tensor_bucket_module

        model_runner_module = types.ModuleType("sglang.srt.model_executor.model_runner")
        model_runner_module.FlattenedTensorBucket = object
        model_executor_module = types.ModuleType("sglang.srt.model_executor")
        model_executor_module.model_runner = model_runner_module

        legacy_patch_torch_module = types.ModuleType("sglang.srt.patch_torch")
        legacy_patch_torch_module.monkey_patch_torch_reductions = lambda: None

        srt_module = types.ModuleType("sglang.srt")
        srt_module.layers = layers_module
        srt_module.model_loader = model_loader_module
        srt_module.utils = utils_module
        srt_module.weight_sync = weight_sync_module
        srt_module.model_executor = model_executor_module
        srt_module.patch_torch = legacy_patch_torch_module

        sglang_module = types.ModuleType("sglang")
        sglang_module.srt = srt_module

        sys.modules["sglang"] = sglang_module
        sys.modules["sglang.srt"] = srt_module
        sys.modules["sglang.srt.layers"] = layers_module
        sys.modules["sglang.srt.layers.quantization"] = quantization_module
        sys.modules["sglang.srt.layers.quantization.fp8_utils"] = fp8_utils_module
        sys.modules["sglang.srt.model_loader"] = model_loader_module
        sys.modules["sglang.srt.model_loader.utils"] = model_loader_utils_module
        sys.modules["sglang.srt.utils"] = utils_module
        sys.modules["sglang.srt.utils.patch_torch"] = patch_torch_module
        sys.modules["sglang.srt.weight_sync"] = weight_sync_module
        sys.modules["sglang.srt.weight_sync.tensor_bucket"] = tensor_bucket_module
        sys.modules["sglang.srt.model_executor"] = model_executor_module
        sys.modules["sglang.srt.model_executor.model_runner"] = model_runner_module
        sys.modules["sglang.srt.patch_torch"] = legacy_patch_torch_module


def test_convert_to_hf_skips_predictive_router_parameters(monkeypatch):
    _install_export_test_stubs()
    import miles.backends.megatron_utils.megatron_to_hf as megatron_to_hf

    sentinel = torch.ones(1)

    def _unexpected(*args, **kwargs):
        raise AssertionError("predictive router params should be skipped before any HF conversion work")

    monkeypatch.setattr(megatron_to_hf, "remove_padding", _unexpected)
    monkeypatch.setattr(megatron_to_hf, "_convert_to_hf_core", _unexpected)
    monkeypatch.setattr(megatron_to_hf, "quantize_params", _unexpected)

    outputs = megatron_to_hf.convert_to_hf(
        Namespace(vocab_size=1),
        "qwen3moe",
        "module.module.decoder.layers.0.mlp.router.bias_predictor.weight",
        sentinel,
    )

    assert outputs == []


