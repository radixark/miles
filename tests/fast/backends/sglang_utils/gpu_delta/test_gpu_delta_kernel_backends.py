"""Kernel-backend registry: exact SGLang runner values, their dispatchers and no generic fallback."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

from miles.backends.sglang_utils.gpu_delta import bindings, kernel_backends
from miles.backends.sglang_utils.gpu_delta.kernel_backends import admits_moe_dispatcher
from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer as fi
from miles.backends.sglang_utils.gpu_delta.kernel_backends import identify_expert_layout
from miles.backends.sglang_utils.gpu_delta.kernel_backends.base import ExpertLayout


def _layer(method_name, **attrs):
    method = type(method_name, (), {})()
    method.runner = None
    method.load_up_proj_weight_first = True
    method.__dict__.update(attrs)
    return SimpleNamespace(quant_method=method, moe_runner_config=SimpleNamespace(is_gated=True))


def _bf16():
    return _layer("UnquantizedFusedMoEMethod", use_flashinfer_trtllm_moe=True)


def _mxfp4():
    layer = _layer("Mxfp4FlashinferTrtllmMoEMethod", _fp8=SimpleNamespace(dequant_fp4_to_fp8=False))
    # The companion tree keeps the loader's FP32 scales beside its kernel images.
    layer.w13_weight_scale_inv = torch.ones((1, 2, 2))
    return layer


def _nvfp4():
    layer = _layer(
        "ModelOptNvFp4FusedMoEMethod",
        _is_cutedsl_v2_standard=True,
        quant_config=SimpleNamespace(is_checkpoint_nvfp4_serialized=True),
    )
    layer._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
    config = type("Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig", (), {})()
    layer._flashinfer_megamoe_layer = SimpleNamespace(_megakernel_config=config)
    return layer


# One admitted layer per registered runner: moe_runner_backend -> (its layer, the identified format).
_RUNNERS = {
    "flashinfer_trtllm": (_bf16, "bf16"),
    "flashinfer_trtllm_routed": (_bf16, "bf16"),
    "flashinfer_mxfp4": (_mxfp4, "mxfp4"),
    "flashinfer_cutedsl": (_nvfp4, "nvfp4"),
    "flashinfer_megamoe": (_nvfp4, "nvfp4"),
}
# The (moe_runner_backend, moe_a2a_backend) launch pairs fixed-topology admission accepts.
_LAUNCH_PAIRS = {
    ("flashinfer_trtllm", "none"),
    ("flashinfer_trtllm_routed", "none"),
    ("flashinfer_mxfp4", "none"),
    ("flashinfer_cutedsl", "none"),
    ("flashinfer_megamoe", "flashinfer_megamoe"),
}
# Every runner and dispatcher value the installed SGLang defines, plus spellings no exact lookup may
# accept: the megamoe runner alias, the former short labels and both FlashInfer dispatcher names.
_RUNNER_VALUES = sorted({backend.value for backend in MoeRunnerBackend} | {"megamoe", "trtllm", "cutedsl"})
_A2A_VALUES = sorted({backend.value for backend in MoeA2ABackend} | {"flashinfer", "flashinfer_nvlink_one_sided"})
# Every attribute and method generic code reads from an identified layer.
_INTERFACE = sorted(ExpertLayout.__annotations__) + sorted(
    name for name, value in vars(ExpertLayout).items() if callable(value) and not name.startswith("_")
)


def test_registry_lists_exactly_the_admitted_runners():
    assert set(kernel_backends._MOE_RUNNER_BACKENDS) == set(_RUNNERS)
    with pytest.raises(TypeError):
        kernel_backends._MOE_RUNNER_BACKENDS["triton"] = kernel_backends._MOE_RUNNER_BACKENDS["flashinfer_trtllm"]


@pytest.mark.parametrize("runner", sorted(_RUNNERS))
def test_registered_runner_identifies_through_its_backend_module(runner):
    layer, fmt = _RUNNERS[runner]
    layout = identify_expert_layout(layer(), runner)
    assert layout == fi.identify_layout(layer(), kernel_backends._MOE_RUNNER_BACKENDS[runner].storage)
    assert layout.format == fmt
    # The backend module takes the row's storage family, never the runner value itself.
    with pytest.raises(ValueError, match=f"^unsupported FlashInfer expert backend: {runner}$"):
        fi.identify_layout(layer(), runner)
    assert [name for name in _INTERFACE if not hasattr(layout, name)] == []


# "auto" is unresolved, "megamoe" is the deprecated runner alias of SGLang's DeepGEMM MegaMoE
# dispatcher, "experimental_sgl_trtllm" is a LoRA runner sharing TRTLLM weight preparation and the
# short labels are receiver-internal: none may match by alias or prefix.
@pytest.mark.parametrize(
    "runner",
    ["auto", "megamoe", "experimental_sgl_trtllm", "trtllm", "cutedsl", "flashinfer_cutlass", "triton", "deep_gemm"],
)
def test_unlisted_runner_is_rejected_with_the_supported_list(runner):
    with pytest.raises(ValueError, match=f"unsupported GPU delta MoE runner backend: {runner}; supported: "):
        identify_expert_layout(_bf16(), runner)


def test_dispatchers_are_exactly_the_admitted_launch_pairs():
    for runner in _RUNNER_VALUES:
        for a2a in _A2A_VALUES:
            assert admits_moe_dispatcher(runner, a2a) == ((runner, a2a) in _LAUNCH_PAIRS), (runner, a2a)


def test_layer_runner_and_launch_runner_resolve_to_the_exact_value():
    layer = _nvfp4()
    layer.quant_method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL)
    assert bindings._identify_moe_layout(layer) == fi.identify_layout(layer, "cutedsl")
    # A direct TRTLLM wrapper owns no runner; identification reads the launch value instead.
    layer = _bf16()
    launch = SimpleNamespace(get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(moe_runner_backend="megamoe")))
    with patch.dict(sys.modules, {"sglang.srt.runtime_context": launch}):
        with pytest.raises(ValueError, match="unsupported GPU delta MoE runner backend: megamoe"):
            bindings._identify_moe_layout(layer)


def test_backend_modules_import_only_shared_leaves():
    # bindings imports this package, so a backend module importing bindings, layout, models or receiver
    # would form a cycle; SGLang and kernel libraries stay lazy, so importing the registry loads neither.
    package = Path(kernel_backends.__file__).parent
    shared = {f"{bindings.__package__}.byte_views", f"{bindings.__package__}.derived"}
    for path in sorted(package.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, (path.name, "relative import")
                modules = [node.module]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] not in {"sglang", "flashinfer", "deep_gemm", "triton"}, (path.name, module)
                if module.startswith("miles."):
                    assert module in shared or module.startswith(kernel_backends.__name__), (path.name, module)
