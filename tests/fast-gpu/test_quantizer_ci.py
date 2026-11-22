"""`quantize_params_compressed_tensors` re-quantizes exactly the weights the checkpoint stores packed."""

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

# The quantizer hardcodes `device="cuda"` throughout, so it runs on a GPU worker.
register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

register_rocm_ci(est_time=40, suite="stage-c-2-gpu-mi350", labels=["precision"], nightly=True)

import importlib.util
from argparse import Namespace

import torch

import miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 as qfp8
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    quantize_params_compressed_tensors,
)
from miles.utils.mxfp4 import dequantize_mxfp4, quantize_mxfp4

CONFIG = {
    "format": "int-quantized",
    "config_groups": {"group_0": {"weights": {"group_size": 128, "symmetric": True}}},
}


def test_only_the_packed_basenames_are_quantized():
    """A 2-D BF16 weight the checkpoint keeps unpacked (router, residual projection) must pass through."""
    params = [
        ("model.layers.0.experts.0.w1.weight", torch.randn(256, 256, device="cuda")),
        ("model.layers.0.gate.weight", torch.randn(256, 256, device="cuda")),
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(256, 256, device="cuda")),
    ]
    result_names = [
        name for name, _ in quantize_params_compressed_tensors(params, CONFIG, {"model.layers.0.experts.0.w1"})
    ]

    assert "model.layers.0.experts.0.w1.weight_packed" in result_names
    assert "model.layers.0.experts.0.w1.weight" not in result_names
    assert "model.layers.0.gate.weight" in result_names
    assert "model.layers.0.self_attn.q_proj.weight" in result_names


def test_mxfp4_quantize_inverts_the_checkpoint_dequantizer():
    """``quantize_mxfp4`` must be the exact inverse of the dequantizer used to
    read the K3 checkpoint: any value that came out of an MXFP4 checkpoint has
    to re-encode to the same bits, or every weight sync degrades the rollout
    weights a little further. The magnitude thresholds, sign-bit position,
    nibble order and exponent bias all have to agree.

    Also pins the mxfp4 branch of ``quantize_params_compressed_tensors``, which
    emits only weight_packed/weight_scale -- the int-quantized branch's extra
    weight_shape/weight_zero_point tensors would be rejected by the receiver.
    """
    group_size = 32
    packed = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
    scale = torch.randint(96, 144, (16, 4), dtype=torch.uint8, device="cuda")
    weight = dequantize_mxfp4(packed, scale, group_size)

    actual_packed, actual_scale = quantize_mxfp4(weight, group_size)
    actual = dequantize_mxfp4(actual_packed, actual_scale, group_size)

    torch.testing.assert_close(actual, weight, rtol=0, atol=0)

    config = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "group_size": group_size,
                    "symmetric": True,
                    "type": "float",
                    "num_bits": 4,
                    "scale_dtype": "torch.uint8",
                }
            }
        },
    }
    results = quantize_params_compressed_tensors(
        [("model.layers.0.experts.0.w1.weight", torch.randn(64, 64, device="cuda"))],
        config,
        {"model.layers.0.experts.0.w1"},
    )

    assert [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in results] == [
        ("model.layers.0.experts.0.w1.weight_packed", torch.uint8, (64, 32)),
        ("model.layers.0.experts.0.w1.weight_scale", torch.uint8, (64, 2)),
    ]


def test_fp8_batched_expert_quantization_matches_scalar(monkeypatch):
    """``quantize_expert_params_fp8`` must match ``quantize_params_fp8`` bit-for-bit and stride-for-stride."""
    device = torch.device("cuda", torch.cuda.current_device())
    kw = {"dtype": torch.bfloat16, "device": device}
    torch.manual_seed(42)
    fc1_e2 = torch.linspace(-3.0, 3.0, 130 * 256, **kw).reshape(130, 256)
    fc1_e2[0, 0] = 1e4

    def _fc1(idx, w, with_scale=False):
        g, u = w.chunk(2, dim=0)
        p = f"model.layers.0.mlp.experts.{idx}"
        extra = [(f"{p}.gate_proj.weight_scale", torch.ones(1, device=device))] if with_scale else []
        return f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{idx}", [
            (f"{p}.gate_proj.weight", g),
            *extra,
            (f"{p}.up_proj.weight", u),
        ]

    def _fc2(layer, idx, w, mtp=False):
        mid = "transformer_layer." if mtp else ""
        kind = "mtp" if mtp else "decoder"
        return f"module.module.{kind}.layers.{layer}.{mid}mlp.experts.linear_fc2.weight{idx}", [
            (f"model.layers.{layer + int(mtp)}.mlp.experts.{idx}.down_proj.weight", w)
        ]

    # Unaligned rows=65 (packed_sf_k=1), cols=640 (packed_sf_k=2), strided, .T, zeros, outliers, _scale, exclusions.
    units = [
        _fc1(0, torch.randn(130, 256, **kw), with_scale=True),
        _fc2(0, 0, torch.randn(384, 640, **kw)[::2]),
        _fc1(1, torch.zeros(130, 256, **kw)),
        _fc2(0, 1, torch.randn(192, 640, **kw)),
        _fc1(2, fc1_e2),
        _fc2(0, 2, torch.randn(256, 128, **kw).T),
        _fc2(0, 3, torch.randn(256, 128, **kw).T),
        _fc2(2, 0, torch.randn(192, 640, **kw)),
        _fc2(0, 0, torch.randn(128, 256, **kw), mtp=True),
    ]

    base = {"quant_method": "fp8", "activation_scheme": "dynamic", "ignored_layers": ["model.layers.2.mlp.experts"]}
    block = {**base, "weight_block_size": [128, 128]}
    cases = [
        ("per_tensor", base, "1", None),
        ("blockwise_128", block, "1", "fp8_blockwise_cast"),
        ("blockwise_64_128", {**base, "weight_block_size": [64, 128]}, "1", "fp8_blockwise_cast"),
    ]
    if qfp8.per_block_cast_to_fp8 is not None:
        cases.append(("per_block", block, "0", "per_block_cast_to_fp8"))
    if (
        qfp8.quant_weight_ue8m0 is not None
        and importlib.util.find_spec("deep_gemm") is not None
        and torch.cuda.get_device_capability(device)[0] in (9, 10)
    ):
        cases.append(("ue8m0", block, "1", "quant_weight_ue8m0"))

    args = Namespace()
    for mode_label, cfg, fp32_env, spy_attr in cases:
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", fp32_env)
        monkeypatch.setattr(qfp8, "_get_scale_format", lambda *_a, _m=mode_label: "ue8m0" if _m == "ue8m0" else None)
        expected = [qfp8.quantize_params_fp8(args, m_name, params, cfg) for m_name, params in units]
        spy_calls = []
        if spy_attr is not None:
            orig = getattr(qfp8, spy_attr)
            monkeypatch.setattr(
                qfp8,
                spy_attr,
                lambda w, *a, _o=orig, _c=spy_calls, **kw: (_c.append(tuple(w.shape)), _o(w, *a, **kw))[1],
            )

        actual = qfp8.quantize_expert_params_fp8(args, units, cfg)
        for a_unit, e_unit in zip(actual, expected, strict=True):
            assert [n for n, _ in a_unit] == [n for n, _ in e_unit]
            for (_, a_t), (_, e_t) in zip(a_unit, e_unit, strict=True):
                assert a_t.device == e_t.device and a_t.dtype == e_t.dtype
                assert a_t.shape == e_t.shape and a_t.stride() == e_t.stride()
                assert torch.equal(
                    a_t.contiguous().reshape(-1).view(torch.uint8), e_t.contiguous().reshape(-1).view(torch.uint8)
                )
        if spy_attr is not None:
            assert len(spy_calls) == 5, f"{mode_label}: expected 5 kernel calls, got {spy_calls}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
