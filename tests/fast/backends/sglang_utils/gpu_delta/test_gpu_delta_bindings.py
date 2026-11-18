"""CPU canonical bindings, physical byte layouts, and format admission contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe
from tests.fast.fixtures.gpu_delta_fixtures import (
    apply_segments,
    blocked_scales_padded,
    interleave,
    quantized_layer,
    shuffled_rows,
)

from miles.backends.sglang_utils.gpu_delta.bindings import (
    ConsumerSnapshot,
    TensorBinding,
    _identify_moe_layout,
    _moe_binding,
    prepare_raw_xor,
)
from miles.backends.sglang_utils.gpu_delta.byte_views import _DtypeMismatch
from miles.backends.sglang_utils.gpu_delta.kernel_backends import admits_moe_dispatcher
from miles.backends.sglang_utils.gpu_delta.kernel_backends import flashinfer as fi
from miles.backends.sglang_utils.gpu_delta.kernel_backends import identify_expert_layout
from miles.backends.sglang_utils.gpu_delta.kernel_backends.flashinfer import FlashInferLayout


def _block_scale_layer():
    return SimpleNamespace(
        moe_tp_size=1,
        _map_global_expert_id_to_local_expert_id=lambda expert: (0 if expert == 2 else -1),
        w2_weight_scale_inv=torch.full((1, 17, 15), -3.0),
    )


@pytest.mark.parametrize("checkpoint_dtype", ["F32", "U8"])
def test_block_scale_prior_is_identical_on_local_and_foreign_ranks(
    checkpoint_dtype,
):
    layout = FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    layer = _block_scale_layer()
    local, foreign = ({"dtype": checkpoint_dtype, "shape": [17, 15]} for _ in range(2))
    binding = _moe_binding("scale", local, layer, 2, "down", "weight_scale_inv", layout)
    assert _moe_binding("scale", foreign, layer, 3, "down", "weight_scale_inv", layout) is None
    expected = {
        "dtype": "U8",
        "shape": [17, 15],
        "format": "fp8-block128-ue8m0",
        "encoding": "xor_raw_bytes",
    }
    if checkpoint_dtype == "F32":
        expected["checkpoint_dtype"] = "F32"
    assert local == foreign == expected
    assert binding.dtype == "U8" and binding.shape == (17, 15)
    assert binding.encoding == "xor_raw_bytes" and binding.format == "fp8-block128-ue8m0"
    with pytest.raises(ValueError, match="exact UE8M0"):
        _moe_binding(
            "scale",
            {"dtype": "I32", "shape": [17, 15]},
            layer,
            2,
            "down",
            "weight_scale_inv",
            layout,
        )


@pytest.mark.parametrize("kind", ["ue8m0", "bf16-scalar", "bf16-vector", "scalar", "vector"])
def test_small_xor_is_reversible_without_live_preparation_writes(kind):
    if kind == "ue8m0":
        before = torch.arange(255, dtype=torch.uint8)
        after = before.flip(0)
        target = before.view(torch.float8_e8m0fnu).float().reshape(17, 15)
        binding = TensorBinding(
            "scale",
            "U8",
            (17, 15),
            [[0, 17], [0, 15]],
            (target,),
            encoding="xor_raw_bytes",
            format="fp8-block128-ue8m0",
        )
        expected = after.view(torch.float8_e8m0fnu).float().reshape(target.shape)
    else:
        before = torch.tensor([1.5, 2.0, 4.0], dtype=torch.bfloat16 if kind.startswith("bf16-") else torch.float32)
        if kind.endswith("scalar"):
            before = before[0]
        after = before * 2
        target = before.float().clone()
        binding = TensorBinding(
            kind,
            "BF16" if kind.startswith("bf16-") else "F32",
            tuple(before.shape),
            [[0, n] for n in before.shape],
            (target,),
            encoding="xor_raw_bytes",
        )
        expected = after.float()
        before, after = (value.reshape(-1).view(torch.uint8) for value in (before, after))
    original = target.clone()
    mask = before ^ after
    for wanted in (expected, original):
        current = target.clone()
        ((actual, physical_mask, predicted),) = prepare_raw_xor(
            [(binding, mask)], {id(target)} if kind.endswith("scalar") else ()
        )
        assert actual is target and torch.equal(target, current)
        if kind.endswith("vector"):
            assert predicted is None
            if kind == "vector":
                assert physical_mask.data_ptr() == mask.data_ptr()
        else:
            assert torch.equal(predicted, wanted)
        target.reshape(-1).view(torch.uint8).bitwise_xor_(physical_mask)
        assert torch.equal(target, wanted)
    if kind == "ue8m0":
        assert target.reshape(-1).view(torch.int32)[0].item() == 0x00400000


def test_cutedsl_bindings_preserve_scale_images_padding_and_aliases():
    # Independently derive physical indices, including row/column tails. This
    # catches accidental reshapes that clone, alias double-XOR, and padding writes.
    for projection, rows, cols in (
        ("gate", 64, 3),
        ("up", 192, 19),
        ("down", 17, 3),
        ("down", 129, 8),
        ("down", 191, 19),
        ("down", 256, 64),
    ):
        full_rows = rows if projection == "down" else 2 * rows
        pr, pc = (full_rows + 127) // 128 * 128, (cols + 3) // 4 * 4
        for independent in (False, True):
            primary = torch.randint(256, (pr, pc), dtype=torch.uint8)
            physical = primary.view(pr // 128, pc // 4, 32, 4, 4)
            mma = physical.permute(2, 3, 0, 4, 1).unsqueeze(-1)
            if independent:
                mma = mma.clone(memory_format=torch.preserve_format)
            stem = "w2" if projection == "down" else "w13"
            layer = SimpleNamespace(
                moe_tp_size=1,
                **{
                    f"{stem}_blockscale_swizzled": primary.unsqueeze(0),
                    f"{stem}_blockscale_mma": mma,
                    f"{stem}_weight_scale": primary.unsqueeze(0),
                    f"{stem}_weight": torch.zeros((1, full_rows, cols * 8), dtype=torch.uint8),
                },
            )
            layout = fi.FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", True)
            layer._map_global_expert_id_to_local_expert_id = lambda expert: expert
            binding = _moe_binding(
                "scale",
                {"dtype": "F8_E4M3", "shape": [rows, cols]},
                layer,
                0,
                projection,
                "weight_scale",
                layout,
            )
            segments = binding.segments
            assert len(binding.storage) == 1 + independent
            images = [primary]
            if independent:
                images.append(mma[..., 0].permute(2, 4, 0, 1, 3).view_as(primary))
            before = [image.clone() for image in images]
            pointers = [image.data_ptr() for image in images]
            mask = torch.randint(256, (rows, cols), dtype=torch.uint8)
            row = torch.arange(rows)[:, None]
            if projection != "down":
                row = row // 64 * 128 + row % 64
                if projection == "gate":
                    row += 64
            col = torch.arange(cols)[None, :]
            offset = (row // 128 * (pc // 4) + col // 4) * 512 + row % 32 * 16 + row % 128 // 32 * 4 + col % 4
            apply_segments(segments, mask)
            for image, original in zip(images, before, strict=False):
                expected = original.flatten().clone()
                expected[offset.flatten()] ^= mask.flatten()
                assert torch.equal(image.flatten(), expected)
            apply_segments(segments, mask)
            for image, original, pointer in zip(images, before, pointers, strict=False):
                assert torch.equal(image, original)
                assert image.data_ptr() == pointer


# FlashInfer's own gated-row interleave, copied verbatim from flashinfer/fused_moe/prepare.py (FlashInfer b88169f8).
# interleave_cute_dsl_moe_weights(weight, block_scales, swap_ab) applies it along dim 1 to the [up; gate] W13 weights
# and their row-major block scales: 16-row groups under swap-AB (SM100 and SM103), 64-row groups otherwise.
def _interleave_linear_and_gate(x: torch.Tensor, group_size: int = 64, dim: int = -1) -> torch.Tensor:
    """Interleave the linear and gate halves of a SwiGLU gemm1 weight."""
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


@pytest.mark.parametrize(
    "swap_ab,tp_size",
    [
        # The companion tree's FlashInfer 0.7.2 wrapper records no swap-AB; interleave_w13_halves takes 64-row groups.
        (None, 1),
        # SGLang main without swap-AB (off SM100/SM103), on MoE TP rank 1 of 2: CuTe DSL splits the canonical rows
        # without padding, so the rank's layer holds the interleave of its own row slice.
        (False, 2),
    ],
    ids=["companion-tp1", "main-tp2"],
)
def test_cutedsl_w13_bindings_follow_flashinfers_gated_interleave(swap_ab, tp_size):
    # SGLang main's standard CuTe DSL loader stores [up; gate] W13 halves and interleaves the weights and row-major
    # block scales with interleave_cute_dsl_moe_weights under the swap-AB decision its CuteDslMoEWrapper records;
    # without swap-AB that interleave takes 64-row groups, as the companion tree's interleave_w13_halves does. It then
    # swizzles the scales into the image w13_weight_scale aliases and serves that image's MMA layout as
    # w13_blockscale_mma, a strided view in SGLang main; here the MMA layout has storage of its own, so each image
    # must receive the XOR.
    group = 64
    # Two local experts, each canonical half 384 rows: three 64-row groups per rank at MoE TP2; 7 scale columns pad
    # to 8.
    experts, rows, cols = (3, 7), 384, 112
    tp_rank = tp_size - 1
    owned = [rows // tp_size * tp_rank, rows // tp_size * (tp_rank + 1)]
    local_rows = owned[1] - owned[0]
    canonical = {
        (expert, projection, suffix): torch.randint(256, (rows, cols // divisor), dtype=torch.uint8)
        for expert in experts
        for projection in ("gate", "up")
        for suffix, divisor in (("weight", 2), ("weight_scale", 16))
    }
    # Down projections are [hidden rows, intermediate columns]; MoE TP splits their columns without padding.
    canonical |= {
        (expert, "down", suffix): torch.randint(256, (cols, rows // divisor), dtype=torch.uint8)
        for expert in experts
        for suffix, divisor in (("weight", 2), ("weight_scale", 16))
    }

    def down(suffix):
        width = rows // (2 if suffix == "weight" else 16) // tp_size
        local = [canonical[expert, "down", suffix][:, width * tp_rank : width * (tp_rank + 1)] for expert in experts]
        if suffix == "weight":
            return torch.stack(local)
        return torch.stack([blocked_scales_padded(scale).view(128, -1) for scale in local])

    def interleaved(suffix):
        halves = [
            torch.cat(
                [canonical[expert, "up", suffix][slice(*owned)], canonical[expert, "gate", suffix][slice(*owned)]]
            )
            for expert in experts
        ]
        return _interleave_linear_and_gate(torch.stack(halves), group_size=group, dim=1)

    def swizzled():
        return torch.stack(
            [blocked_scales_padded(scale).view(2 * local_rows, -1) for scale in interleaved("weight_scale")]
        )

    layer = quantized_layer(
        "ModelOptNvFp4FusedMoEMethod",
        _is_cutedsl_v2_standard=True,
        load_up_proj_weight_first=True,
        quant_config=SimpleNamespace(is_checkpoint_nvfp4_serialized=True),
    )
    layer._cutedsl_wrapper.quant_mode = "w4a16"
    if swap_ab is not None:
        layer._cutedsl_wrapper.swap_ab = swap_ab
    layer.moe_tp_size, layer.moe_tp_rank, layer.use_padded_loading = tp_size, tp_rank, False
    layer._map_global_expert_id_to_local_expert_id = experts.index
    layer.w13_weight = interleaved("weight")
    layer.w13_blockscale_swizzled = layer.w13_weight_scale = swizzled().view(torch.float8_e4m3fn)
    # The MMA layout's own [expert, row tile, column tile, 32, 4, 4] storage, viewed with its public axes.
    tiles = swizzled().view(len(experts), 2 * local_rows // 128, -1, 32, 4, 4)
    layer.w13_blockscale_mma = tiles.permute(3, 4, 1, 5, 2, 0)
    layer.w2_weight = down("weight")
    layer.w2_blockscale_swizzled = layer.w2_weight_scale = down("weight_scale").view(torch.float8_e4m3fn)
    layout = identify_expert_layout(layer, "flashinfer_cutedsl")
    assert layout == FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", True)
    assert layout.pair.group == group
    for expert in experts:
        for projection in ("gate", "up", "down"):
            for suffix, dtype in (("weight", "U8"), ("weight_scale", "F8_E4M3")):
                # Full-projection masks: the rank's binding selects and applies only its own rows (columns of down).
                mask = torch.randint(256, canonical[expert, projection, suffix].shape, dtype=torch.uint8)
                name = f"model.layers.0.mlp.experts.{expert}.{projection}_proj.{suffix}"
                binding = _moe_binding(
                    name, {"dtype": dtype, "shape": list(mask.shape)}, layer, expert, projection, suffix, layout
                )
                if projection == "down":
                    width = mask.shape[1] // tp_size
                    assert binding.slices == [[0, mask.shape[0]], [width * tp_rank, width * (tp_rank + 1)]]
                else:
                    assert binding.slices == [owned, [0, mask.shape[1]]]
                for view in binding.apply_bindings():
                    source = view.selected_bytes(mask.flatten())
                    for target in view.destinations:
                        target.bitwise_xor_(source.reshape(target.shape))
                canonical[expert, projection, suffix] ^= mask
    # Every canonical gate/up row the rank owns and its block scales sit where FlashInfer's interleave puts them, in
    # both scale images, and the scale padding stays untouched.
    assert torch.equal(layer.w13_weight, interleaved("weight"))
    assert torch.equal(layer.w13_blockscale_swizzled.view(torch.uint8), swizzled())
    assert torch.equal(tiles, swizzled().view_as(tiles))
    assert torch.equal(layer.w2_weight, down("weight"))
    assert torch.equal(layer.w2_blockscale_swizzled.view(torch.uint8), down("weight_scale"))


@pytest.mark.parametrize(
    "backend,transposed,image,served_dtype",
    [
        ("cutedsl", False, "weight", torch.uint8),
        ("cutedsl", False, "weight", torch.int8),
        ("cutedsl", False, "weight", torch.float8_e5m2),
        ("cutedsl", False, "scale", torch.int8),
        ("cutedsl", False, "mma", torch.float8_e5m2),
        # SGLang's MegaMoE serves the packed weights as float4_e2m1fn_x2, through a transposed view.
        ("megamoe", True, "weight", torch.float4_e2m1fn_x2),
    ],
)
def test_packed_dtype_admission_preserves_original_storage_types(backend, transposed, image, served_dtype):
    weight = torch.zeros((1, 128, 32), dtype=torch.uint8)
    if transposed:
        weight = weight.transpose(1, 2)
    primary = torch.zeros((128, 4), dtype=torch.uint8)
    mma = primary.view(1, 1, 32, 4, 4).permute(2, 3, 0, 4, 1).unsqueeze(-1)
    mma = mma.clone(memory_format=torch.preserve_format)
    if image == "weight":
        weight = weight.view(served_dtype)
    elif image == "scale":
        primary = primary.view(served_dtype)
    else:
        mma = mma.view(served_dtype)
    layer = SimpleNamespace(
        moe_tp_size=1,
        _map_global_expert_id_to_local_expert_id=lambda expert: expert,
        w2_weight=weight,
        w2_weight_scale=primary.unsqueeze(0),
        w2_blockscale_swizzled=primary.unsqueeze(0),
        w2_blockscale_mma=mma,
    )
    suffix, canonical_dtype, shape = (
        ("weight", "U8", [128, 32]) if image == "weight" else ("weight_scale", "F8_E4M3", [128, 4])
    )
    tensors = (weight, primary, mma)
    before = [tensor.contiguous().view(torch.uint8).clone() for tensor in tensors]

    def bind():
        return _moe_binding(
            "projection",
            {"dtype": canonical_dtype, "shape": shape},
            layer,
            0,
            "down",
            suffix,
            FlashInferLayout("nvfp4", backend, True, "w4a16"),
        )

    # Binding admits original images, even when the payload has no changed
    # bytes and the secondary MMA view is reinterpreted as U8.
    if served_dtype == torch.float8_e5m2:
        with pytest.raises(fi._DtypeMismatch, match="torch.float8_e5m2"):
            bind()
    else:
        assert bind().storage
    for tensor, original in zip(tensors, before, strict=True):
        assert torch.equal(tensor.contiguous().view(torch.uint8), original)


def test_blocked_scale_bindings_check_only_the_weight_element_width():
    # A one-byte weight outside its allowed set is reported once, on the weight binding. MegaMoE serves the packed
    # weights through a transposed view of contiguous [128, 32] rows.
    layer = SimpleNamespace(
        w2_weight=torch.zeros((1, 128, 32), dtype=torch.uint8).transpose(1, 2).view(torch.float8_e4m3fn),
        w2_weight_scale=torch.zeros((1, 512), dtype=torch.uint8),
    )
    layout = FlashInferLayout("nvfp4", "megamoe", True, "w4a16")
    with pytest.raises(fi._DtypeMismatch, match="nvfp4 weight served=torch.float8_e4m3fn"):
        layout.expert_destinations(layer, 0, "down", "weight", (128, 32), "U8")
    assert layout.expert_destinations(layer, 0, "down", "weight_scale", (128, 4), "F8_E4M3")
    # Scale geometry still requires the weight's element width.
    layer.w2_weight = torch.zeros((1, 128, 16), dtype=torch.int16).transpose(1, 2)
    with pytest.raises(fi._DtypeMismatch, match="1-byte storage"):
        layout.expert_destinations(layer, 0, "down", "weight_scale", (128, 4), "F8_E4M3")


def test_layout_admission_rejects_conversion_and_unsupported_pairs():
    nv = quantized_layer(
        "ModelOptNvFp4FusedMoEMethod",
        _is_cutedsl_v2_standard=True,
        load_up_proj_weight_first=True,
        quant_config=SimpleNamespace(is_checkpoint_nvfp4_serialized=True),
    )
    # Each primary runner serves the activation mode its warmed consumer records: CuTe DSL's wrapper, MegaMoE's config
    # class.
    for mode in ("w4a4", "w4a16"):
        nv._cutedsl_wrapper.quant_mode = mode
        assert identify_expert_layout(nv, "flashinfer_cutedsl") == FlashInferLayout(
            "nvfp4", "cutedsl", True, mode, True
        )
    nv._flashinfer_megamoe_layer = None
    with pytest.raises(ValueError, match="FlashInfer MegaMoE must be warmed before admission"):
        identify_expert_layout(nv, "flashinfer_megamoe")
    for config, mode in (
        ("Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig", "w4a4"),
        ("Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig", "w4a16"),
    ):
        nv._flashinfer_megamoe_layer = SimpleNamespace(_megakernel_config=type(config, (), {})())
        assert identify_expert_layout(nv, "flashinfer_megamoe") == FlashInferLayout(
            "nvfp4", "megamoe", True, mode, False
        )
    nv._flashinfer_megamoe_layer._megakernel_config = object()
    with pytest.raises(ValueError, match="unsupported FlashInfer NVFP4 MegaMoE"):
        identify_expert_layout(nv, "flashinfer_megamoe")
    # On SM100 and SM103 SGLang main interleaves 16-row groups (swap-AB), a layout no CUDA check has compared yet;
    # the warmed wrapper records the decision. The TRTLLM layout of the same method is unaffected.
    nv._cutedsl_wrapper.swap_ab = True
    with pytest.raises(ValueError, match="does not yet admit SGLang main's swap-AB CuTe DSL expert layout"):
        identify_expert_layout(nv, "flashinfer_cutedsl")
    assert identify_expert_layout(nv, "flashinfer_trtllm").backend == "trtllm"
    # CuTe DSL admits only the standard layout of a warmed wrapper in a known mode.
    nv._cutedsl_wrapper.quant_mode = "fp8"
    with pytest.raises(ValueError, match="^unsupported CuTe DSL quant mode: fp8$"):
        identify_expert_layout(nv, "flashinfer_cutedsl")
    nv._cutedsl_wrapper = None
    with pytest.raises(ValueError, match="CuTe DSL warmup must initialize its wrapper"):
        identify_expert_layout(nv, "flashinfer_cutedsl")
    nv.quant_method._is_cutedsl_v2_standard = False
    with pytest.raises(ValueError, match="GPU delta requires the standard CuTe DSL layout"):
        identify_expert_layout(nv, "flashinfer_cutedsl")
    nv.quant_config.is_checkpoint_nvfp4_serialized = False
    with pytest.raises(ValueError, match="on-load"):
        identify_expert_layout(nv, "flashinfer_cutedsl")


def _secondary_layer(method, *, gated=True, scale_inv=None, gate_up_interleaved=False, **attrs):
    layer = quantized_layer(method, **attrs)
    layer.moe_runner_config = SimpleNamespace(is_gated=gated, gate_up_interleaved=gate_up_interleaved)
    if scale_inv is not None:
        # The live dtype of the loader's MXFP4 scale parameter tells the SGLang trees' scale storages apart.
        layer.w13_weight_scale_inv = torch.zeros((1, 2, 2), dtype=scale_inv)
    return layer


def _bf16(gated=True):
    # load_up_proj_weight_first follows FlashInfer CUTLASS, so TRTLLM's BF16 experts load gate first.
    return _secondary_layer(
        "UnquantizedFusedMoEMethod", gated=gated, use_flashinfer_trtllm_moe=True, load_up_proj_weight_first=False
    )


def _nvfp4(gated=True):
    return _secondary_layer(
        "ModelOptNvFp4FusedMoEMethod",
        gated=gated,
        _is_cutedsl_v2_standard=True,
        load_up_proj_weight_first=gated,
        quant_config=SimpleNamespace(is_checkpoint_nvfp4_serialized=True),
    )


def test_runner_and_dispatcher_admission(monkeypatch):
    allowed = {
        ("flashinfer_trtllm", "none"),
        ("flashinfer_trtllm_routed", "none"),
        ("flashinfer_mxfp4", "none"),
        ("flashinfer_cutedsl", "none"),
        ("flashinfer_megamoe", "flashinfer_megamoe"),
    }
    for runner in MoeRunnerBackend:
        for dispatcher in MoeA2ABackend:
            pair = (runner.value, dispatcher.value)
            assert admits_moe_dispatcher(*pair) == (pair in allowed), pair

    layer = _nvfp4()
    layer.quant_method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL)
    assert _identify_moe_layout(layer) == identify_expert_layout(layer, "flashinfer_cutedsl")
    # Direct TRTLLM wrappers read the launch setting. Former aliases and unresolved runners must fail closed.
    launch = SimpleNamespace(moe=SimpleNamespace())
    monkeypatch.setattr("sglang.srt.runtime_context.get_exec", lambda: launch)
    for runner in ("auto", "megamoe", "experimental_sgl_trtllm", "trtllm", "cutedsl"):
        launch.moe.moe_runner_backend = runner
        with pytest.raises(ValueError, match=f"unsupported GPU delta MoE runner backend: {runner}; supported: "):
            _identify_moe_layout(_bf16())


def _fp8(*, serialized=True, block128=False, **attrs):
    # Fp8MoEMethod: block-128 UE8M0 scales, MXFP8 (use_mxfp8) or FP4 experts (is_fp4_expert).
    config = SimpleNamespace(is_checkpoint_fp8_serialized=serialized)
    if block128:
        config.scale_fmt, config.weight_block_size = "ue8m0", [128, 128]
    return _secondary_layer("Fp8MoEMethod", quant_config=config, **attrs)


def _mxfp4(scale_inv, dequant=False):
    return _secondary_layer(
        "Mxfp4FlashinferTrtllmMoEMethod", scale_inv=scale_inv, _fp8=SimpleNamespace(dequant_fp4_to_fp8=dequant)
    )


def _mxfp4_method(**flags):
    # Mxfp4MoEMethod's own SM100 TRTLLM kernel over separate gate/up halves, unless a flag says otherwise.
    native = {"_fi_kernel": "trtllm_sm100", "use_mega_moe": False, "use_deep_gemm": False, "use_marlin": False}
    return _secondary_layer("Mxfp4MoEMethod", **(native | flags))


def _fp4_experts(scale_inv=None):
    return _fp8(is_fp4_expert=True, scale_inv=scale_inv)


def test_secondary_rows_identify_or_fail_closed(monkeypatch):
    def identify(layer, runner, official_shuffle):
        # Both SGLang trees read SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE once, at import.
        monkeypatch.setattr(mxfp4_flashinfer_trtllm_moe, "_USE_OFFICIAL_SHUFFLE", official_shuffle)
        return identify_expert_layout(layer, runner)

    def admits(layer, runner, expected, scale=None, official_shuffle=True):
        layout = identify(layer, runner, official_shuffle)
        assert layout == expected
        if scale is not None:
            # An MXFP4 storage's scale parameter suffix and served dtypes.
            assert (layout.pair.scale, layout.pair.scale_served) == scale
        return layout

    def fails_closed(layer, runner, message, official_shuffle=True):
        with pytest.raises(ValueError, match=message):
            identify(layer, runner, official_shuffle)

    # One row per admitted secondary storage.
    admits(_bf16(), "flashinfer_trtllm", FlashInferLayout("bf16", "trtllm", True))
    admits(_bf16(gated=False), "flashinfer_trtllm", FlashInferLayout("bf16", "trtllm", False))
    admits(_fp8(block128=True), "flashinfer_trtllm", FlashInferLayout("fp8-block128-ue8m0", "trtllm", True))
    for runner in ("flashinfer_trtllm", "flashinfer_trtllm_routed"):
        admits(_nvfp4(), runner, FlashInferLayout("nvfp4", "trtllm", True, "w4a4", True))
    admits(_nvfp4(gated=False), "flashinfer_cutedsl", FlashInferLayout("nvfp4", "cutedsl", False, "w4a4", True))
    admits(_fp8(use_mxfp8=True), "flashinfer_trtllm", FlashInferLayout("mxfp8", "trtllm", True))
    admits(_fp8(use_mxfp8=True), "flashinfer_megamoe", FlashInferLayout("mxfp8", "megamoe", True))
    # MXFP4: the companion tree keeps the loader's FP32 scales beside its kernel images; SGLang main writes the kernel
    # image in their place.
    trtllm = FlashInferLayout("mxfp4", "trtllm", True, up_first=True)
    megamoe = FlashInferLayout("mxfp4", "megamoe", True)
    ue8m0, e4m3, i32 = (torch.uint8, torch.int8, torch.float8_e8m0fnu), (torch.float8_e4m3fn,), (torch.int32,)
    companion = admits(_mxfp4(torch.float32), "flashinfer_mxfp4", trtllm, ("weight_scale_inv_shuffled", ue8m0))
    # The companion tree registers the GEMM1 clamp on the layer, so the consumer snapshot pins it there.
    assert companion.pair.consumers == ("_gemm1_clamp_limit_tensor",)
    trtllm_main = replace(trtllm, override=fi._MXFP4_TRTLLM_MAIN)
    admits(_mxfp4(torch.float8_e4m3fn), "flashinfer_mxfp4", trtllm_main, ("weight_scale_inv", e4m3))
    admits(_fp4_experts(torch.float32), "flashinfer_megamoe", megamoe, ("weight_scale_inv_shuffled", i32))
    megamoe_main = replace(megamoe, override=fi._MXFP4_MEGAMOE_MAIN)
    admits(_fp4_experts(torch.int32), "flashinfer_megamoe", megamoe_main, ("weight_scale_inv", i32))
    # Mxfp4MoEMethod prepares its own layout and never reads the shuffle switch; its kernel reads the loader's scale
    # names and owns no clamp tensor.
    native = _mxfp4_method()
    native_layout = replace(trtllm, override=fi._MXFP4_MOE_METHOD)
    layout = admits(native, "flashinfer_mxfp4", native_layout, ("weight_scale", e4m3), official_shuffle=False)
    for stem in ("w13", "w2"):
        setattr(native, f"{stem}_weight", torch.zeros((1, 64, 32), dtype=torch.uint8))
        setattr(native, f"{stem}_weight_scale", torch.zeros((1, 128, 4), dtype=torch.float8_e4m3fn))
    *_, w13_scale, w2_scale = layout.consumer_objects(native)
    assert w13_scale is native.w13_weight_scale and w2_scale is native.w2_weight_scale

    # One row per fail-closed rule.
    fails_closed(_bf16(), "flashinfer_megamoe", "no native FlashInfer BF16")
    fails_closed(_fp8(block128=True), "flashinfer_megamoe", "native UE8M0")
    fails_closed(_fp8(serialized=False), "flashinfer_trtllm", "native UE8M0")
    fails_closed(_fp8(serialized=False, use_mxfp8=True), "flashinfer_trtllm", "serialized MXFP8")
    fails_closed(_fp8(use_mxfp8=True), "flashinfer_cutedsl", "no native FP8/MXFP4 path")
    fails_closed(_fp8(use_mxfp8=True, convert_mxfp8_to_block=True), "flashinfer_trtllm", "requantization")
    fails_closed(_mxfp4(torch.float32, dequant=True), "flashinfer_mxfp4", "native packed")
    # The legacy shuffle keeps [w3; w1] halves, so 254 of 256 w13 rows would sit where group 1 does not map them.
    legacy = "requires SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE=true"
    fails_closed(_mxfp4(torch.float32), "flashinfer_mxfp4", legacy, official_shuffle=False)
    unrecognized = "unrecognized MXFP4 expert scale storage: w13_weight_scale_inv is "
    fails_closed(_mxfp4(torch.uint8), "flashinfer_mxfp4", unrecognized + "torch.uint8")
    fails_closed(_fp4_experts(torch.float8_e4m3fn), "flashinfer_megamoe", unrecognized + "torch.float8_e4m3fn")
    fails_closed(_mxfp4_method(_fi_kernel="triton"), "flashinfer_mxfp4", "unsupported native MXFP4")
    # MegaMoE rebuilds DeepGEMM weights even beside the flashinfer_mxfp4 runner, whose _fi_kernel stays set.
    transforms = "does not admit MXFP4 MegaMoE, DeepGEMM or Marlin expert transforms"
    fails_closed(_mxfp4_method(use_mega_moe=True), "flashinfer_mxfp4", transforms)
    fails_closed(_mxfp4_method(gate_up_interleaved=True), "flashinfer_mxfp4", "separate gate/up halves")
    fails_closed(_fp4_experts(), "flashinfer_trtllm", "require its MXFP4 method")
    # A non-gated layer keeps only up in its W13 slot, its NVFP4 global scale included.
    scales = SimpleNamespace(
        _map_global_expert_id_to_local_expert_id=lambda expert: expert,
        w13_weight_scale_2=torch.arange(3.0),
        w2_weight_scale_2=torch.arange(3.0),
    )
    non_gated = FlashInferLayout("nvfp4", "trtllm", False, "w4a4")
    meta = {"dtype": "F32", "shape": [], "encoding": "xor_raw_bytes"}
    for projection, slot in (("up", scales.w13_weight_scale_2[1]), ("down", scales.w2_weight_scale_2[1])):
        binding = _moe_binding(projection, meta, scales, 1, projection, "weight_scale_2", non_gated)
        assert binding.storage[0].data_ptr() == slot.data_ptr()
    with pytest.raises(ValueError, match="^non-gated experts have no gate projection$"):
        _moe_binding("gate", meta, scales, 1, "gate", "weight_scale_2", non_gated)


@pytest.mark.parametrize("fmt,backend", [("nvfp4", "unknown"), ("unknown", "trtllm"), ("bf16", "megamoe")])
def test_unsupported_layouts_cannot_bypass_admission(fmt, backend):
    # Down and non-gated projections skip interleaving, but must still reject
    # unsupported layouts before reading or constructing live tensor views.
    for projection, gated in (("up", True), ("down", True), ("up", False)):
        with pytest.raises(ValueError, match="unsupported"):
            fi.FlashInferLayout(fmt, backend, gated).expert_destinations(
                SimpleNamespace(),
                0,
                projection,
                "weight",
                (128, 64),
                "U8",
            )


def test_nonalias_scale_image_and_incorrect_geometry_fail_closed():
    layer = SimpleNamespace(
        w2_weight=torch.zeros((1, 128, 32), dtype=torch.uint8),
        w2_weight_scale=torch.zeros((1, 128, 4), dtype=torch.uint8),
        w2_blockscale_swizzled=torch.zeros((1, 128, 4), dtype=torch.uint8),
    )
    layout = fi.FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", True)
    with pytest.raises(ValueError, match="does not alias"):
        layout.expert_destinations(layer, 0, "down", "weight_scale", (128, 4), "F8_E4M3")
    layer.w2_weight_scale = layer.w2_blockscale_swizzled
    with pytest.raises(ValueError, match="row storage"):
        layout.expert_destinations(layer, 0, "down", "weight_scale", (129, 4), "F8_E4M3")
    layer.w2_weight = torch.zeros((1, 128), dtype=torch.uint8)
    with pytest.raises(ValueError, match="packed weight geometry"):
        layout.expert_destinations(layer, 0, "down", "weight", (128, 1), "U8")


@pytest.mark.parametrize(
    "projection,suffix,tp_rank,tp_size,partition",
    [
        ("gate", "weight", 1, 2, 128),
        ("gate", "weight_scale", 1, 2, 128),
        ("down", "weight", 1, 2, 128),
        ("down", "weight_scale", 1, 2, 128),
        ("down", "weight_scale", 3, 4, 64),
    ],
)
def test_trtllm_tp_preserves_padded_loader_offsets(projection, suffix, tp_rank, tp_size, partition):
    # A 192-wide checkpoint has logical TP2 width 96, but this loader partitions
    # by its padded width 128: rank one owns [128:192], not [96:192].
    scale = suffix == "weight_scale"
    divisor = 16 if scale else 2
    shape = (192, 128 // divisor) if projection == "gate" else (128, 192 // divisor)
    rows = 2 * partition if projection == "gate" else 128
    cols = (128 if projection == "gate" else partition) // divisor
    logical = torch.zeros(rows, cols, dtype=torch.uint8)
    physical = blocked_scales_padded(shuffled_rows(logical)) if scale else shuffled_rows(logical)
    stem = "w13" if projection == "gate" else "w2"
    layer = SimpleNamespace(
        moe_tp_size=tp_size,
        moe_tp_rank=tp_rank,
        use_padded_loading=True,
        intermediate_size_per_partition=partition,
        _map_global_expert_id_to_local_expert_id=lambda _: 0,
    )
    weight_cols = (128 if projection == "gate" else partition) // 2
    setattr(
        layer,
        stem + "_weight",
        torch.zeros(rows, weight_cols, dtype=torch.uint8).unsqueeze(0),
    )
    setattr(layer, stem + "_" + suffix, physical.unsqueeze(0))
    binding = _moe_binding(
        "expert",
        {"dtype": "F8_E4M3" if scale else "U8", "shape": list(shape)},
        layer,
        0,
        projection,
        suffix,
        FlashInferLayout("nvfp4", "trtllm", True, "w4a4"),
    )
    if tp_size == 4:
        # Padded rank three owns no checkpoint bytes and must leave its entire
        # zero-filled consumer image untouched.
        assert binding is None
    else:
        mask = torch.randint(256, shape, dtype=torch.uint8)
        for view in binding.apply_bindings():
            source = view.selected_bytes(mask.flatten())
            for target in view.destinations:
                target.bitwise_xor_(source.reshape(target.shape))
        if projection == "gate":
            logical[:128:2] = mask[128:]
        else:
            logical[:, : 64 // divisor] = mask[:, 128 // divisor :]
    expected = shuffled_rows(logical)
    if scale:
        expected = blocked_scales_padded(expected)
    assert torch.equal(physical, expected)


@pytest.mark.parametrize(
    "fmt,backend,group,up_first",
    [
        ("nvfp4", "trtllm", 1, False),
        ("mxfp8", "trtllm", 1, False),
        ("mxfp4", "trtllm", 1, True),
        ("bf16", "trtllm", 1, False),
        ("nvfp4", "megamoe", 16, False),
        ("mxfp8", "megamoe", 32, False),
    ],
)
def test_factored_destinations_match_loader_permutations(fmt, backend, group, up_first):
    # Full independent physical replays prove gate/up order, bytes of BF16,
    # scale row shuffling, both Mega transpose policies, and excluded padding.
    # W2 takes the same row shuffle, column blocks and transposes as W13, without the gate/up interleave.
    rows, physical_cols = 192, 384
    dtype = "BF16" if fmt == "bf16" else "F8_E4M3" if fmt == "mxfp8" else "U8"
    elem = 2 if fmt == "bf16" else 1
    layout = fi.FlashInferLayout(fmt, backend, True, up_first=up_first)
    divisor = {"nvfp4": 8, "mxfp8": 32, "mxfp4": 16}.get(fmt)

    def weight_transform(value, gated):
        if gated:
            value = interleave(value, group)
        if backend == "trtllm":
            value = shuffled_rows(value)
        if fmt == "bf16":
            return (
                value.reshape(value.shape[0], physical_cols // 128, 128)
                .transpose(0, 1)
                .contiguous()
                .view(torch.bfloat16)
            )
        if backend == "megamoe":
            value = value.T
            if fmt == "mxfp8":
                value = value.contiguous()
        return value

    def scale_transform(value, gated):
        if gated:
            value = interleave(value, group)
        if backend == "trtllm":
            value = shuffled_rows(value)
        return blocked_scales_padded(value)

    layer = SimpleNamespace()
    for stem, projections in (("w13", ("gate", "up")), ("w2", ("down",))):
        gated = stem == "w13"
        original = torch.randint(256, ((2 if gated else 1) * rows, physical_cols), dtype=torch.uint8)
        actual = weight_transform(original.clone(), gated)
        setattr(layer, f"{stem}_weight", actual.unsqueeze(0))
        scales = None
        if divisor:
            scales = torch.randint(256, (original.shape[0], physical_cols // divisor), dtype=torch.uint8)
            scale_name = {
                "nvfp4": "weight_scale",
                "mxfp8": "weight_scale_inv",
                "mxfp4": "weight_scale_inv_shuffled",
            }[fmt]
            setattr(layer, f"{stem}_{scale_name}", scale_transform(scales, gated).unsqueeze(0))
            if fmt == "mxfp4":
                setattr(layer, f"{stem}_weight_scale_inv", torch.ones_like(scales, dtype=torch.float32))
        for projection in projections:
            half = int((projection == "up") != up_first) if gated else 0
            # Tail rows and columns create coarse affine rectangles, never padding writes.
            nr, nc = rows - 3, physical_cols // elem - 3
            mask = torch.randint(256, (nr, nc * elem), dtype=torch.uint8)
            segments = layout.expert_destinations(layer, 0, projection, "weight", (nr, nc), dtype)
            if elem == 2:
                apply_segments(segments, mask.view(torch.bfloat16))
            else:
                apply_segments(segments, mask)
            original[half * rows : half * rows + nr, : nc * elem] ^= mask
            assert torch.equal(actual.view(torch.uint8), weight_transform(original, gated).view(torch.uint8))
            if scales is not None:
                nc = scales.shape[1] - 1
                mask = torch.randint(256, (nr, nc), dtype=torch.uint8)
                segments = layout.expert_destinations(
                    layer,
                    0,
                    projection,
                    "weight_scale",
                    (nr, nc),
                    "F8_E4M3" if fmt == "nvfp4" else "U8",
                )
                apply_segments(segments, mask)
                scales[half * rows : half * rows + nr, :nc] ^= mask
                target = fi._scale_parameter(layer, stem, layout)[0]
                assert torch.equal(target.flatten(), scale_transform(scales, gated))
        if fmt == "mxfp4":
            # The companion tree's FP32 loader scales stay untouched beside the kernel image.
            assert torch.equal(
                getattr(layer, f"{stem}_weight_scale_inv"), torch.ones_like(scales, dtype=torch.float32)
            )


@pytest.mark.parametrize(
    "layout,suffix,dtype,error,message",
    [
        (
            FlashInferLayout("nvfp4", "megamoe", True, "w4a16"),
            "weight",
            "I8",
            _DtypeMismatch,
            "^checkpoint=I8; nvfp4 weights require canonical U8, served=torch.float4_e2m1fn_x2$",
        ),
        (
            FlashInferLayout("nvfp4", "megamoe", True, "w4a16"),
            "weight_scale",
            "U8",
            _DtypeMismatch,
            "^checkpoint=U8; nvfp4 block scales require canonical F8_E4M3$",
        ),
        (
            FlashInferLayout("nvfp4", "cutedsl", True, "w4a16", True),
            "weight_scale",
            "U8",
            _DtypeMismatch,
            "^checkpoint=U8; NVFP4 block scales require canonical F8_E4M3 bytes$",
        ),
        # Packed FP4 block scales bind only as weight_scale; block-128 and MXFP8 bind weight_scale_inv before this
        # rule. Mxfp4FlashinferTrtllmMoEMethod fills FP32 w*_weight_scale_inv by numeric cast, so U8 codes bound
        # under that name never reach the kernel image unchanged.
        (
            FlashInferLayout("nvfp4", "megamoe", True, "w4a16"),
            "weight_scale_inv",
            "U8",
            ValueError,
            "^unsupported FlashInfer expert suffix: weight_scale_inv$",
        ),
        (
            FlashInferLayout("mxfp4", "trtllm", True, up_first=True),
            "weight_scale_inv",
            "U8",
            ValueError,
            "^unsupported FlashInfer expert suffix: weight_scale_inv$",
        ),
    ],
    ids=["megamoe-weight", "megamoe-scale", "cutedsl-scale", "megamoe-scale-inv", "mxfp4-trtllm-scale-inv"],
)
def test_fp4_leaves_require_canonical_dtypes_and_scale_names(layout, suffix, dtype, error, message):
    # Packed weights are U8 only, and the loaders fill F8_E4M3 scale parameters by copy_, which casts any other dtype
    # numerically. MegaMoE serves the packed weights as float4_e2m1fn_x2 through a transposed view; CuTe DSL checks
    # its scale dtype before reading any storage.
    weight = torch.zeros((1, 256, 64), dtype=torch.uint8)
    if layout.backend != "trtllm":
        weight = weight.transpose(1, 2).view(torch.float4_e2m1fn_x2)
    layer = SimpleNamespace(
        moe_tp_size=1, _map_global_expert_id_to_local_expert_id=lambda expert: 0, w13_weight=weight
    )
    with pytest.raises(error, match=message):
        _moe_binding(
            f"model.layers.0.mlp.experts.0.up_proj.{suffix}",
            {"dtype": dtype, "shape": [128, 8]},
            layer,
            0,
            "up",
            suffix,
            layout,
        )


def test_block128_complete_scale_view_and_consumer_snapshot():
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(runner=None),
        moe_runner_config=SimpleNamespace(is_gated=True),
        w13_weight=torch.zeros((1, 512, 256), dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros((1, 256, 256), dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones((1, 4, 2), dtype=torch.float32),
        w2_weight_scale_inv=torch.ones((1, 2, 2), dtype=torch.float32),
        _flashinfer_trtllm_gemm1_alpha=None,
        _flashinfer_trtllm_gemm1_beta=None,
        _flashinfer_trtllm_gemm1_clamp_limit=None,
    )
    layout = fi.FlashInferLayout("fp8-block128-ue8m0", "trtllm", True)
    target = layout.expert_scale_target(layer, 0, "up", (2, 2))
    target.fill_(3)
    assert torch.equal(layer.w13_weight_scale_inv[0, :2], torch.ones((2, 2)))
    assert torch.equal(layer.w13_weight_scale_inv[0, 2:], torch.full((2, 2), 3.0))
    before = tuple(map(id, layout.consumer_objects(layer)))
    layer._flashinfer_trtllm_gemm1_beta = torch.ones(1)
    assert tuple(map(id, layout.consumer_objects(layer))) != before
    # A consumer field the pair names may be None, but a missing one is an unsupported layout.
    del layer._flashinfer_trtllm_gemm1_alpha
    with pytest.raises(
        ValueError,
        match="^unsupported FlashInfer expert layout: SimpleNamespace has no _flashinfer_trtllm_gemm1_alpha$",
    ):
        layout.consumer_objects(layer)
    # A block-scale suffix without byte-layout scales is fatal, not a collected dtype mismatch.
    with pytest.raises(ValueError, match="unsupported FlashInfer block-scale format") as error:
        layout.expert_destinations(layer, 0, "up", "weight_scale", (2, 2), "U8")
    assert not isinstance(error.value, fi._DtypeMismatch)


def test_megamoe_mxfp4_packed_ue8m0_views():
    # The pre-Mega input is four exponent bytes per I32, column-major by MN.
    # Compose the actual two tensor transforms, independent of address formulas.
    rows, cols = 256, 512
    weight = torch.randint(256, (2 * rows, cols), dtype=torch.uint8)
    scales = torch.randint(256, (2 * rows, cols // 16), dtype=torch.uint8)

    def transform_scale(value):
        packed = value.contiguous().view(torch.int32)
        packed = packed.T.contiguous().T
        interleaved = interleave(packed, 8)
        transposed = interleaved.reshape(-1, 4, 32, packed.shape[1]).transpose(1, 2).reshape_as(packed)
        return transposed.T.contiguous().T

    actual_weight = interleave(weight, 8)
    actual_scale = transform_scale(scales)
    layer = SimpleNamespace(
        w13_weight=actual_weight.unsqueeze(0),
        w13_weight_scale_inv=torch.ones_like(scales, dtype=torch.float32),
        w13_weight_scale_inv_shuffled=actual_scale.unsqueeze(0),
        quant_method=SimpleNamespace(),
    )
    layout = fi.FlashInferLayout("mxfp4", "megamoe", True)
    for projection, half in (("gate", 0), ("up", 1)):
        for suffix, original, live, transform in (
            ("weight", weight, actual_weight, lambda x: interleave(x, 8)),
            ("weight_scale", scales, actual_scale, transform_scale),
        ):
            r, c = rows - 5, original.shape[1] - 3
            mask = torch.randint(256, (r, c), dtype=torch.uint8)
            segments = layout.expert_destinations(layer, 0, projection, suffix, (r, c), "U8")
            apply_segments(segments, mask)
            original[half * rows : half * rows + r, :c] ^= mask
            assert torch.equal(live, transform(original))
    assert torch.equal(layer.w13_weight_scale_inv, torch.ones_like(scales, dtype=torch.float32))


@pytest.mark.parametrize("backend", ["cutedsl", "megamoe"])
def test_consumer_snapshot_tracks_live_scale_storage(backend):
    def scalar():
        return torch.ones(1, dtype=torch.float32)

    layer = SimpleNamespace(
        quant_method=SimpleNamespace(runner=None),
        moe_runner_config=SimpleNamespace(is_gated=True),
        w13_weight=torch.zeros((1, 256, 128), dtype=torch.uint8),
        w2_weight=torch.zeros((1, 128, 128), dtype=torch.uint8),
        w13_weight_scale=torch.zeros(512, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(512, dtype=torch.uint8),
        g1_alphas=scalar(),
        g1_alphas_up=scalar(),
        g2_alphas=scalar(),
        w13_input_scale_quant=scalar(),
        w2_input_scale_quant=scalar(),
    )
    if backend == "cutedsl":
        layer._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
        layer._cutedsl_scales = (scalar(), scalar(), scalar())
        layer._cutedsl_input_scale = scalar()
        layer.w13_blockscale_swizzled = layer.w13_weight_scale
        layer.w2_blockscale_swizzled = layer.w2_weight_scale
        layout = FlashInferLayout("nvfp4", backend, True, "w4a16", True)
        changed = ("_cutedsl_scales",)
    elif backend == "megamoe":
        # Current MegaMoE stages calibration per forward from the layer; its
        # config no longer owns fc1_alpha/fc2_alpha/fc1_norm_const tensors.
        layer._flashinfer_megamoe_layer = SimpleNamespace(
            _megakernel_config=SimpleNamespace(input_norm_const=1.0),
            _kernel=object(),
            _transformed=(
                (layer.w13_weight, layer.w13_weight_scale),
                (layer.w2_weight, layer.w2_weight_scale),
            ),
        )
        layer._flashinfer_megamoe_forward = object()
        layer._flashinfer_megamoe_input_norm_const = 1.0
        layout = FlashInferLayout("nvfp4", backend, True, "w4a16")
        changed = ("g1_alphas", "g2_alphas", "w2_input_scale_quant")
    else:
        raise ValueError(f"unsupported consumer backend: {backend}")
    for name in changed:
        snapshot = ConsumerSnapshot(lambda: layout.consumer_objects(layer))
        snapshot.check_objects()
        value = getattr(layer, name)
        # A rebuilt consumer holds new tensors; a CuTe DSL rewarm stores a new scale tuple.
        setattr(layer, name, tuple(map(torch.clone, value)) if isinstance(value, tuple) else value.clone())
        with pytest.raises(RuntimeError, match="consumer storage changed"):
            snapshot.check_objects()


def test_missing_layout_fields_are_admission_errors():
    # A field the identified pair names (its scale parameter and consumers) that another SGLang tree lacks is an
    # unsupported layout: a ValueError, which describe reports with the rank prefix, never an AttributeError.
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_method=SimpleNamespace(runner=None),
        # MegaMoE serves the packed weights through transposed views.
        w13_weight=torch.zeros((1, 256, 64), dtype=torch.uint8).transpose(1, 2),
        w2_weight=torch.zeros((1, 128, 64), dtype=torch.uint8).transpose(1, 2),
        w13_weight_scale=torch.zeros((1, 256 * 8), dtype=torch.uint8),
    )
    layout = FlashInferLayout("nvfp4", "megamoe", True, "w4a16")
    with pytest.raises(
        ValueError, match="^unsupported FlashInfer expert layout: SimpleNamespace has no w2_weight_scale$"
    ):
        layout.consumer_objects(layer)
    with pytest.raises(
        ValueError, match="^unsupported FlashInfer expert layout: SimpleNamespace has no w2_weight_scale$"
    ):
        layout.expert_destinations(layer, 0, "down", "weight_scale", (128, 8), "F8_E4M3")
