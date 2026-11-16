"""FlashInfer routed-expert layouts as views of live consumer bytes (CUDA receiver only).

Mirrors SGLang's FlashInfer MoE runners (``layers/moe/moe_runner/flashinfer_trtllm.py`` and
``flashinfer_cutedsl.py``, ``layers/moe/flashinfer_megamoe.py``) and the quantization methods that
prepare their weights: TRTLLM and its routed variant, the MXFP4 runner, CuTe DSL and MegaMoE, with
every format they admit and NVFP4's derived calibration. They share one module because they share its
row permutations, blocked scale tiles and consumer fields. These admission-time helpers factor each
runner's fixed permutations into the affine byte views of ``byte_views.py``; they never transform a
delta, requantize a weight, or allocate a weight image. Admitted runner values are listed only in
``kernel_backends/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from miles.backends.sglang_utils.gpu_delta.byte_views import (
    DestinationSegment,
    _byte_view,
    _check_served,
    _DtypeMismatch,
    _factored_segments,
    _same_image,
    _unique_images,
)
from miles.backends.sglang_utils.gpu_delta.derived import DerivedUpdate, _validate_checks


@dataclass(frozen=True)
class _Format:
    """Canonical checkpoint bytes shared by every backend of one format."""

    weight: str
    weight_served: tuple[torch.dtype, ...]
    scale: str | None = None
    scale_divisor: int = 0  # physical weight bytes per block scale


@dataclass(frozen=True)
class _Pair:
    """Live storage of one admitted (backend, format) expert layout."""

    scale: str | None  # scale parameter suffix
    group: int  # gate/up interleave rows; zero keeps plain halves
    shuffle: bool  # TRTLLM's 32-row transpose
    weight: str  # physical weight orientation: rows, blocked, transposed or k_major
    scale_image: str | None  # physical scale layout: blocked, cutedsl or deepgemm
    scale_served: tuple[torch.dtype, ...] = ()
    consumers: tuple[str, ...] = ()  # required TRTLLM consumer fields of the layer
    method_consumers: tuple[str, ...] = ()  # required consumer fields of its quantization method


_PACKED = (torch.uint8, torch.int8)
# SGLang's MegaMoE NVFP4 path serves FlashInfer's packed weights as float4_e2m1fn_x2,
# the checkpoint's two-E2M1-per-byte U8 bytes unchanged.
_NVFP4_PACKED = _PACKED + (torch.float4_e2m1fn_x2,)
_E4M3 = (torch.float8_e4m3fn, torch.uint8, torch.int8)
_UE8M0 = (torch.uint8, torch.int8, torch.float8_e8m0fnu)
# UE8M0 scale bytes a method types as float8_e4m3fn for its kernel, unchanged.
_E4M3_VIEW = (torch.float8_e4m3fn,)
_FORMATS = {
    "bf16": _Format("BF16", (torch.bfloat16,)),
    "nvfp4": _Format("U8", _NVFP4_PACKED, "F8_E4M3", 8),
    "mxfp4": _Format("U8", _PACKED, "U8", 16),
    "mxfp8": _Format("F8_E4M3", _E4M3, "U8", 32),
    "fp8-block128-ue8m0": _Format("F8_E4M3", _E4M3),
}
_TRTLLM_GEMM1 = (
    "_flashinfer_trtllm_gemm1_alpha",
    "_flashinfer_trtllm_gemm1_beta",
    "_flashinfer_trtllm_gemm1_clamp_limit",
)
_PAIRS = {
    ("trtllm", "bf16"): _Pair(None, 1, True, "blocked", None),
    ("trtllm", "nvfp4"): _Pair("weight_scale", 1, True, "rows", "blocked", _E4M3),
    ("trtllm", "mxfp4"): _Pair(
        "weight_scale_inv_shuffled", 1, True, "rows", "blocked", _UE8M0, ("_gemm1_clamp_limit_tensor",)
    ),
    ("trtllm", "mxfp8"): _Pair("weight_scale_inv", 1, True, "rows", "blocked", _UE8M0, _TRTLLM_GEMM1),
    # Block-128 projections use plain halves; their FP32 scales are complete values.
    ("trtllm", "fp8-block128-ue8m0"): _Pair("weight_scale_inv", 0, False, "rows", None, consumers=_TRTLLM_GEMM1),
    # Standard W4A4/W4A16 (the companion tree; SGLang main without swap-AB): Up64, Gate64, Up64, Gate64, ...
    ("cutedsl", "nvfp4"): _Pair("weight_scale", 64, False, "rows", "cutedsl", _E4M3),
    # Gate16, Up16, Gate16, Up16, ...
    ("megamoe", "nvfp4"): _Pair("weight_scale", 16, False, "transposed", "blocked", _E4M3),
    ("megamoe", "mxfp8"): _Pair("weight_scale_inv", 32, False, "k_major", "blocked", _UE8M0),
    # Retain checkpoint scales separately; target the shuffled consumer.
    ("megamoe", "mxfp4"): _Pair("weight_scale_inv_shuffled", 8, False, "rows", "deepgemm", (torch.int32,)),
}
# Mxfp4MoEMethod's TRTLLM kernel (Kimi-K3's compressed-tensors MXFP4) reads the loader's scale name, typed
# float8_e4m3fn over its shuffled, interleaved UE8M0 bytes in both SGLang trees, and owns no clamp tensor.
_MXFP4_MOE_METHOD = replace(_PAIRS["trtllm", "mxfp4"], scale="weight_scale", scale_served=_E4M3_VIEW, consumers=())
# The _PAIRS MXFP4 rows above are the companion SGLang tree's storage, which keeps the loader's FP32 scales in
# w*_weight_scale_inv beside the kernel images in w*_weight_scale_inv_shuffled. SGLang main writes the kernel
# image over w*_weight_scale_inv instead. Its Mxfp4FlashinferTrtllmMoEMethod
# (layers/quantization/mxfp4_flashinfer_trtllm_moe.py) types that image float8_e4m3fn, pads the intermediate to
# 128 outside every destination, reads constant output*_scale_scalar buffers and keeps the GEMM1 clamp on the
# method; its MegaMoE (layers/moe/flashinfer_megamoe.py) binds the DeepGEMM-packed scales there.
_MXFP4_TRTLLM_MAIN = replace(
    _PAIRS["trtllm", "mxfp4"],
    scale="weight_scale_inv",
    scale_served=_E4M3_VIEW,
    consumers=(
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
        "use_cuda_mxfp4_decode_moe",
    ),
    method_consumers=("_gemm1_clamp_limit_tensor",),
)
_MXFP4_MEGAMOE_MAIN = replace(_PAIRS["megamoe", "mxfp4"], scale="weight_scale_inv")


@dataclass(frozen=True)
class FlashInferLayout:
    """The ExpertLayout of one routed-expert layer on a FlashInfer runner."""

    format: str
    backend: str
    gated: bool
    quant_mode: str = ""
    up_first: bool = False
    # identify_layout sets this only for a method whose storage differs from its pair's row.
    override: _Pair | None = None

    @property
    def pair(self):
        pair = self.override or _PAIRS.get((self.backend, self.format))
        if pair is None:
            raise ValueError(f"unsupported FlashInfer expert layout: {self.backend}/{self.format}")
        return pair

    # ExpertLayout (kernel_backends/base.py). Each method forwards to the module function of the same
    # name, which takes the layout explicitly; inside a method body that name is the function.
    def expert_slices(self, layer, projection, suffix, shape):
        return expert_slices(layer, self, projection, suffix, shape)

    def expert_scale_target(self, layer, local_expert, projection, canonical_shape):
        return expert_scale_target(layer, self, local_expert, projection, canonical_shape)

    def expert_destinations(self, layer, local_expert, projection, suffix, canonical_shape, canonical_dtype):
        return expert_destinations(layer, self, local_expert, projection, suffix, canonical_shape, canonical_dtype)

    def consumer_objects(self, layer):
        return consumer_objects(layer, self)

    def expert_derived_updates(self, layer):
        return expert_derived_updates(layer, self)


def _backend(backend):
    # The storage family of each exact runner value kernel_backends/__init__.py routes here; the
    # routed and MXFP4 runners consume TRTLLM's shuffled storage.
    if backend in {"flashinfer_trtllm", "flashinfer_trtllm_routed", "flashinfer_mxfp4"}:
        resolved = "trtllm"
    elif backend == "flashinfer_cutedsl":
        resolved = "cutedsl"
    elif backend == "flashinfer_megamoe":
        resolved = "megamoe"
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {backend}")
    return resolved


def _mxfp4_trtllm_gated_interleave():
    """Whether SGLang's Mxfp4FlashinferTrtllmMoEMethod stored FlashInfer's gated row interleave.

    Both SGLang trees read SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE once, at import, into this module flag and
    interleave the gate/up rows only under it. The legacy shuffle stores 32-row shuffled [w3; w1] halves,
    which the TRTLLM MXFP4 pairs (group 1) do not describe.
    """
    from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe

    return mxfp4_flashinfer_trtllm_moe._USE_OFFICIAL_SHUFFLE


def _mxfp4_scale_storage(layer, main_served, main):
    """The MXFP4 scale storage of the SGLang tree that prepared this layer, from its live scale parameter.

    The loader's FP32 w*_weight_scale_inv is the companion tree's storage: no override of the _PAIRS row.
    The kernel image in its place, served as ``main_served``, is SGLang main's ``main`` pair.
    """
    served = _consumer_field(layer, "w13_weight_scale_inv").dtype
    if served == torch.float32:
        return None
    elif served == main_served:
        return main
    else:
        raise ValueError(f"unrecognized MXFP4 expert scale storage: w13_weight_scale_inv is {served}")


def _require_cutedsl_without_swap_ab(layer):
    """Admit the standard CuTe DSL layout only in its 64-row W13 interleave (the _PAIRS row).

    The companion tree interleaves 64-row groups with ``interleave_w13_halves``. SGLang main interleaves with
    FlashInfer's ``interleave_cute_dsl_moe_weights`` and builds the CuteDslMoEWrapper under one
    ``use_cutedsl_moe_swap_ab`` decision, made on the CUDA device the experts run on (CPU offloading moves them
    back between forwards): 64-row groups, except the swap-AB layout's 16-row groups on SM100 and SM103, which
    no CUDA check has compared yet. The warmed wrapper records that decision (FlashInfer 0.7.4 or later); the
    receiver README's FlashInfer section states the rule.
    """
    wrapper = layer._cutedsl_wrapper
    if (
        hasattr(wrapper, "swap_ab") and wrapper.swap_ab
    ):  # config-access-exempt: Wrappers before FlashInfer 0.7.4 lack it.
        raise ValueError(
            "GPU delta does not yet admit SGLang main's swap-AB CuTe DSL expert layout (16-row W13 groups on "
            "SM100 and SM103); serve W4A16 routed experts on flashinfer_megamoe (TRTLLM serves NVFP4 as W4A4)"
        )


def identify_layout(layer, backend):
    """Identify an existing inference path; do not select or construct one."""
    backend = _backend(backend)
    method = layer.quant_method
    name = type(method).__name__
    gated = bool(layer.moe_runner_config.is_gated)
    if name == "UnquantizedFusedMoEMethod":
        if backend != "trtllm" or not method.use_flashinfer_trtllm_moe:
            raise ValueError("no native FlashInfer BF16 expert path for this backend")
        layout = FlashInferLayout("bf16", backend, gated, up_first=method.load_up_proj_weight_first)
    elif name == "ModelOptNvFp4FusedMoEMethod":
        if not method.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("GPU delta does not admit on-load NVFP4 quantization")
        if backend == "trtllm":
            layout = FlashInferLayout(
                "nvfp4",
                backend,
                gated,
                "w4a4",
                up_first=method.load_up_proj_weight_first,
            )
        elif backend == "cutedsl":
            if not method._is_cutedsl_v2_standard:
                raise ValueError("GPU delta requires the standard CuTe DSL layout")
            if layer._cutedsl_wrapper is None:
                raise ValueError("CuTe DSL warmup must initialize its wrapper")
            mode = layer._cutedsl_wrapper.quant_mode
            if mode not in {"w4a4", "w4a16"}:
                raise ValueError(f"unsupported CuTe DSL quant mode: {mode}")
            _require_cutedsl_without_swap_ab(layer)
            layout = FlashInferLayout("nvfp4", backend, gated, mode, up_first=True)
        elif backend == "megamoe":
            mega = layer._flashinfer_megamoe_layer
            if mega is None:
                # SGLang main builds each MegaMoE layer on the first forward that routes tokens through it.
                raise ValueError("FlashInfer MegaMoE must be warmed before admission")
            config_name = type(mega._megakernel_config).__name__
            if config_name == "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig":
                mode = "w4a4"
            elif config_name == "Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig":
                mode = "w4a16"
            else:
                raise ValueError("unsupported FlashInfer NVFP4 MegaMoE consumer")
            layout = FlashInferLayout("nvfp4", backend, gated, mode, up_first=False)
        else:
            raise ValueError(f"unsupported NVFP4 expert backend: {backend}")
    elif name == "Mxfp4FlashinferTrtllmMoEMethod":
        if backend != "trtllm" or method._fp8.dequant_fp4_to_fp8:
            raise ValueError("MXFP4 TRTLLM requires native packed expert weights")
        if not _mxfp4_trtllm_gated_interleave():
            raise ValueError(
                "GPU delta requires SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE=true: MXFP4 TRTLLM experts "
                "without FlashInfer's gated row interleave are not admitted"
            )
        storage = _mxfp4_scale_storage(layer, torch.float8_e4m3fn, _MXFP4_TRTLLM_MAIN)
        layout = FlashInferLayout("mxfp4", backend, gated, up_first=True, override=storage)
    elif name == "Mxfp4MoEMethod":
        # These rebuild the expert weights in their own layouts, MegaMoE even beside a FlashInfer runner.
        if method.use_mega_moe or method.use_deep_gemm or method.use_marlin:
            raise ValueError("GPU delta does not admit MXFP4 MegaMoE, DeepGEMM or Marlin expert transforms")
        if backend != "trtllm" or method._fi_kernel != "trtllm_sm100":
            raise ValueError("unsupported native MXFP4 FlashInfer expert path")
        # The method pairs (up_i, gate_i) rows from separate gate/up halves only when they are not interleaved.
        if layer.moe_runner_config.gate_up_interleaved:
            raise ValueError("GPU delta requires Mxfp4MoEMethod experts with separate gate/up halves")
        layout = FlashInferLayout("mxfp4", backend, gated, up_first=True, override=_MXFP4_MOE_METHOD)
    elif name == "Fp8MoEMethod":
        if backend == "cutedsl":
            raise ValueError("ordinary CuTe DSL MoE has no native FP8/MXFP4 path")
        if method.dequant_fp4_to_fp8 or method.convert_mxfp8_to_block:
            raise ValueError("GPU delta does not admit weight requantization")
        config = method.quant_config
        if method.use_mxfp8:
            if not config.is_checkpoint_fp8_serialized:
                raise ValueError("GPU delta requires serialized MXFP8 weights")
            layout = FlashInferLayout("mxfp8", backend, gated)
        elif method.is_fp4_expert:
            if backend != "megamoe":
                raise ValueError("native FP4 TRTLLM experts require its MXFP4 method")
            storage = _mxfp4_scale_storage(layer, torch.int32, _MXFP4_MEGAMOE_MAIN)
            layout = FlashInferLayout("mxfp4", backend, gated, override=storage)
        elif (
            backend == "trtllm"
            and config.scale_fmt == "ue8m0"
            and config.weight_block_size == [128, 128]
            and config.is_checkpoint_fp8_serialized
        ):
            layout = FlashInferLayout("fp8-block128-ue8m0", backend, gated)
        else:
            raise ValueError("unsupported FlashInfer expert format; native UE8M0 is required")
    else:
        raise ValueError(f"unsupported FlashInfer expert quantization: {name}")
    return layout


def _cutedsl_scale_images(layer, stem, expert, allowed):
    primary = getattr(layer, f"{stem}_blockscale_swizzled")[
        expert
    ]  # config-access-exempt: Select the admitted gate/up or down CuTe scale image.
    source = getattr(layer, f"{stem}_weight_scale")[
        expert
    ]  # config-access-exempt: Select the admitted gate/up or down scale parameter.
    mma = getattr(
        layer, f"{stem}_blockscale_mma", None
    )  # config-access-exempt: CuTe may retain an additional MMA scale image for this projection.
    # Admit the original storage types; the MMA image is viewed as primary's dtype below.
    for image in (primary, source) if mma is None else (primary, mma, source):
        _check_served(image, allowed, "NVFP4 CuTe DSL scale")
    images = [primary]
    if mma is not None:
        # Public MMA axes: (32,4,mtiles,4,ktiles,E). This permutation
        # recovers physical order and creates no temporary scale tensor.
        physical = mma[..., expert].permute(2, 4, 0, 1, 3)
        if not physical.is_contiguous():
            raise ValueError("unsupported nonstandard CuTe DSL MMA physical strides")
        images.append(physical.view(primary.shape).view(primary.dtype))
    images = _unique_images(images)
    if not any(_same_image(source, image) for image in images):
        raise ValueError("NVFP4 source scale does not alias a prepared scale image")
    return images


def _row_map(row, group, half, shuffle):
    # These are the backend's fixed bit permutations, not a per-row index map:
    # gated groups, then TRTLLM's 32-row transpose (4 x 8 -> 8 x 4).
    if group:
        row = row // group * (2 * group) + half * group + row % group
    if shuffle:
        row = row // 32 * 32 + row % 4 * 8 + row % 32 // 4
    return row


def _blocked_scale_segments(images, rows, cols, padded_cols, mapped):
    """FlashInfer's to_blocked scales: [128, 4] tiles with transposed [4, 32] row factors."""

    def row_offset(row):
        row = mapped(row)
        return row // 128 * padded_cols * 128 + row % 32 * 16 + row % 128 // 32 * 4

    return _factored_segments(images, rows, cols, 1, row_offset, 4, 1, 512)


def _consumer_field(owner, name):
    """A consumer field the identified layout names; a missing one is an unsupported layout.

    Raised as ValueError, so describe reports it with the rank prefix like every admission error.
    """
    try:
        return getattr(owner, name)  # config-access-exempt: Read a field the admitted expert layout names.
    except AttributeError as error:
        raise ValueError(f"unsupported FlashInfer expert layout: {type(owner).__name__} has no {name}") from error


def _scale_parameter(layer, stem, layout):
    return _consumer_field(layer, f"{stem}_{layout.pair.scale}")


def _stem(projection):
    if projection == "down":
        return "w2"
    elif projection in {"gate", "up"}:
        return "w13"
    else:
        raise ValueError(f"unsupported FlashInfer expert projection: {projection}")


def _packed_image(image, layout):
    if image.element_size() != 1:
        raise _DtypeMismatch(f"{layout.format} weight layout requires 1-byte storage, served={image.dtype}")
    if image.ndim != 2:
        raise ValueError("unexpected FlashInfer packed weight geometry")
    return image


def _weight_image(layer, stem, expert, layout):
    """Return the live weight image, its physical rows and its physical byte columns.

    Scale bindings derive their geometry from this image, so only its element
    width is checked here; the weight's own binding admits its served dtype.
    """
    image = getattr(layer, f"{stem}_weight")[
        expert
    ]  # config-access-exempt: Select the admitted gate/up or down weight image.
    orientation = layout.pair.weight
    if orientation == "blocked":
        if image.element_size() != 2:
            raise _DtypeMismatch(f"{layout.format} weight layout requires 2-byte storage, served={image.dtype}")
        if image.ndim != 3 or image.shape[-1] != 64:
            raise ValueError("unexpected TRTLLM BF16 blocked weight geometry")
        return image, image.shape[1], image.shape[0] * 128
    elif orientation == "rows":
        image = _packed_image(image, layout)
        return image, image.shape[0], image.shape[1]
    elif orientation == "transposed":
        # NVFP4 MegaMoE retains the transpose as a view; recover physical packed rows.
        image = _packed_image(image, layout).transpose(0, 1)
        return image, image.shape[0], image.shape[1]
    elif orientation == "k_major":
        # MXFP8 MegaMoE materializes the transpose; its physical rows are K-major.
        image = _packed_image(image, layout)
        return image, image.shape[1], image.shape[0]
    else:
        raise ValueError(f"unsupported FlashInfer weight orientation: {orientation}")


def expert_scale_target(layer, layout, local_expert, projection, canonical_shape):
    """Existing FP32 API storage for lossless block-128 UE8M0 expansion.

    Preparation translates the canonical exponent XOR to a physical FP32 XOR;
    ordinary byte-layout views cannot express this change of representation.
    """
    stem = _stem(projection)
    target = getattr(layer, f"{stem}_weight_scale_inv")[
        local_expert
    ]  # config-access-exempt: Select the admitted block128 projection scale parameter.
    if target.dtype != torch.float32:
        raise _DtypeMismatch(f"block128 UE8M0 layout requires served=F32 (torch.float32), found {target.dtype}")
    if target.ndim != 2:
        raise ValueError("unexpected block-128 FP32 scale image")
    rows, cols = canonical_shape
    if layout.gated and projection != "down":
        if target.shape[0] % 2:
            raise ValueError("invalid block-128 gate/up scale geometry")
        half_rows = target.shape[0] // 2
        start = half_rows if projection == "up" else 0
        if rows > half_rows:
            raise ValueError("block-128 scale projection exceeds its half")
        target = target[start : start + rows]
    if rows > target.shape[0] or cols > target.shape[1]:
        raise ValueError("block-128 scale projection exceeds live storage")
    return target[:rows, :cols]


def expert_slices(layer, layout, projection, suffix, shape):
    """Canonical projection owned by the ordinary expert loader's TP rank.

    CuTe DSL and MegaMoE partition checkpoint extents before padding. TRTLLM's
    padded loader instead uses its physical intermediate extent as the rank
    stride. Keep that distinction at admission; apply sees only byte strides.
    """
    slices = [[0, extent] for extent in shape]
    if layer.moe_tp_size != 1:
        if projection == "down":
            axis = 1
        elif projection in {"gate", "up"}:
            axis = 0
        else:
            raise ValueError(f"unsupported FlashInfer expert projection: {projection}")
        extent = shape[axis]
        if layer.use_padded_loading:
            divisor = 1
            if suffix in {"weight_scale", "weight_scale_inv"}:
                if layout.format == "fp8-block128-ue8m0":
                    divisor = 128
                elif axis == 1 and layout.format == "nvfp4":
                    divisor = 16
                elif axis == 1 and layout.format in {"mxfp4", "mxfp8"}:
                    divisor = 32
                elif axis == 1:
                    raise ValueError(f"unsupported FlashInfer padded scale format: {layout.format}")
            elif axis == 1 and layout.format in {"nvfp4", "mxfp4"}:
                divisor = 2
            width = (layer.intermediate_size_per_partition + divisor - 1) // divisor
        else:
            width, remainder = divmod(extent, layer.moe_tp_size)
            if remainder:
                raise ValueError("canonical expert extent must divide its TP group")
        start = min(layer.moe_tp_rank * width, extent)
        slices[axis] = [start, min(start + width, extent)]
    return slices


def expert_destinations(layer, layout, local_expert, projection, suffix, canonical_shape, canonical_dtype):
    """Return disjoint canonical rectangles and live bit-preserving views.

    Canonical shapes are the local logical projection, excluding padding.
    Scalar calibration and FP32 scale expansion use separately prepared physical
    XOR masks; they cannot use the canonical mask's byte-layout destinations.
    """
    pair, fmt = layout.pair, _FORMATS[layout.format]
    stem = _stem(projection)
    if projection == "down" or not layout.gated:
        group, half = 0, 0
    else:
        group, half = pair.group, int((projection == "up") != layout.up_first)

    def mapped(row):
        return _row_map(row, group, half, pair.shuffle)

    rows, cols = canonical_shape
    if pair.scale_image == "cutedsl" and suffix == "weight_scale":
        if canonical_dtype != fmt.scale:
            raise _DtypeMismatch(f"checkpoint={canonical_dtype}; NVFP4 block scales require canonical F8_E4M3 bytes")
        images = _cutedsl_scale_images(layer, stem, local_expert, pair.scale_served)
        primary = images[0]
        if primary.ndim != 2 or any(not image.is_contiguous() or image.numel() != primary.numel() for image in images):
            raise ValueError("unexpected CuTe DSL block-scale padding geometry")
        padded_rows, padded_cols = primary.shape
        if padded_rows % 128 or padded_cols % 4 or cols > padded_cols:
            raise ValueError("unexpected CuTe DSL block-scale padding geometry")
        if rows > (padded_rows // 2 if group else padded_rows):
            raise ValueError("canonical scale projection exceeds FlashInfer row storage")
        return _blocked_scale_segments(images, rows, cols, padded_cols, mapped)

    weight, physical_rows, physical_cols = _weight_image(layer, stem, local_expert, layout)
    if suffix == "weight":
        _check_served(weight, fmt.weight_served, f"{layout.format} weight")
    if not weight.is_contiguous():
        raise ValueError("unexpected noncontiguous FlashInfer physical storage")
    half_rows = physical_rows // (2 if layout.gated and projection != "down" else 1)
    if rows > half_rows:
        raise ValueError("canonical projection exceeds FlashInfer row storage")
    if group and half_rows % group:
        raise ValueError("invalid FlashInfer gate/up group alignment")
    if pair.shuffle and physical_rows % 32:
        raise ValueError("TRTLLM weight row shuffle requires a multiple of 32")
    if suffix == "weight":
        if canonical_dtype != fmt.weight:
            raise _DtypeMismatch(
                f"checkpoint={canonical_dtype}; {layout.format} weights require canonical {fmt.weight}, "
                f"served={weight.dtype}"
            )
        # _weight_image admitted this width for the layout's orientation.
        element_size = weight.element_size()
        if cols * element_size > physical_cols:
            raise ValueError("canonical weight exceeds FlashInfer column storage")
        if layout.format == "fp8-block128-ue8m0":
            start = half_rows if layout.gated and projection == "up" else 0
            return (DestinationSegment(((0, rows), (0, cols)), (_byte_view(weight[start : start + rows, :cols]),)),)
        # Physical byte strides of one canonical row, column and column block.
        if pair.weight == "blocked":
            row_stride, block, stride, block_stride = 128, 128, 1, physical_rows * 128
        elif pair.weight == "k_major":
            row_stride, block, stride, block_stride = 1, 1, physical_rows, physical_rows
        elif pair.weight in {"rows", "transposed"}:
            row_stride, block, stride, block_stride = physical_cols, 1, 1, 1
        else:
            raise ValueError(f"unsupported FlashInfer weight orientation: {pair.weight}")

        def row_offset(row):
            return mapped(row) * row_stride

        return _factored_segments([weight], rows, cols, element_size, row_offset, block, stride, block_stride)
    if suffix != "weight_scale":
        raise ValueError(f"unsupported FlashInfer expert suffix: {suffix}")
    if fmt.scale is None:
        raise ValueError(f"unsupported FlashInfer block-scale format: {layout.format}")
    if canonical_dtype != fmt.scale:
        raise _DtypeMismatch(
            f"checkpoint={canonical_dtype}; {layout.format} block scales require canonical {fmt.scale}"
        )
    if physical_cols % fmt.scale_divisor:
        raise ValueError("weight columns do not cover complete scale groups")
    scale_cols = physical_cols // fmt.scale_divisor
    if cols > scale_cols:
        raise ValueError("canonical scales exceed FlashInfer column storage")
    scale = _scale_parameter(layer, stem, layout)[local_expert]
    _check_served(scale, pair.scale_served, f"{layout.format} block scale")
    if pair.scale_image == "deepgemm":
        # DeepGEMM packs four UE8M0 bytes per int32 in MN-major storage;
        # MegaMoE then transposes each [4,32] row group for UTCCP.
        if (
            scale.ndim != 2
            or tuple(scale.shape) != (physical_rows, (scale_cols + 3) // 4)
            or scale.stride() != (1, physical_rows)
            or physical_rows % 128
        ):
            raise ValueError("unexpected DeepGEMM packed UE8M0 scale geometry")

        def packed_row_offset(row):
            row = mapped(row)
            return (row // 128 * 128 + row % 32 * 4 + row % 128 // 32) * 4

        return _factored_segments([scale.transpose(0, 1)], rows, cols, 1, packed_row_offset, 4, 1, physical_rows * 4)
    elif pair.scale_image == "blocked":
        padded_cols = (scale_cols + 3) // 4 * 4
        if not scale.is_contiguous() or scale.numel() != (physical_rows + 127) // 128 * 128 * padded_cols:
            raise ValueError("unexpected FlashInfer block-scale padding geometry")
        return _blocked_scale_segments([scale], rows, cols, padded_cols, mapped)
    else:
        raise ValueError(f"unsupported FlashInfer block-scale image: {pair.scale_image}")


def consumer_objects(layer, layout):
    """Snapshot selected consumer objects and non-parameter tensor identities."""
    method = layer.quant_method
    pair = layout.pair
    objects = [
        method,
        method.runner,
        layer.moe_runner_config,
        layer.moe_runner_config.is_gated,
        layer.w13_weight,
        layer.w2_weight,
    ]
    if pair.scale is not None:
        objects.extend(
            (
                _scale_parameter(layer, "w13", layout),
                _scale_parameter(layer, "w2", layout),
            )
        )
    if layout.format == "nvfp4":
        objects.extend(
            (
                layer.g1_alphas,
                layer.g1_alphas_up,
                layer.g2_alphas,
                layer.w13_input_scale_quant,
                layer.w2_input_scale_quant,
            )
        )
    if layout.backend == "cutedsl":
        objects.extend(
            (
                layer._cutedsl_wrapper,
                layer._cutedsl_wrapper.quant_mode,
                layer._cutedsl_scales,
                *layer._cutedsl_scales,
                layer._cutedsl_input_scale,
                layer.w13_blockscale_swizzled,
                layer.w2_blockscale_swizzled,
            )
        )
        # MMA images are created by the standard v2 path when required.
        objects.extend(
            (
                getattr(
                    layer, "w13_blockscale_mma", None
                ),  # config-access-exempt: CuTe may retain a second gate/up MMA scale image.
                getattr(
                    layer, "w2_blockscale_mma", None
                ),  # config-access-exempt: CuTe may retain a second down-projection MMA scale image.
            )
        )
    elif layout.backend == "megamoe":
        mega = layer._flashinfer_megamoe_layer
        if mega is None:
            raise ValueError("FlashInfer MegaMoE must be warmed before admission")
        objects.extend(
            (
                mega,
                mega._megakernel_config,
                mega._kernel,
                mega._transformed,
                layer._flashinfer_megamoe_forward,
            )
        )
        for weights, scales in mega._transformed:
            objects.extend((weights, scales))
        if layout.format == "nvfp4" and layout.quant_mode == "w4a4":
            config = mega._megakernel_config
            objects.extend(
                (
                    config.input_norm_const,
                    layer._flashinfer_megamoe_input_norm_const,
                )
            )
    elif layout.backend == "trtllm":
        if layout.format == "nvfp4":
            # TRTLLM activation-dependent optional tensors; absence is itself captured.
            objects.extend(
                getattr(
                    layer, name, None
                )  # config-access-exempt: Snapshot optional backend-owned consumer fields for identity checks.
                for name in (
                    "gemm1_alpha",
                    "gemm1_beta",
                    "gemm1_clamp_limit",
                    "g1_scale_c",
                )
            )
    else:
        raise ValueError(f"unsupported FlashInfer consumer backend: {layout.backend}")
    objects.extend(_consumer_field(layer, name) for name in pair.consumers)
    objects.extend(_consumer_field(method, name) for name in pair.method_consumers)
    return tuple(objects)


def _normal_scale_helpers():
    # Keep dependencies lazy: byte-layout planning need not load inference
    # kernels merely to describe an update.
    from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas, _input_scale_to_local_experts

    return _compute_gemm1_alphas, _input_scale_to_local_experts


def _cutedsl_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import resolve_cutedsl_standard_scales

    return resolve_cutedsl_standard_scales


def _trtllm_scale_helper():
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c

    return _compute_g1_scale_c


def nvfp4_derived_update(layer, layout):
    """Describe the selected FlashInfer consumer's small NVFP4 scale updates.

    This mirrors the ordinary loader's scale policy and reuses its pure helpers.
    It neither invokes weight postprocessing nor replaces captured tensors.
    MegaMOE's input normalization is a captured scalar, so a change to that
    scalar is rejected during preparation, before any live weight is modified.
    """
    # identify_layout read the backend and quant mode from the warmed consumer;
    # finish() rechecks that consumer's identity before building this update.
    backend = layout.backend
    if backend == "trtllm":
        trtllm, cute, mega = True, False, False
    elif backend == "cutedsl":
        trtllm, cute, mega = False, True, False
    elif backend == "megamoe":
        trtllm, cute, mega = False, False, True
    else:
        raise ValueError(f"unsupported NVFP4 derived consumer: {backend}")
    if layout.quant_mode == "w4a4":
        w4a16 = False
    elif layout.quant_mode == "w4a16" and not trtllm:
        w4a16 = True
    else:
        raise ValueError(f"unsupported NVFP4 derived quant mode for {backend}: {layout.quant_mode!r}")
    per_token = bool(layer.quant_config.use_per_token_activation)
    neutral_input = w4a16 or per_token
    config = layer.moe_runner_config
    gated, activation = config.is_gated, config.activation
    local_experts, experts, ep_rank = (
        layer.num_local_experts,
        layer.num_experts,
        layer.moe_ep_rank,
    )
    scalar_input13 = layer.w13_input_scale_quant.ndim == 0
    source_names = ["w13_weight_scale_2", "w2_weight_scale_2"]
    if not neutral_input:
        source_names.extend(("w13_input_scale", "w2_input_scale"))
    sources = tuple(
        getattr(layer, name) for name in source_names
    )  # config-access-exempt: Resolve admitted named calibration buffers on the inference layer.
    destinations = [layer.g1_alphas, layer.g1_alphas_up, layer.g2_alphas]

    # W4A16 already has neutral activation constants. Retain its direct-copy
    # refresh rather than recomputing those constants every update.
    if not w4a16:
        destinations.extend((layer.w13_input_scale_quant, layer.w2_input_scale_quant))
    if trtllm:
        destinations.append(layer.g1_scale_c)
    extra_names = []
    if trtllm and gated:
        if (
            getattr(layer, "gemm1_clamp_limit", None) is not None
        ):  # config-access-exempt: The selected TRTLLM inference path may retain a clamp image.
            extra_names.append("gemm1_clamp_limit")
        if (
            getattr(layer, "gemm1_beta", None) is not None
        ):  # config-access-exempt: The selected TRTLLM inference path may retain a beta image.
            extra_names.append("gemm1_beta")
        destinations.extend(
            getattr(layer, name) for name in extra_names
        )  # config-access-exempt: Resolve admitted named calibration buffers on the inference layer.
    if cute:
        scales = layer._cutedsl_scales
        input_scale = layer._cutedsl_input_scale
        if not isinstance(scales, tuple) or len(scales) != 3:
            raise ValueError("CuTe DSL scale metadata must be warmed before admission")
        destinations.extend((scales[0], scales[2]) if w4a16 else (*scales, input_scale))
    mega_scalar = None
    if mega and not w4a16:
        if per_token:
            raise ValueError("FlashInfer MegaMOE does not support per-token NVFP4")
        consumer = layer._flashinfer_megamoe_layer
        cached = layer._flashinfer_megamoe_input_norm_const
        if consumer is None or cached is None:
            raise ValueError("FlashInfer MegaMOE must be warmed before delta admission")
        mega_scalar = float(consumer._megakernel_config.input_norm_const)
        if float(cached) != mega_scalar:
            raise ValueError("FlashInfer MegaMOE input normalization cache disagrees")
    for tensor in (*sources, *destinations):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("NVFP4 derived calibration requires FP32 scalar tensors")

    # Resolve pure inference helpers and configuration once at admission.
    if not w4a16:
        gemm1_alphas, local_input = _normal_scale_helpers()
    if cute and not w4a16:
        cutedsl_values = _cutedsl_scale_helper()
        fixed_input_scale = layer._cutedsl_input_scale
    if trtllm:
        trtllm_scale_c = _trtllm_scale_helper()
    clamp = config.gemm1_clamp_limit or config.swiglu_limit
    if "gemm1_clamp_limit" in extra_names and (clamp is None or activation == "situ"):
        raise ValueError("unsupported TRTLLM activation clamp metadata")
    tied_clamp = "gemm1_clamp_limit" in extra_names
    tied_clamp_error = (
        "GPU delta requires tied TRTLLM gate/up scales with activation clamping; "
        "unequal scales require block-scale requantization"
    )
    if tied_clamp:
        # The ordinary loader otherwise rescales one half of the checkpoint
        # block scales. Those bytes cannot be updated by canonical XOR masks.
        weight13 = layer.w13_weight_scale_2
        _validate_checks((((weight13[:, 0] == weight13[:, 1]).all(), tied_clamp_error),))
    situ_beta = config.gemm1_clamp_limit
    extra_outputs = []
    for name in extra_names:
        if name == "gemm1_clamp_limit":
            extra_outputs.append(lambda shadow: (clamp / shadow.g1_alphas).to(torch.float32))
        elif name == "gemm1_beta":
            # SGLang's get_activation_type admits silu, gelu and situ for gated TRTLLM experts.
            if activation == "situ":
                extra_outputs.append(lambda shadow: torch.full_like(shadow.g1_alphas, situ_beta))
            elif activation in {"silu", "gelu"}:
                extra_outputs.append(lambda shadow: (1 / shadow.g1_alphas_up).to(torch.float32))
            else:
                raise ValueError(f"unsupported TRTLLM gemm1_beta activation: {activation}")
        else:
            raise ValueError(f"unsupported TRTLLM GEMM1 image: {name}")
    neutral13, neutral2 = layer.w13_input_scale_quant, layer.w2_input_scale_quant
    if per_token and not w4a16:
        compute_device = next((source.device for source in sources if source.is_cuda), sources[0].device)
        neutral13, neutral2 = neutral13.to(compute_device), neutral2.to(compute_device)
        if cute:
            fixed_input_scale = fixed_input_scale.to(compute_device)

    def gate_up(weight13):
        if gated and weight13.ndim == 2 and weight13.shape[1] >= 2:
            return weight13[:, 0], weight13[:, 1]
        gate = weight13.reshape(weight13.shape[0])
        return gate, gate

    def compute(values):
        weight13, weight2 = values[:2]
        if w4a16:
            gate, up = gate_up(weight13)
            return (gate, up, weight2, gate, weight2) if cute else (gate, up, weight2)

        shadow = SimpleNamespace(
            num_local_experts=local_experts,
            num_experts=experts,
            moe_ep_rank=ep_rank,
            w13_weight_scale_2=weight13,
            w2_weight_scale_2=weight2,
        )
        if neutral_input:
            # Per-token consumers already have neutral activation constants.
            # Reuse those captured tensors rather than allocating ones and
            # recomputing their reciprocals for every update.
            shadow.g1_alphas, shadow.g1_alphas_up = gate_up(weight13)
            shadow.g2_alphas = weight2
            shadow.w13_input_scale_quant = neutral13
            shadow.w2_input_scale_quant = neutral2
        else:
            raw13, raw2 = values[2:]
            input13 = raw13.max().to(torch.float32)
            if cute or mega:
                input2 = local_input(raw2, local_experts, experts, ep_rank)
                if cute:
                    input13 = input13.expand(local_experts)
                    if scalar_input13:
                        input13 = input13[0]
            else:
                input2 = raw2.max().to(torch.float32)
            shadow.g1_alphas, shadow.g1_alphas_up = gemm1_alphas(weight13, input13, gated)
            shadow.g2_alphas = (input2 * weight2).to(torch.float32)
            shadow.w13_input_scale_quant = (1 / input13).to(torch.float32)
            shadow.w2_input_scale_quant = (1 / input2).to(torch.float32)
        outputs = [
            shadow.g1_alphas,
            shadow.g1_alphas_up,
            shadow.g2_alphas,
            shadow.w13_input_scale_quant,
            shadow.w2_input_scale_quant,
        ]
        if trtllm:
            outputs.append(
                trtllm_scale_c(
                    shadow.w2_input_scale_quant,
                    shadow.g1_alphas,
                    shadow.g1_alphas_up,
                    gated,
                    activation,
                )
            )
        outputs.extend(output(shadow) for output in extra_outputs)
        if cute:
            w1, fc2_input, w2, input_scale = cutedsl_values(shadow)
            if per_token:
                input_scale = fixed_input_scale
            outputs.extend((w1, fc2_input, w2, input_scale))
        return tuple(outputs)

    def checks(outputs):
        predicates = []
        if mega and not w4a16:
            predicates.append(
                (
                    # Match the former Python-float comparison exactly, including a
                    # captured scalar that is not representable as an FP32 value.
                    outputs[3].max().to(torch.float64) == mega_scalar,
                    "FlashInfer MegaMOE input normalization changed; CUDA graph recapture is required",
                )
            )
        if mega and gated:
            predicates.append(
                (
                    torch.isclose(outputs[0], outputs[1]).all(),
                    "FlashInfer MegaMOE requires matching gate/up FC1 alphas",
                )
            )
        if tied_clamp:
            predicates.append(((outputs[0] == outputs[1]).all(), tied_clamp_error))
        return tuple(predicates)

    return DerivedUpdate(sources, tuple(destinations), compute, checks if mega or tied_clamp else None)


def expert_derived_updates(layer, layout):
    """The derived calibration updates of one identified layer, by canonical format."""
    if layout.format == "nvfp4":
        return (nvfp4_derived_update(layer, layout),)
    elif layout.format in {"bf16", "mxfp4", "mxfp8", "fp8-block128-ue8m0"}:
        # These formats carry no per-tensor scale, so no consumer value derives from one.
        return ()
    else:
        raise ValueError(f"unsupported GPU delta expert format: {layout.format}")
