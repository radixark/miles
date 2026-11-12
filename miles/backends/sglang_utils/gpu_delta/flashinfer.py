"""FlashInfer expert layouts expressed as views of live consumer bytes.

The apply kernel knows only affine byte coordinates. These admission-time
helpers factor the inference backend's permutations into those coordinates.
They never transform a delta, requantize a weight, or allocate a weight image.
A segment selects a rectangle of one canonical projection; its destinations
have the same flattened byte order, and may cover several independent images.
Padding is deliberately outside every destination.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import torch


class _DtypeMismatch(ValueError):
    """A logical or physical dtype mismatch, collected by layout admission."""


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
    consumers: tuple[str, ...] = ()  # required TRTLLM consumer fields


_PACKED = (torch.uint8, torch.int8)
# SGLang's MegaMoE NVFP4 path serves FlashInfer's packed weights as float4_e2m1fn_x2,
# the checkpoint's two-E2M1-per-byte U8 bytes unchanged.
_NVFP4_PACKED = _PACKED + (torch.float4_e2m1fn_x2,)
_E4M3 = (torch.float8_e4m3fn, torch.uint8, torch.int8)
_UE8M0 = (torch.uint8, torch.int8, torch.float8_e8m0fnu)
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
    # Standard W4A4/W4A16: Up64, Gate64, Up64, Gate64, ...
    ("cutedsl", "nvfp4"): _Pair("weight_scale", 64, False, "rows", "cutedsl", _E4M3),
    # Gate16, Up16, Gate16, Up16, ...
    ("megamoe", "nvfp4"): _Pair("weight_scale", 16, False, "transposed", "blocked", _E4M3),
    ("megamoe", "mxfp8"): _Pair("weight_scale_inv", 32, False, "k_major", "blocked", _UE8M0),
    # Retain checkpoint scales separately; target the shuffled consumer.
    ("megamoe", "mxfp4"): _Pair("weight_scale_inv_shuffled", 8, False, "rows", "deepgemm", (torch.int32,)),
}
# Mxfp4MoEMethod's TRTLLM kernel reads the loader's scale name and owns no clamp tensor.
_MXFP4_MOE_METHOD = replace(_PAIRS["trtllm", "mxfp4"], scale="weight_scale", consumers=())


@dataclass(frozen=True)
class FlashInferLayout:
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


@dataclass(frozen=True)
class DestinationSegment:
    # Coordinates are canonical ELEMENTS, not bytes. Only the returned views
    # use byte coordinates, so BF16 source selections retain their element size.
    slices: tuple[tuple[int, int], ...]
    destinations: tuple[torch.Tensor, ...]
    source_shape: tuple[int, ...] | None = None


def _backend(backend):
    backend = backend.value if isinstance(backend, Enum) else backend
    if backend in {
        "trtllm",
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
    }:
        resolved = "trtllm"
    elif backend in {"cutedsl", "flashinfer_cutedsl"}:
        resolved = "cutedsl"
    elif backend in {"megamoe", "flashinfer_megamoe"}:
        resolved = "megamoe"
    else:
        raise ValueError(f"unsupported FlashInfer expert backend: {backend}")
    return resolved


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
            layout = FlashInferLayout("nvfp4", backend, gated, mode, up_first=True)
        elif backend == "megamoe":
            config = layer._flashinfer_megamoe_layer._megakernel_config
            config_name = type(config).__name__
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
        layout = FlashInferLayout("mxfp4", backend, gated, up_first=True)
    elif name == "Mxfp4MoEMethod":
        if backend != "trtllm" or method._fi_kernel != "trtllm_sm100":
            raise ValueError("unsupported native MXFP4 FlashInfer expert path")
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
            layout = FlashInferLayout("mxfp4", backend, gated)
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


def _byte_view(tensor):
    # A strided outer dimension is fine, but dtype reinterpretation requires
    # contiguous elements within each row. Never silently clone a destination.
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.stride(-1) != 1:
        raise ValueError("delta destination requires a contiguous innermost dimension")
    return tensor.detach().view(torch.uint8)


def _check_served(image, allowed, role):
    if image.dtype not in allowed:
        raise _DtypeMismatch(f"{role} served={image.dtype}, allowed_served={[str(dtype) for dtype in allowed]}")


def _same_image(a, b):
    return (
        a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and a.storage_offset() * a.element_size() == b.storage_offset() * b.element_size()
        and a.numel() * a.element_size() == b.numel() * b.element_size()
    )


def _unique_images(images):
    unique = []
    for image in images:
        if not any(_same_image(image, current) for current in unique):
            unique.append(image)
    return unique


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


def _row_pieces(rows):
    """One full-128-row piece and at most seven power-of-two tail pieces."""
    start = 0
    for unit in (128, 64, 32, 16, 8, 4, 2, 1):
        count = (rows - start) // unit
        if count:
            yield start, start + count * unit, unit
            start += count * unit


def _row_map(row, group, half, shuffle):
    # These are the backend's fixed bit permutations, not a per-row index map:
    # gated groups, then TRTLLM's 32-row transpose (4 x 8 -> 8 x 4).
    if group:
        row = row // group * (2 * group) + half * group + row % group
    if shuffle:
        row = row // 32 * 32 + row % 4 * 8 + row % 32 // 4
    return row


def _factored_segments(
    images,
    rows,
    cols,
    element_size,
    row_offset,
    column_block,
    column_stride,
    block_stride,
):
    """Factor fixed row-bit permutations into affine axes, excluding padding.

    A 128-row bulk plus binary tail pieces gives at most eight row rectangles.
    Columns need at most a full-block rectangle and one tail. No tensor data,
    index tensor, or weight-sized intermediate is constructed.
    """
    byte_cols = cols * element_size
    full = byte_cols // column_block * column_block
    columns = [(0, full)] if full else []
    if full < byte_cols:
        columns.append((full, byte_cols))
    images = tuple(_byte_view(image) for image in _unique_images(images))
    result = []
    for r0, r1, unit in _row_pieces(rows):
        bits = [1 << bit for bit in reversed(range(unit.bit_length() - 1))]
        shape = ((r1 - r0) // unit, *(2 for _ in bits))
        origin = row_offset(r0)
        strides = (
            row_offset(r0 + unit) - origin,
            *(row_offset(r0 + bit) - origin for bit in bits),
        )
        for c0, c1 in columns:
            if (c1 - c0) % column_block == 0:
                cshape = ((c1 - c0) // column_block, column_block)
                cstrides = (block_stride, column_stride)
            else:
                cshape, cstrides = (c1 - c0,), (column_stride,)
            offset = origin + c0 // column_block * block_stride
            destinations = tuple(
                image.as_strided(
                    (*shape, *cshape),
                    (*strides, *cstrides),
                    image.storage_offset() + offset,
                )
                for image in images
            )
            result.append(DestinationSegment(((r0, r1), (c0 // element_size, c1 // element_size)), destinations))
    return tuple(result)


def _blocked_scale_segments(images, rows, cols, padded_cols, mapped):
    """FlashInfer's to_blocked scales: [128, 4] tiles with transposed [4, 32] row factors."""

    def row_offset(row):
        row = mapped(row)
        return row // 128 * padded_cols * 128 + row % 32 * 16 + row % 128 // 32 * 4

    return _factored_segments(images, rows, cols, 1, row_offset, 4, 1, 512)


def _scale_parameter(layer, stem, layout):
    return getattr(
        layer, f"{stem}_{layout.pair.scale}"
    )  # config-access-exempt: Resolve the admitted backend-specific expert scale field.


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
    elif layout.format == "nvfp4":
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
    objects.extend(
        getattr(layer, name) for name in pair.consumers
    )  # config-access-exempt: Snapshot the admitted pair's required consumer fields.
    return tuple(objects)
