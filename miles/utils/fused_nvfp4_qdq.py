"""Fused CuTe DSL NVFP4 quantize-dequantize for fake QAT.

The kernel keeps the E4M3 block scale and packed E2M1 values in registers and
writes only the dequantized BF16/FP16 result. Its arithmetic order mirrors
Transformer Engine's 1D, 1x16, per-tensor NVFP4 implementation. The vectorized
load, FP4 conversion, and Four Over Six structure are adapted from FlashInfer's
CuTe DSL NVFP4 quantizer.

Supported contract:

* contiguous rank-2 BF16 or FP16 input on SM10x;
* 1x16 block scaling and a caller-provided FP32 per-tensor amax;
* round-to-nearest quantization with ordinary quant fast math disabled;
* standard NVFP4, plus the full Four Over Six MAE/MSE, E4M3-max 256/448,
  and exact/FP16-error matrix;
* no stochastic rounding, RHT, 2D quantization, transpose, or row scaling.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

_FP32_MAX = 3.4028234663852886e38
_FP4_BLOCK_SIZE = 16
_STANDARD_THREADS = 256
# Preserve the 4-CTA compile launch bound while using a deeper runtime grid to
# reduce each thread's grid-stride work on model-sized tensors.
_STANDARD_MIN_BLOCKS_PER_SM = 4
_STANDARD_GRID_BLOCKS_PER_SM = 24
_4OVER6_THREADS = 128
# The largest specialization uses 56 registers/thread, so 8x128 threads stays
# below the SM10x 64K-register budget without spills while doubling active CTAs.
_4OVER6_BLOCKS_PER_SM = 8
_INT32_MAX = 2**31 - 1


class NVFP4QDQErrorMode(IntEnum):
    """Four Over Six candidate error metric."""

    MAE = 0
    MSE = 1


@dataclass(frozen=True)
class NVFP4QDQConfig:
    """Compile-time numerical configuration for fused NVFP4 QDQ.

    ``error_use_fast_math`` retains Transformer Engine's public knob name, but
    under the TE 2.17 contract it selects FP16-rounded candidate-error math.
    """

    use_4over6: bool = False
    e4m3_max: int = 448
    error_mode: NVFP4QDQErrorMode = NVFP4QDQErrorMode.MAE
    error_use_fast_math: bool = False

    def __post_init__(self) -> None:
        if self.e4m3_max not in (256, 448):
            raise ValueError(f"NVFP4 E4M3 max must be 256 or 448, got {self.e4m3_max}.")
        if not self.use_4over6 and self.e4m3_max != 448:
            raise ValueError("E4M3 max 256 is only supported by Four Over Six.")
        if not self.use_4over6 and self.error_use_fast_math:
            raise ValueError("The FP16 error contract only applies to Four Over Six.")


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip()
    if value == "1":
        return True
    if value in ("0", ""):
        return False
    raise ValueError(f"{name} must be 0 or 1, got {value!r}.")


def _env_applies_to_weights(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in ("none", "weights", "activations", "all"):
        raise ValueError(f"{name} must be one of none, weights, activations, or all; got {value!r}.")
    return value in ("weights", "all")


def current_nvfp4_qdq_config() -> NVFP4QDQConfig:
    """Resolve the weight QDQ contract from Transformer Engine environment variables."""
    if _env_flag("NVTE_USE_FAST_MATH"):
        raise ValueError(
            "Fused NVFP4 QDQ requires NVTE_USE_FAST_MATH=0; ordinary quant fast math "
            "is outside its numerical contract."
        )

    use_4over6 = _env_applies_to_weights("NVTE_NVFP4_4OVER6", "none")
    use_e4m3_256 = _env_applies_to_weights("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    error_mode_name = os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()
    try:
        error_mode = NVFP4QDQErrorMode[error_mode_name]
    except KeyError as exc:
        raise ValueError(f"NVTE_NVFP4_4OVER6_ERR_MODE must be MAE or MSE, got {error_mode_name!r}.") from exc

    error_use_fast_math = _env_flag("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")
    return NVFP4QDQConfig(
        use_4over6=use_4over6,
        e4m3_max=256 if use_4over6 and use_e4m3_256 else 448,
        error_mode=error_mode,
        # Despite the legacy variable name, this selects TE's FP16 candidate-error contract.
        error_use_fast_math=(use_4over6 and error_use_fast_math),
    )


@dsl_user_op
def _get_ptr(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    elem_ptr = tensor.iterator + offset
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _load_v4_u32(base_ptr: Int64, *, loc=None, ip=None) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _store_v4_u32(base_ptr: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fadd_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fsub_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _scale_f32x2_rn(value0: Float32, value1: Float32, scale: Float32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Multiply two FP32 values by one common scale with packed RN arithmetic."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Float32(value0).ir_value(loc=loc, ip=ip),
            Float32(value1).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b64 values, scales;
            mov.b64 values, {$2, $3};
            mov.b64 scales, {$4, $4};
            mul.f32x2 values, values, scales;
            mov.b64 {$0, $1}, values;
        }
        """,
        "=f,=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _square_f32x2_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Square two FP32 values with the packed RN instruction used by TE."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b64 values;
            mov.b64 values, {$2, $3};
            mul.f32x2 values, values, values;
            mov.b64 {$0, $1}, values;
        }
        """,
        "=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _fdiv_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmin(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fabs(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_abs(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bfloat2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_max_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bfloat2_max_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_to_f32x2(x: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bfloat2_to_f32x2(x: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e4m3_to_f32(a: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2;
                .reg .b16 lo, hi;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                mov.b32 {lo, hi}, h2;
                cvt.f32.f16 $0, lo;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e2m1x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e2m1x2_to_f32x2(a: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            .reg .b32 h2;
            .reg .b16 lo, hi;
            .reg .b32 code_lo, code_hi, bits_lo, bits_hi;
            .reg .f32 f_lo, f_hi;
            .reg .pred negzero_lo, negzero_hi;

            mov.b32 {byte0, byte1, byte2, byte3}, $2;
            cvt.rn.f16x2.e2m1x2 h2, byte0;
            mov.b32 {lo, hi}, h2;
            cvt.f32.f16 f_lo, lo;
            cvt.f32.f16 f_hi, hi;
            mov.b32 bits_lo, f_lo;
            mov.b32 bits_hi, f_hi;
            and.b32 code_lo, $2, 0xF;
            shr.u32 code_hi, $2, 4;
            and.b32 code_hi, code_hi, 0xF;
            setp.eq.u32 negzero_lo, code_lo, 0x8;
            setp.eq.u32 negzero_hi, code_hi, 0x8;
            selp.u32 bits_lo, 0x80000000, bits_lo, negzero_lo;
            selp.u32 bits_hi, 0x80000000, bits_hi, negzero_hi;
            mov.b32 $0, bits_lo;
            mov.b32 $1, bits_hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _scaled_e2m1x2_e4m3_to_f32x2(packed: Uint32, scale: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip), Uint32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 fp8_pair, scale_h, unused_h, lo, hi;
            .reg .b32 q_h2, scale_h2, product_h2;
            mov.b32 {b0, b1, b2, b3}, $2;
            cvt.rn.f16x2.e2m1x2 q_h2, b0;
            cvt.u16.u32 fp8_pair, $3;
            cvt.rn.f16x2.e4m3x2 scale_h2, fp8_pair;
            mov.b32 {scale_h, unused_h}, scale_h2;
            mov.b32 scale_h2, {scale_h, scale_h};
            mul.rn.f16x2 product_h2, q_h2, scale_h2;
            mov.b32 {lo, hi}, product_h2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_f32x2_to_half2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.f16.f32 lo, $1;
                cvt.rn.f16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_f32x2_to_bfloat2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.bf16.f32 lo, $1;
                cvt.rn.bf16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _normal_block_scale(block_amax: Float32, global_encode_scale: Float32, *, loc=None, ip=None) -> Float32:
    """TE's intentionally associated ``amax * (S_enc * (1/6))`` expression."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_encode_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred zero;
                .reg .f32 scale_mul, result;
                setp.eq.f32 zero, $1, 0f00000000;
                mul.rn.f32 scale_mul, $2, 0f3E2AAAAB;
                mul.rn.f32 result, $1, scale_mul;
                selp.f32 $0, 0f00000000, result, zero;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _input_values(words: tuple, is_bfloat16: bool) -> tuple:
    values = ()
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr(is_bfloat16):
            lo, hi = _bfloat2_to_f32x2(words[i])
        else:
            lo, hi = _half2_to_f32x2(words[i])
        values = values + (lo, hi)
    return values


@cute.jit
def _block_amax(words: tuple, is_bfloat16: bool) -> Float32:
    maxima = ()
    for i in cutlass.range_constexpr(8):
        value = _half2_abs(words[i])
        maxima = maxima + (value,)

    if cutlass.const_expr(is_bfloat16):
        max01 = _bfloat2_max(maxima[0], maxima[1])
        max23 = _bfloat2_max(maxima[2], maxima[3])
        max45 = _bfloat2_max(maxima[4], maxima[5])
        max67 = _bfloat2_max(maxima[6], maxima[7])
        max_value = _bfloat2_max(_bfloat2_max(max01, max23), _bfloat2_max(max45, max67))
        return _bfloat2_max_to_f32(max_value)

    max01 = _half2_max(maxima[0], maxima[1])
    max23 = _half2_max(maxima[2], maxima[3])
    max45 = _half2_max(maxima[4], maxima[5])
    max67 = _half2_max(maxima[6], maxima[7])
    max_value = _half2_max(_half2_max(max01, max23), _half2_max(max45, max67))
    return _half2_max_to_f32(max_value)


@cute.jit
def _scale_pack_e2m1(values: tuple, scale: Float32) -> tuple[Uint32, Uint32]:
    """Scale and immediately pack each eight-value half to limit live FP32 state."""
    scaled_lo = ()
    for pair_idx in cutlass.range_constexpr(4):
        value0, value1 = _scale_f32x2_rn(values[2 * pair_idx], values[2 * pair_idx + 1], scale)
        scaled_lo = scaled_lo + (value0, value1)
    lo = _cvt_e2m1x8(
        scaled_lo[0],
        scaled_lo[1],
        scaled_lo[2],
        scaled_lo[3],
        scaled_lo[4],
        scaled_lo[5],
        scaled_lo[6],
        scaled_lo[7],
    )

    scaled_hi = ()
    for pair_idx in cutlass.range_constexpr(4, 8):
        value0, value1 = _scale_f32x2_rn(values[2 * pair_idx], values[2 * pair_idx + 1], scale)
        scaled_hi = scaled_hi + (value0, value1)
    hi = _cvt_e2m1x8(
        scaled_hi[0],
        scaled_hi[1],
        scaled_hi[2],
        scaled_hi[3],
        scaled_hi[4],
        scaled_hi[5],
        scaled_hi[6],
        scaled_hi[7],
    )
    return lo, hi


@cute.jit
def _scale_pack_input_words(words: tuple, scale: Float32, is_bfloat16: bool) -> tuple[Uint32, Uint32]:
    """Standard-path input conversion with at most eight scaled FP32 values live."""
    scaled_lo = ()
    for i in cutlass.range_constexpr(4):
        if cutlass.const_expr(is_bfloat16):
            value0, value1 = _bfloat2_to_f32x2(words[i])
        else:
            value0, value1 = _half2_to_f32x2(words[i])
        value0, value1 = _scale_f32x2_rn(value0, value1, scale)
        scaled_lo = scaled_lo + (value0, value1)
    lo = _cvt_e2m1x8(
        scaled_lo[0],
        scaled_lo[1],
        scaled_lo[2],
        scaled_lo[3],
        scaled_lo[4],
        scaled_lo[5],
        scaled_lo[6],
        scaled_lo[7],
    )

    scaled_hi = ()
    for i in cutlass.range_constexpr(4, 8):
        if cutlass.const_expr(is_bfloat16):
            value0, value1 = _bfloat2_to_f32x2(words[i])
        else:
            value0, value1 = _half2_to_f32x2(words[i])
        value0, value1 = _scale_f32x2_rn(value0, value1, scale)
        scaled_hi = scaled_hi + (value0, value1)
    hi = _cvt_e2m1x8(
        scaled_hi[0],
        scaled_hi[1],
        scaled_hi[2],
        scaled_hi[3],
        scaled_hi[4],
        scaled_hi[5],
        scaled_hi[6],
        scaled_hi[7],
    )
    return lo, hi


@cute.jit
def _global_encode_scale(amax: Float32, e4m3_max: int) -> Float32:
    scale = _fdiv_rn(Float32(float(e4m3_max * 6)), amax)
    scale = _fmin(scale, Float32(_FP32_MAX))
    if amax == Float32(0.0):
        scale = Float32(1.0)
    if scale == Float32(0.0):
        scale = Float32(1.0)
    return scale


@cute.jit
def _candidate_inverse_scale(scale_f32: Float32, global_decode_scale: Float32) -> Float32:
    product = _fmul_rn(scale_f32, global_decode_scale)
    return _fmin(_fdiv_rn(Float32(1.0), product), Float32(_FP32_MAX))


@cute.jit
def _standard_quantize(
    words: tuple,
    block_amax: Float32,
    global_encode_scale: Float32,
    global_decode_scale: Float32,
    is_bfloat16: bool,
) -> tuple[Uint32, Uint32, Uint32]:
    scale_high_precision = _normal_block_scale(block_amax, global_encode_scale)
    scale = _cvt_f32_to_e4m3(scale_high_precision)
    scale_f32 = _cvt_e4m3_to_f32(scale)
    inverse = _candidate_inverse_scale(scale_f32, global_decode_scale)
    lo, hi = _scale_pack_input_words(words, inverse, is_bfloat16)
    return scale, lo, hi


@cute.jit
def _candidate_error(
    original: tuple,
    lo: Uint32,
    hi: Uint32,
    scale: Uint32,
    global_amax: Float32,
    global_encode_scale: Float32,
    config: NVFP4QDQConfig,
) -> Float32:
    error = Float32(0.0)
    if cutlass.const_expr(config.error_use_fast_math):
        for pair_idx in cutlass.range_constexpr(8):
            if cutlass.const_expr(pair_idx < 4):
                packed_pair = lo >> Uint32(8 * pair_idx)
            else:
                packed_pair = hi >> Uint32(8 * (pair_idx - 4))
            candidate0, candidate1 = _scaled_e2m1x2_e4m3_to_f32x2(packed_pair, scale)
            original0, original1 = _scale_f32x2_rn(
                original[2 * pair_idx], original[2 * pair_idx + 1], global_encode_scale
            )
            diff0 = _fsub_rn(candidate0, original0)
            diff1 = _fsub_rn(candidate1, original1)
            if cutlass.const_expr(config.error_mode == NVFP4QDQErrorMode.MSE):
                term0, term1 = _square_f32x2_rn(diff0, diff1)
            else:
                term0 = _fabs(diff0)
                term1 = _fabs(diff1)
            error = _fadd_rn(error, term0)
            error = _fadd_rn(error, term1)
        return error

    denominator = Float32(float(6 * config.e4m3_max))
    scale_f32 = _cvt_e4m3_to_f32(scale)
    for pair_idx in cutlass.range_constexpr(8):
        if cutlass.const_expr(pair_idx < 4):
            packed_pair = lo >> Uint32(8 * pair_idx)
        else:
            packed_pair = hi >> Uint32(8 * (pair_idx - 4))
        candidate0, candidate1 = _e2m1x2_to_f32x2(packed_pair)

        dequant0 = _fmul_rn(candidate0, scale_f32)
        dequant0 = _fmul_rn(dequant0, global_amax)
        dequant0 = _fdiv_rn(dequant0, denominator)
        diff0 = _fsub_rn(dequant0, original[2 * pair_idx])
        if cutlass.const_expr(config.error_mode == NVFP4QDQErrorMode.MSE):
            term0 = _fmul_rn(diff0, diff0)
        else:
            term0 = _fabs(diff0)
        error = _fadd_rn(error, term0)

        dequant1 = _fmul_rn(candidate1, scale_f32)
        dequant1 = _fmul_rn(dequant1, global_amax)
        dequant1 = _fdiv_rn(dequant1, denominator)
        diff1 = _fsub_rn(dequant1, original[2 * pair_idx + 1])
        if cutlass.const_expr(config.error_mode == NVFP4QDQErrorMode.MSE):
            term1 = _fmul_rn(diff1, diff1)
        else:
            term1 = _fabs(diff1)
        error = _fadd_rn(error, term1)
    return error


@cute.jit
def _four_over_six_quantize(
    values: tuple,
    block_amax: Float32,
    global_amax: Float32,
    global_encode_scale: Float32,
    global_decode_scale: Float32,
    config: NVFP4QDQConfig,
) -> tuple[Uint32, Uint32, Uint32]:
    # TE intentionally associates this differently from standard NVFP4. Keep
    # the zero-amax path in the same expression stream as TE as well: packing
    # the candidates is what preserves negative-zero E2M1 sign bits.
    scale6_hp = _fmul_rn(_fdiv_rn(block_amax, Float32(6.0)), global_encode_scale)
    scale4_hp = _fmul_rn(scale6_hp, Float32(1.5))
    scale4 = _cvt_f32_to_e4m3(_fmin(scale4_hp, Float32(448.0)))
    scale6 = _cvt_f32_to_e4m3(_fmin(scale6_hp, Float32(448.0)))
    scale4_f32 = _cvt_e4m3_to_f32(scale4)
    scale6_f32 = _cvt_e4m3_to_f32(scale6)
    inv4 = _candidate_inverse_scale(scale4_f32, global_decode_scale)
    inv6 = _candidate_inverse_scale(scale6_f32, global_decode_scale)
    lo4, hi4 = _scale_pack_e2m1(values, inv4)
    lo6, hi6 = _scale_pack_e2m1(values, inv6)
    error4 = _candidate_error(values, lo4, hi4, scale4, global_amax, global_encode_scale, config)
    error6 = _candidate_error(values, lo6, hi6, scale6, global_amax, global_encode_scale, config)

    # Strict comparison is part of the contract: ties select map-to-6.
    selected_scale = scale6
    selected_lo = lo6
    selected_hi = hi6
    if error4 < error6:
        selected_scale = scale4
        selected_lo = lo4
        selected_hi = hi4

    return selected_scale, selected_lo, selected_hi


@cute.jit
def _dequantize_pack_pair(packed_pair: Uint32, final_scale: Float32, is_bfloat16: bool) -> Uint32:
    q0, q1 = _e2m1x2_to_f32x2(packed_pair)
    out0, out1 = _scale_f32x2_rn(q0, q1, final_scale)
    if cutlass.const_expr(is_bfloat16):
        return _pack_f32x2_to_bfloat2(out0, out1)
    return _pack_f32x2_to_half2(out0, out1)


@cute.jit
def _dequantize_store(
    output: cute.Tensor,
    offset: Int32,
    lo: Uint32,
    hi: Uint32,
    scale: Uint32,
    global_amax: Float32,
    e4m3_max: int,
    is_bfloat16: bool,
) -> None:
    """Dequantize, cast, and store one half-block at a time."""
    scale_f32 = _cvt_e4m3_to_f32(scale)
    final_scale = _fmul_rn(scale_f32, global_amax)
    final_scale = _fmul_rn(final_scale, Float32(1.0 / float(6 * e4m3_max)))

    ptr0 = _get_ptr(output, offset)
    out0 = _dequantize_pack_pair(lo, final_scale, is_bfloat16)
    out1 = _dequantize_pack_pair(lo >> Uint32(8), final_scale, is_bfloat16)
    out2 = _dequantize_pack_pair(lo >> Uint32(16), final_scale, is_bfloat16)
    out3 = _dequantize_pack_pair(lo >> Uint32(24), final_scale, is_bfloat16)
    _store_v4_u32(ptr0, out0, out1, out2, out3)

    ptr1 = _get_ptr(output, offset + Int32(8))
    out4 = _dequantize_pack_pair(hi, final_scale, is_bfloat16)
    out5 = _dequantize_pack_pair(hi >> Uint32(8), final_scale, is_bfloat16)
    out6 = _dequantize_pack_pair(hi >> Uint32(16), final_scale, is_bfloat16)
    out7 = _dequantize_pack_pair(hi >> Uint32(24), final_scale, is_bfloat16)
    _store_v4_u32(ptr1, out4, out5, out6, out7)


class _NVFP4QDQKernel:
    """One thread processes one contiguous 1x16 quantization block."""

    def __init__(self, is_bfloat16: bool, config: NVFP4QDQConfig) -> None:
        self.is_bfloat16 = is_bfloat16
        self.config = config
        if config.use_4over6:
            self.threads = _4OVER6_THREADS
            self.min_blocks_per_sm = _4OVER6_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _4OVER6_BLOCKS_PER_SM
        else:
            self.threads = _STANDARD_THREADS
            self.min_blocks_per_sm = _STANDARD_MIN_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _STANDARD_GRID_BLOCKS_PER_SM

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amax: cute.Tensor,
        total_blocks: Int32,
        num_ctas: Int32,
        stream,
    ) -> None:
        self.kernel(input_tensor, output_tensor, global_amax, total_blocks).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads, 1, 1],
            max_number_threads=[self.threads, 1, 1],
            min_blocks_per_mp=self.min_blocks_per_sm,
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amax: cute.Tensor,
        total_blocks: Int32,
    ) -> None:
        """Quantize and immediately dequantize grid-stride 1x16 blocks."""
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()

        amax = Float32(global_amax[Int32(0)])
        global_encode_scale = _global_encode_scale(amax, self.config.e4m3_max)
        global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)
        block = block_idx * Int32(self.threads) + thread_idx
        stride = grid_dim * Int32(self.threads)
        while block < total_blocks:
            offset = block * Int32(_FP4_BLOCK_SIZE)
            ptr0 = _get_ptr(input_tensor, offset)
            ptr1 = _get_ptr(input_tensor, offset + Int32(8))
            w0, w1, w2, w3 = _load_v4_u32(ptr0)
            w4, w5, w6, w7 = _load_v4_u32(ptr1)
            words = (w0, w1, w2, w3, w4, w5, w6, w7)
            block_amax = _block_amax(words, self.is_bfloat16)

            if cutlass.const_expr(self.config.use_4over6):
                values = _input_values(words, self.is_bfloat16)
                scale, lo, hi = _four_over_six_quantize(
                    values, block_amax, amax, global_encode_scale, global_decode_scale, self.config
                )
            else:
                scale, lo, hi = _standard_quantize(
                    words, block_amax, global_encode_scale, global_decode_scale, self.is_bfloat16
                )

            _dequantize_store(output_tensor, offset, lo, hi, scale, amax, self.config.e4m3_max, self.is_bfloat16)
            block = block + stride


@dataclass(frozen=True)
class _NVFP4QDQSpecialization:
    """Compiled callable and its statically selected launch geometry."""

    launch: Any
    threads: int
    grid_blocks_per_sm: int


_KERNEL_CACHE: dict[tuple[Any, ...], _NVFP4QDQSpecialization] = {}


@functools.cache
def _device_info(device_index: int) -> tuple[tuple[int, int], int]:
    """Cache immutable capability and SM-count metadata outside the QAT hot path."""
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
        multiprocessors = torch.cuda.get_device_properties(device_index).multi_processor_count
    return capability, multiprocessors


def _validate_input(x: torch.Tensor, amax: torch.Tensor) -> tuple[int, tuple[int, int], int, int]:
    if not x.is_cuda:
        raise ValueError("Fused NVFP4 QDQ requires a CUDA tensor.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Fused NVFP4 QDQ supports BF16 and FP16, got {x.dtype}.")
    if x.ndim != 2:
        raise ValueError(f"Fused NVFP4 QDQ requires a rank-2 tensor, got shape {tuple(x.shape)}.")
    if not x.is_contiguous():
        raise ValueError("Fused NVFP4 QDQ requires a contiguous tensor.")
    if x.data_ptr() % 16 != 0:
        raise ValueError("Fused NVFP4 QDQ requires a 16-byte-aligned input tensor.")
    if x.shape[1] % _FP4_BLOCK_SIZE != 0:
        raise ValueError(f"Fused NVFP4 QDQ requires K divisible by {_FP4_BLOCK_SIZE}, got {x.shape[1]}.")
    num_elements = x.numel()
    if num_elements == 0:
        raise ValueError("Fused NVFP4 QDQ does not support empty tensors.")
    if num_elements > _INT32_MAX:
        raise ValueError(f"Fused NVFP4 QDQ supports at most {_INT32_MAX} elements, got {num_elements}.")
    if not amax.is_cuda or amax.device != x.device:
        raise ValueError("The FP32 per-tensor amax must be on the input tensor's CUDA device.")
    if amax.dtype != torch.float32 or amax.numel() != 1:
        raise TypeError("The per-tensor amax must contain exactly one FP32 value.")
    device_index = x.device.index
    if device_index is None:
        raise RuntimeError("CUDA tensor does not have a concrete device index.")
    capability, multiprocessors = _device_info(device_index)
    if capability[0] != 10:
        raise ValueError(f"Fused NVFP4 QDQ requires SM10x, got compute capability {capability}.")
    return device_index, capability, multiprocessors, num_elements


def _compile_specialization(dtype: torch.dtype, config: NVFP4QDQConfig) -> _NVFP4QDQSpecialization:
    """Compile one dtype/config specialization outside the steady-state path."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up fused NVFP4 QDQ before CUDA graph capture.")
    kernel = _NVFP4QDQKernel(dtype == torch.bfloat16, config)
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    dynamic_elements = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(element_type, (dynamic_elements,), assumed_align=16)
    output_fake = cute.runtime.make_fake_compact_tensor(element_type, (dynamic_elements,), assumed_align=16)
    amax_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        input_fake,
        output_fake,
        amax_fake,
        Int32(1),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return _NVFP4QDQSpecialization(
        launch=compiled, threads=kernel.threads, grid_blocks_per_sm=kernel.grid_blocks_per_sm
    )


def _launch_fused_nvfp4_qdq(
    x: torch.Tensor,
    amax: torch.Tensor,
    config: NVFP4QDQConfig,
    capability: tuple[int, int],
    multiprocessors: int,
    num_elements: int,
) -> torch.Tensor:
    """Launch on the current CUDA device with cached static dispatch."""
    key = (capability, x.dtype, config)
    specialization = _KERNEL_CACHE.get(key)
    if specialization is None:
        specialization = _compile_specialization(x.dtype, config)
        _KERNEL_CACHE[key] = specialization

    output = torch.empty_like(x)
    input_flat = x.detach().view(-1)
    output_flat = output.view(-1)
    amax_flat = amax.detach().reshape(1)
    total_blocks = num_elements // _FP4_BLOCK_SIZE
    num_ctas = min(
        (total_blocks + specialization.threads - 1) // specialization.threads,
        multiprocessors * specialization.grid_blocks_per_sm,
    )
    specialization.launch(input_flat, output_flat, amax_flat, total_blocks, num_ctas)
    return output


def compute_nvfp4_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute the TE-compatible FP32 per-tensor amax with PyTorch."""
    if x.numel() == 0:
        raise ValueError("Cannot compute NVFP4 amax for an empty tensor.")
    return torch.linalg.vector_norm(x.detach(), ord=float("inf"), dtype=torch.float32)


def fused_nvfp4_qdq(x: torch.Tensor, amax: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Run register-resident NVFP4 QDQ and return a detached high-precision tensor."""
    if config is None:
        config = current_nvfp4_qdq_config()
    device_index, capability, multiprocessors, num_elements = _validate_input(x, amax)

    if torch.cuda.current_device() == device_index:
        return _launch_fused_nvfp4_qdq(x, amax, config, capability, multiprocessors, num_elements)
    with torch.cuda.device(device_index):
        return _launch_fused_nvfp4_qdq(x, amax, config, capability, multiprocessors, num_elements)


class _FusedNVFP4QDQSTE(torch.autograd.Function):
    """Identity backward around the non-differentiable fused QDQ kernel."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, amax: torch.Tensor, config: NVFP4QDQConfig) -> torch.Tensor:
        """Apply fused QDQ in the STE forward pass."""
        del ctx
        return fused_nvfp4_qdq(x, amax, config)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Propagate the weight gradient through fake quantization unchanged."""
        del ctx
        return grad_output, None, None


def fake_nvfp4_quantization_ste(x: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Apply fused NVFP4 QDQ in forward and the straight-through estimator in backward."""
    if config is None:
        config = current_nvfp4_qdq_config()
    amax = compute_nvfp4_amax(x)
    output = _FusedNVFP4QDQSTE.apply(x, amax, config)
    if hasattr(x, "main_grad"):
        output.main_grad = x.main_grad
    return output


__all__ = [
    "NVFP4QDQConfig",
    "NVFP4QDQErrorMode",
    "compute_nvfp4_amax",
    "current_nvfp4_qdq_config",
    "fake_nvfp4_quantization_ste",
    "fused_nvfp4_qdq",
]
