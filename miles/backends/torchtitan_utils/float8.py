"""Dynamic rowwise W8A8 FP8 training for TorchTitan on TPU (`torch_tpu`).

Forward projections quantize activations per token row and weights per output channel (`amax / 448.0` in `float8_e4m3fn`),
matching `sglang-torchtpu --quantization fp8 --tpu-fp8-runtime channelwise`. Parameters stay in BF16 with FP32 optimizer
states, so FSDP2 collectives, checkpoints, LoRA adapters, and ICI weight transfers keep their BF16 layouts unchanged.
"""

import dataclasses
import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.models.base import BaseModel
from torchtitan.models.common.linear import ColumnParallelLinear, GroupedLinear, Linear, RowParallelLinear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear

from miles.backends.torchtitan_utils.arguments import TITAN_FP8_RECIPES
from miles.platforms import current_platform

logger = logging.getLogger(__name__)

_FMAX_E4M3 = float(torch.finfo(torch.float8_e4m3fn).max)
_FMAX_E5M2 = float(torch.finfo(torch.float8_e5m2).max)
_SCALE_MIN = 1e-12
_MIN_SCALED_MM_DIM = 16


def _quantize_rowwise_2d(
    tensor_2d: torch.Tensor, fp8_dtype: torch.dtype, fmax: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D `[rows, cols]` tensor per row, returning `(quantized, scale_rows_1)`."""
    tensor_f32 = current_platform.optimization_barrier(tensor_2d).float()
    scale = torch.clamp(tensor_f32.abs().amax(dim=1, keepdim=True) / fmax, min=_SCALE_MIN)
    quantized = current_platform.optimization_barrier(
        torch.clamp(tensor_f32 / scale, min=-fmax, max=fmax).to(fp8_dtype)
    )
    return quantized, scale


def _quantize_last_dim(tensor: torch.Tensor, fp8_dtype: torch.dtype, fmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `tensor` along its trailing contraction axis, returning `(quantized, scale_without_last_dim)`."""
    tensor_f32 = current_platform.optimization_barrier(tensor).float()
    scale_expanded = torch.clamp(tensor_f32.abs().amax(dim=-1, keepdim=True) / fmax, min=_SCALE_MIN)
    quantized = current_platform.optimization_barrier(
        torch.clamp(tensor_f32 / scale_expanded, min=-fmax, max=fmax).to(fp8_dtype)
    )
    return quantized, scale_expanded.squeeze(-1)


def _linear_spmd_typecheck(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | spmd.rules.NdimWithSpmdType:
    stack = "_" if weight.dim() == 3 else ""
    output = spmd.rules.einsum(f"...k,{stack}nk->...n", input, weight, linear_in=(0, 1))
    if bias is None:
        return output
    return spmd.rules.einsum(f"...n,{stack}n->...n", output, bias, linear_in=((0, 1),))


def _float8_linear_forward(
    ctx,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fp8_wgrad: bool,
) -> torch.Tensor:
    weight_NK, bias_N = Linear._flatten_weight_and_bias(weight, bias)
    input_MK = input.reshape(-1, input.shape[-1])
    input_q_MK, input_scale_M1 = _quantize_rowwise_2d(input_MK, torch.float8_e4m3fn, _FMAX_E4M3)
    weight_q_NK, weight_scale_N1 = _quantize_rowwise_2d(weight_NK, torch.float8_e4m3fn, _FMAX_E4M3)
    output_MN = torch._scaled_mm(
        input_q_MK,
        weight_q_NK.t(),
        input_scale_M1,
        weight_scale_N1.t(),
        bias=bias_N,
        out_dtype=input.dtype,
    )
    ctx.save_for_backward(input_MK, weight_NK)
    ctx.input_shape = input.shape
    ctx.weight_shape = weight.shape
    ctx.bias_shape = None if bias is None else bias.shape
    ctx.fp8_wgrad = fp8_wgrad
    return output_MN.reshape(*input.shape[:-1], weight_NK.shape[0])


def _float8_linear_backward(
    ctx, grad_output: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    input_MK, weight_NK = ctx.saved_tensors
    grad_MN = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input = grad_weight = grad_bias = None

    if ctx.needs_input_grad[0]:
        grad_q_MN, grad_scale_M1 = _quantize_rowwise_2d(grad_MN, torch.float8_e5m2, _FMAX_E5M2)
        weight_KN = weight_NK.t().contiguous()
        weight_q_KN, weight_scale_K1 = _quantize_rowwise_2d(weight_KN, torch.float8_e4m3fn, _FMAX_E4M3)
        grad_input_MK = torch._scaled_mm(
            grad_q_MN,
            weight_q_KN.t(),
            grad_scale_M1,
            weight_scale_K1.t(),
            out_dtype=grad_output.dtype,
        )
        grad_input = grad_input_MK.reshape(ctx.input_shape)

    if ctx.needs_input_grad[1]:
        if ctx.fp8_wgrad:
            grad_NM = grad_MN.t().contiguous()
            grad_q_NM, grad_scale_N1 = _quantize_rowwise_2d(grad_NM, torch.float8_e5m2, _FMAX_E5M2)
            input_KM = input_MK.t().contiguous()
            input_q_KM, input_scale_K1 = _quantize_rowwise_2d(input_KM, torch.float8_e4m3fn, _FMAX_E4M3)
            grad_weight_NK = torch._scaled_mm(
                grad_q_NM,
                input_q_KM.t(),
                grad_scale_N1,
                input_scale_K1.t(),
                out_dtype=weight_NK.dtype,
            )
        else:
            grad_weight_NK = grad_MN.t().mm(input_MK.to(grad_MN.dtype)).to(weight_NK.dtype)
        grad_weight = grad_weight_NK.reshape(ctx.weight_shape)

    if ctx.needs_input_grad[2] and ctx.bias_shape is not None:
        grad_bias = grad_MN.sum(0).reshape(ctx.bias_shape)

    return grad_input, grad_weight, grad_bias


@spmd.register_autograd_function
class _Float8LinearRowwiseFunction(torch.autograd.Function):
    """Rowwise FP8 linear with FP8 `fprop`, `dgrad`, and `wgrad`."""

    spmd_typecheck = staticmethod(_linear_spmd_typecheck)

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return _float8_linear_forward(ctx, input, weight, bias, fp8_wgrad=True)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return _float8_linear_backward(ctx, grad_output)


@spmd.register_autograd_function
class _Float8LinearHpWgradFunction(torch.autograd.Function):
    """Rowwise FP8 linear with FP8 `fprop` + `dgrad` and high-precision BF16 `wgrad`."""

    spmd_typecheck = staticmethod(_linear_spmd_typecheck)

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return _float8_linear_forward(ctx, input, weight, bias, fp8_wgrad=False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return _float8_linear_backward(ctx, grad_output)


@spmd.register_autograd_function
class _Float8GroupedMmFunction(torch.autograd.Function):
    """Rowwise FP8 grouped GEMM (`fprop` + `dgrad` via `torch._scaled_grouped_mm`, BF16 `wgrad` via `torch._grouped_mm`)."""

    @staticmethod
    def spmd_typecheck(
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor | spmd.rules.NdimWithSpmdType:
        del offsets_E
        return spmd.rules.einsum("ri,eoi->ro", input_RI, weight_EOI, linear_in=(0, 1))

    @staticmethod
    def forward(
        ctx,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        offsets_i32 = offsets_E if offsets_E.dtype == torch.int32 else offsets_E.to(torch.int32)
        input_q_RI, input_scale_R = _quantize_last_dim(input_RI, torch.float8_e4m3fn, _FMAX_E4M3)
        weight_q_EOI, weight_scale_EO = _quantize_last_dim(weight_EOI, torch.float8_e4m3fn, _FMAX_E4M3)
        output_RO = torch._scaled_grouped_mm(
            input_q_RI,
            weight_q_EOI.transpose(-2, -1),
            input_scale_R,
            weight_scale_EO,
            offs=offsets_i32,
            out_dtype=input_RI.dtype,
        )
        ctx.save_for_backward(input_RI, weight_EOI, offsets_i32)
        return output_RO

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_RO: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        input_RI, weight_EOI, offsets_i32 = ctx.saved_tensors
        grad_input_RI = grad_weight_EOI = None

        if ctx.needs_input_grad[0]:
            grad_q_RO, grad_scale_R = _quantize_last_dim(grad_RO, torch.float8_e5m2, _FMAX_E5M2)
            weight_EIO = weight_EOI.transpose(-2, -1).contiguous()
            weight_q_EIO, weight_scale_EI = _quantize_last_dim(weight_EIO, torch.float8_e4m3fn, _FMAX_E4M3)
            grad_input_RI = torch._scaled_grouped_mm(
                grad_q_RO,
                weight_q_EIO.transpose(-2, -1),
                grad_scale_R,
                weight_scale_EI,
                offs=offsets_i32,
                out_dtype=grad_RO.dtype,
            )

        if ctx.needs_input_grad[1]:
            grad_weight_EOI = torch._grouped_mm(
                grad_RO.t(),
                input_RI.to(grad_RO.dtype),
                offs=offsets_i32,
            ).to(weight_EOI.dtype)

        return grad_input_RI, grad_weight_EOI, None


@functools.cache
def get_float8_linear(parent_cls: type[Linear], *, fp8_wgrad: bool) -> type[Linear]:
    """Return a cached `parent_cls` subclass whose `_linear` seam runs rowwise FP8 `_scaled_mm`."""
    autograd_fn = _Float8LinearRowwiseFunction if fp8_wgrad else _Float8LinearHpWgradFunction
    prefix = "Float8Rowwise" if fp8_wgrad else "Float8RowwiseHpWgrad"

    class _Float8Linear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_cls.Config):  # type: ignore[misc]
            pass

        def _linear(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            if self.inplace_wgrad_accum:
                raise RuntimeError(f"{type(self).__qualname__} requires inplace_wgrad_accum=False")
            return autograd_fn.apply(input, weight, bias)

    _Float8Linear.__name__ = f"{prefix}{parent_cls.__name__}"
    _Float8Linear.__qualname__ = _Float8Linear.__name__
    return _Float8Linear


@functools.cache
def get_float8_grouped_linear(parent_cls: type[GroupedLinear]) -> type[GroupedLinear]:
    """Return a cached `parent_cls` subclass whose `_grouped_mm` seam runs rowwise FP8 `_scaled_grouped_mm`."""

    class _Float8GroupedLinear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_cls.Config):  # type: ignore[misc]
            pass

        def _grouped_mm(
            self,
            *,
            input_RI: torch.Tensor,
            weight_EOI: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            return _Float8GroupedMmFunction.apply(input_RI, weight_EOI, offsets_E)

    _Float8GroupedLinear.__name__ = f"Float8RowwiseHpWgrad{parent_cls.__name__}"
    _Float8GroupedLinear.__qualname__ = _Float8GroupedLinear.__name__
    return _Float8GroupedLinear


def _is_filtered(fqn: str, filter_fqns: Sequence[str]) -> bool:
    return any(pattern in fqn for pattern in filter_fqns)


def _local_linear_dims(cfg: Linear.Config, *, tp_degree: int) -> tuple[int, int]:
    total_out = cfg.num_linears * cfg.out_features
    if tp_degree > 1 and isinstance(cfg, ColumnParallelLinear.Config):
        return cfg.in_features, total_out // tp_degree
    if tp_degree > 1 and isinstance(cfg, RowParallelLinear.Config):
        return cfg.in_features // tp_degree, total_out
    return cfg.in_features, total_out


def _local_grouped_dims(fqn: str, cfg: GroupedLinear.Config, *, tp_degree: int) -> tuple[int, int]:
    total_out = cfg.num_linears * cfg.out_features
    if tp_degree > 1 and fqn.endswith((".w1", ".w3", ".w13")):
        return cfg.in_features, total_out // tp_degree
    if tp_degree > 1 and fqn.endswith(".w2"):
        return cfg.in_features // tp_degree, total_out
    return cfg.in_features, total_out


def _shallow_config_kwargs(cfg: object) -> dict[str, object]:
    return {
        field.name: getattr(
            cfg, field.name
        )  # config-access-exempt: copy TorchTitan Configurable.Config slot fields shallowly
        for field in dataclasses.fields(cfg)
    }


def _replace_child(parent: object, attr: str | int, replacement: object) -> None:
    if isinstance(parent, list):
        parent[int(attr)] = replacement
    else:
        setattr(parent, str(attr), replacement)


def apply_float8_to_model_config(
    model_config: BaseModel.Config,
    *,
    recipe_name: str,
    filter_fqns: Sequence[str] = (),
    tp_degree: int = 1,
) -> None:
    """Convert transformer-block `Linear.Config` and `GroupedLinear.Config` nodes to rowwise FP8 in place.

    Embeddings, `lm_head`, multimodal encoders, MTP layers (everything outside `layers.*`), and `HiMidLoLinear`
    projections (MoE router gates) stay in high precision. Projections whose local GEMM dimensions are not multiples of
    16 are skipped and logged.
    """
    if recipe_name not in TITAN_FP8_RECIPES:
        raise ValueError(f"unknown FP8 recipe {recipe_name!r}; expected one of {TITAN_FP8_RECIPES}")

    fp8_wgrad = recipe_name == "rowwise"
    skipped_unaligned: list[str] = []

    grouped_candidates: list[tuple[str, GroupedLinear.Config, object, str | int]] = []
    for fqn, grouped_cfg, parent, attr in model_config.traverse(GroupedLinear.Config):
        if parent is None or attr is None or not fqn.startswith("layers."):
            continue
        if _is_filtered(fqn, filter_fqns):
            continue
        local_in, local_out = _local_grouped_dims(fqn, grouped_cfg, tp_degree=tp_degree)
        if local_in % _MIN_SCALED_MM_DIM != 0 or local_out % _MIN_SCALED_MM_DIM != 0:
            skipped_unaligned.append(f"{fqn} ({local_in}->{local_out})")
            continue
        grouped_candidates.append((fqn, grouped_cfg, parent, attr))

    if grouped_candidates and fp8_wgrad:
        raise ValueError(
            "MoE grouped expert projections require --titan-fp8-recipe rowwise_with_gw_hp "
            "(ragged expert segments share the contraction axis in WGRAD, so expert weight gradients run in BF16); "
            "pass --titan-fp8-recipe rowwise_with_gw_hp or exclude routed_experts via --titan-fp8-filter-fqns"
        )

    converted_dense = 0
    for fqn, linear_cfg, parent, attr in list(model_config.traverse(Linear.Config)):
        if parent is None or attr is None or not fqn.startswith("layers."):
            continue
        if isinstance(linear_cfg, HiMidLoLinear.Config):
            continue
        if _is_filtered(fqn, filter_fqns):
            continue
        local_in, local_out = _local_linear_dims(linear_cfg, tp_degree=tp_degree)
        if local_in % _MIN_SCALED_MM_DIM != 0 or local_out % _MIN_SCALED_MM_DIM != 0:
            skipped_unaligned.append(f"{fqn} ({local_in}->{local_out})")
            continue
        parent_owner = type(linear_cfg)._owner
        assert parent_owner is not None and issubclass(parent_owner, Linear)
        float8_linear_cls = get_float8_linear(parent_owner, fp8_wgrad=fp8_wgrad)
        kwargs = _shallow_config_kwargs(linear_cfg)
        kwargs["inplace_wgrad_accum"] = False
        _replace_child(parent, attr, float8_linear_cls.Config(**kwargs))
        converted_dense += 1

    converted_grouped = 0
    for _fqn, grouped_cfg, parent, attr in grouped_candidates:
        parent_owner = type(grouped_cfg)._owner
        assert parent_owner is not None and issubclass(parent_owner, GroupedLinear)
        float8_grouped_cls = get_float8_grouped_linear(parent_owner)
        _replace_child(parent, attr, float8_grouped_cls.Config(**_shallow_config_kwargs(grouped_cfg)))
        converted_grouped += 1

    logger.info(
        f"Configured TorchTitan FP8 ({recipe_name}): converted {converted_dense} dense linear and "
        f"{converted_grouped} grouped expert projections"
    )
    if skipped_unaligned:
        logger.info(
            f"Skipped {len(skipped_unaligned)} projection(s) with local dims not divisible by {_MIN_SCALED_MM_DIM}: "
            f"{', '.join(skipped_unaligned)}"
        )
