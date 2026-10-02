# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed grouped NVFP4 fake-QAT QDQ for ordinary contiguous [G, M, K] tensors.

The expert grid and persistent scheduling come from Ziang Li's packed QDQ
experiments in radixark/Megatron-LM#87 (46a4fee12f665a1bc1323576d64e33a14211d1fa).
Numerical helpers and configuration are shared with Miles' rank-2 kernel.
Each expert has its own FP32 amax; no reduction crosses expert boundaries.
TE GroupedTensor adaptation lives in nvfp4_fake_qat, outside the kernel API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from miles.utils.fused_nvfp4_qdq import (
    _FP4_BLOCK_SIZE,
    _INT32_MAX,
    NVFP4QDQConfig,
    _block_amax,
    _dequantize_store,
    _device_info,
    _fdiv_rn,
    _four_over_six_quantize,
    _get_ptr,
    _global_encode_scale,
    _input_values,
    _load_v4_u32,
    _NVFP4QDQSpecialization,
    _standard_quantize,
    current_nvfp4_qdq_config,
)

_MAX_GROUP_COUNT = 2048
_STANDARD_THREADS = 256
_STANDARD_MIN_BLOCKS_PER_SM = 4
_STANDARD_GRID_BLOCKS_PER_SM = 96
_4OVER6_THREADS = 128
_4OVER6_MIN_BLOCKS_PER_SM = 8
_4OVER6_GRID_BLOCKS_PER_SM = 64


class _GroupedNVFP4QDQKernel:
    """Process a homogeneous runtime-sized group of contiguous weights."""

    def __init__(self, is_bfloat16: bool, config: NVFP4QDQConfig) -> None:
        self.is_bfloat16 = is_bfloat16
        self.config = config
        if config.use_4over6:
            self.threads = _4OVER6_THREADS
            self.min_blocks_per_sm = _4OVER6_MIN_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _4OVER6_GRID_BLOCKS_PER_SM
        else:
            self.threads = _STANDARD_THREADS
            self.min_blocks_per_sm = _STANDARD_MIN_BLOCKS_PER_SM
            self.grid_blocks_per_sm = _STANDARD_GRID_BLOCKS_PER_SM

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amaxes: cute.Tensor,
        blocks_per_weight: Int32,
        ctas_per_weight: Int32,
        group_count: Int32,
        stream,
    ) -> None:
        self.kernel(input_tensor, output_tensor, global_amaxes, blocks_per_weight).launch(
            grid=[group_count, ctas_per_weight, 1],
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
        global_amaxes: cute.Tensor,
        blocks_per_weight: Int32,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        group_idx, cta_idx, _ = cute.arch.block_idx()
        _, ctas_per_weight, _ = cute.arch.grid_dim()

        elements_per_weight = blocks_per_weight * Int32(_FP4_BLOCK_SIZE)
        byte_stride = Int64(elements_per_weight) * Int64(2)
        input_base = _get_ptr(input_tensor, Int32(0)) + Int64(group_idx) * byte_stride
        output_base = _get_ptr(output_tensor, Int32(0)) + Int64(group_idx) * byte_stride
        amax = Float32(global_amaxes[group_idx])
        global_encode_scale = _global_encode_scale(amax, self.config.e4m3_max)
        global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)
        block = cta_idx * Int32(self.threads) + thread_idx
        stride = ctas_per_weight * Int32(self.threads)
        while block < blocks_per_weight:
            offset = block * Int32(_FP4_BLOCK_SIZE)
            ptr0 = input_base + Int64(offset) * Int64(2)
            ptr1 = ptr0 + Int64(16)
            w0, w1, w2, w3 = _load_v4_u32(ptr0)
            w4, w5, w6, w7 = _load_v4_u32(ptr1)
            words = (w0, w1, w2, w3, w4, w5, w6, w7)
            block_amax = _block_amax(words, self.is_bfloat16)

            if cutlass.const_expr(self.config.use_4over6):
                values = _input_values(words, self.is_bfloat16)
                scale, lo, hi = _four_over_six_quantize(
                    values,
                    block_amax,
                    amax,
                    global_encode_scale,
                    global_decode_scale,
                    self.config,
                )
            else:
                scale, lo, hi = _standard_quantize(
                    words,
                    block_amax,
                    global_encode_scale,
                    global_decode_scale,
                    self.is_bfloat16,
                )

            _dequantize_store(
                output_base,
                offset,
                lo,
                hi,
                scale,
                amax,
                self.config.e4m3_max,
                self.is_bfloat16,
            )
            block = block + stride


_KERNEL_CACHE: dict[tuple[Any, ...], _NVFP4QDQSpecialization] = {}


@dataclass(frozen=True)
class _GroupedNVFP4QDQInputMetadata:
    """Validated runtime dimensions and device launch metadata."""

    device_index: int
    capability: tuple[int, int]
    multiprocessors: int
    group_count: int
    elements_per_weight: int
    blocks_per_weight: int


def _validate_input(x: torch.Tensor, amaxes: torch.Tensor) -> _GroupedNVFP4QDQInputMetadata:
    if hasattr(x, "rowwise_data"):
        raise TypeError("Pass ordinary packed storage to grouped QDQ; adapt TE weights with nvfp4_fake_qat.")
    if not x.is_cuda:
        raise ValueError("Fused NVFP4 QDQ requires a CUDA tensor.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Fused NVFP4 QDQ supports BF16 and FP16, got {x.dtype}.")
    if x.ndim != 3:
        raise ValueError("Fused NVFP4 QDQ requires a rank-3 [G, M, N] tensor, " f"got shape {tuple(x.shape)}.")
    group_count, rows, columns = x.shape
    if group_count < 1 or group_count > _MAX_GROUP_COUNT:
        raise ValueError(f"Fused NVFP4 QDQ requires 1 <= G <= {_MAX_GROUP_COUNT}, got {group_count}.")
    if rows < 1 or columns < 1:
        raise ValueError("Fused NVFP4 QDQ does not support empty weight dimensions.")
    if not x.is_contiguous():
        raise ValueError("Fused NVFP4 QDQ requires a contiguous tensor.")
    if x.data_ptr() % 16 != 0:
        raise ValueError("Fused NVFP4 QDQ requires a 16-byte-aligned input tensor.")
    if columns % _FP4_BLOCK_SIZE != 0:
        raise ValueError(f"Fused NVFP4 QDQ requires N divisible by {_FP4_BLOCK_SIZE}, got {columns}.")
    elements_per_weight = rows * columns
    if elements_per_weight > _INT32_MAX:
        raise ValueError(
            "Fused NVFP4 QDQ supports at most " f"{_INT32_MAX} elements per weight, got {elements_per_weight}."
        )
    if not amaxes.is_cuda or amaxes.device != x.device:
        raise ValueError("The FP32 per-tensor amaxes must be on the input tensor's CUDA device.")
    if amaxes.dtype != torch.float32:
        raise TypeError("The per-tensor amaxes must use FP32.")
    if amaxes.ndim != 1 or tuple(amaxes.shape) != (group_count,):
        raise ValueError(f"The per-tensor amaxes must have shape ({group_count},), " f"got {tuple(amaxes.shape)}.")
    if not amaxes.is_contiguous():
        raise ValueError("The per-tensor amaxes must be contiguous.")
    device_index = x.device.index
    if device_index is None:
        raise RuntimeError("CUDA tensor does not have a concrete device index.")
    capability, multiprocessors = _device_info(device_index)
    if capability[0] != 10:
        raise ValueError(f"Fused NVFP4 QDQ requires SM10x, got compute capability {capability}.")
    return _GroupedNVFP4QDQInputMetadata(
        device_index=device_index,
        capability=capability,
        multiprocessors=multiprocessors,
        group_count=group_count,
        elements_per_weight=elements_per_weight,
        blocks_per_weight=elements_per_weight // _FP4_BLOCK_SIZE,
    )


def _compile_specialization(dtype: torch.dtype, config: NVFP4QDQConfig) -> _NVFP4QDQSpecialization:
    """Compile one dtype/config specialization outside the steady-state path."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up fused NVFP4 QDQ before CUDA graph capture.")
    kernel = _GroupedNVFP4QDQKernel(dtype == torch.bfloat16, config)
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    dynamic_groups = cute.sym_int()
    dynamic_rows = cute.sym_int()
    dynamic_columns = cute.sym_int()
    input_fake = cute.runtime.make_fake_compact_tensor(
        element_type,
        (dynamic_groups, dynamic_rows, dynamic_columns),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        element_type,
        (dynamic_groups, dynamic_rows, dynamic_columns),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    amax_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (dynamic_groups,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        input_fake,
        output_fake,
        amax_fake,
        Int32(1),
        Int32(1),
        Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return _NVFP4QDQSpecialization(
        launch=compiled,
        threads=kernel.threads,
        grid_blocks_per_sm=kernel.grid_blocks_per_sm,
    )


def _launch_fused_grouped_nvfp4_qdq(
    storage: torch.Tensor,
    amaxes: torch.Tensor,
    config: NVFP4QDQConfig,
    metadata: _GroupedNVFP4QDQInputMetadata,
) -> torch.Tensor:
    """Launch on the current CUDA device with cached static dispatch."""
    key = (metadata.capability, storage.dtype, config)
    specialization = _KERNEL_CACHE.get(key)
    if specialization is None:
        specialization = _compile_specialization(storage.dtype, config)
        _KERNEL_CACHE[key] = specialization

    output = torch.empty(tuple(storage.shape), dtype=storage.dtype, device=storage.device)
    natural_ctas_per_weight = (metadata.blocks_per_weight + specialization.threads - 1) // specialization.threads
    target_total_ctas = metadata.multiprocessors * specialization.grid_blocks_per_sm
    ctas_per_weight = min(
        natural_ctas_per_weight,
        max(
            1,
            (target_total_ctas + metadata.group_count - 1) // metadata.group_count,
        ),
    )
    specialization.launch(
        storage.detach(),
        output,
        amaxes.detach(),
        metadata.blocks_per_weight,
        ctas_per_weight,
        metadata.group_count,
    )
    return output


def compute_grouped_nvfp4_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute one TE-compatible FP32 amax per weight in ``[G, M, N]``."""
    if hasattr(x, "rowwise_data"):
        raise TypeError("Pass ordinary packed storage to grouped QDQ; adapt TE weights with nvfp4_fake_qat.")
    if x.ndim != 3:
        raise ValueError("NVFP4 amax requires a rank-3 [G, M, N] tensor, " f"got shape {tuple(x.shape)}.")
    if x.shape[0] < 1 or x.shape[0] > _MAX_GROUP_COUNT:
        raise ValueError(f"NVFP4 amax requires 1 <= G <= {_MAX_GROUP_COUNT}, got {x.shape[0]}.")
    if x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("Cannot compute NVFP4 amax for an empty weight dimension.")
    return torch.linalg.vector_norm(x.detach(), ord=float("inf"), dim=(1, 2), dtype=torch.float32)


def fused_grouped_nvfp4_qdq(
    x: torch.Tensor,
    amaxes: torch.Tensor,
    config: NVFP4QDQConfig | None = None,
) -> torch.Tensor:
    """QDQ a contiguous ``[G, M, N]`` group with caller-provided FP32 amaxes."""
    if config is None:
        config = current_nvfp4_qdq_config()
    metadata = _validate_input(x, amaxes)

    if torch.cuda.current_device() == metadata.device_index:
        return _launch_fused_grouped_nvfp4_qdq(x, amaxes, config, metadata)
    with torch.cuda.device(metadata.device_index):
        return _launch_fused_grouped_nvfp4_qdq(x, amaxes, config, metadata)


class _FusedNVFP4QDQSTE(torch.autograd.Function):
    """Identity backward around grouped QDQ, including its PyTorch amax."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        config: NVFP4QDQConfig,
    ) -> torch.Tensor:
        del ctx
        return fused_grouped_nvfp4_qdq(x, compute_grouped_nvfp4_amax(x), config)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


def fake_grouped_nvfp4_quantization_ste(x: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Apply grouped QDQ in forward and the straight-through estimator in backward."""
    if config is None:
        config = current_nvfp4_qdq_config()
    output = _FusedNVFP4QDQSTE.apply(x, config)
    if hasattr(x, "main_grad"):
        output.main_grad = x.main_grad
    return output
