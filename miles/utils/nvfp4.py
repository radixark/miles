import os
from collections import defaultdict
from collections.abc import Sequence

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16
TE_NVFP4_SPLIT_ROW_ALIGNMENT = 64
TE_NVFP4_MAX_SPLITS = 64


def nvfp4_weight_e4m3_max() -> int:
    if os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all") and os.getenv(
        "NVTE_NVFP4_4OVER6_E4M3_USE_256", "all"
    ).strip().lower() in ("weights", "all"):
        return 256
    return int(FP8_E4M3_MAX)


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    # Python scalar operands are passed to device kernels without allocating
    # and copying scalar tensors. Both supported numerators (1536 and 2688)
    # are exactly representable in FP32.
    global_encode_scale = torch.div(float(nvfp4_e4m3_max) * FP4_E2M1_MAX, global_amax.to(torch.float32)).clamp_max(
        torch.finfo(torch.float32).max
    )
    # A Python conditional on a CUDA scalar synchronizes once per expert
    # quantization during weight publication. Keep the TE zero-scale rule on
    # device for scalar and batched amax alike.
    return torch.where(global_encode_scale == 0.0, 1.0, global_encode_scale)


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    nvfp4_e4m3_max: int = int(FP8_E4M3_MAX),
) -> torch.Tensor:
    return torch.div(1.0, nvfp4_global_encode_scale_te(global_amax, nvfp4_e4m3_max))


def _nvfp4_4over6_enabled() -> bool:
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in ("weights", "all")


def _pad_rows_for_te_quantizer(weight: torch.Tensor, alignment=TE_NVFP4_ROW_ALIGNMENT) -> torch.Tensor:
    pad_rows = (-weight.shape[0]) % alignment
    if pad_rows == 0:
        return weight
    padding = torch.zeros((pad_rows, weight.shape[1]), device=weight.device, dtype=weight.dtype)
    return torch.cat((weight, padding), dim=0)


def nvfp4_quantize_1d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()
    quantizer = _make_nvfp4_quantizer(nvfp4_e4m3_max)

    quantized = quantizer.quantize(_pad_rows_for_te_quantizer(weight))
    qweight, block_scale = _unpack_nvfp4(quantized, num_rows, num_cols)
    amax = quantized._amax_rowwise.reshape(-1)[0]
    return qweight, block_scale, nvfp4_global_decode_scale_te(amax, nvfp4_e4m3_max)


def _make_nvfp4_quantizer(nvfp4_e4m3_max):
    # TransformerEngine is an optional CUDA dependency.
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    return NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=_nvfp4_4over6_enabled(),
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper(),
        with_random_sign_mask=False,
    )


def _unpack_nvfp4(quantized, num_rows, num_cols):
    qweight = quantized._rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = quantized._rowwise_scale_inv[:num_rows, : num_cols // NVFP4_GROUP_SIZE].contiguous()
    return qweight, block_scale.view(torch.float8_e4m3fn)


def nvfp4_quantize_1d_batch(
    weights: Sequence[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Batch independent NVFP4 AMAX reductions and scale conversion on CUDA.

    TE split_quantize bulk-allocates outputs and computes one AMAX per split.
    Its no-RHT path still launches a quantization kernel for each split. Packing
    disjoint inputs adds a temporary copy bounded by the caller's expert batch.
    """
    batches = defaultdict(list)
    for index, weight in enumerate(weights):
        if weight.dim() != 2 or weight.shape[1] % NVFP4_GROUP_SIZE:
            raise ValueError("NVFP4 batches require 2D weights with K divisible by 16.")
        batches[(weight.device, weight.dtype, weight.shape[1])].append(index)

    outputs = [None] * len(weights)
    for (device, dtype, num_cols), indices in batches.items():
        # TE's allocation and kernel launch use the current CUDA device.
        with torch.cuda.device(device):
            # TE's split AMAX supports only BF16 with K % 128 == 0. Keep other
            # dtypes and narrow/empty inputs on the existing scalar path.
            if dtype != torch.bfloat16 or num_cols % 128 or any(weights[index].numel() == 0 for index in indices):
                for index in indices:
                    outputs[index] = nvfp4_quantize_1d(weights[index])
                continue
            for start in range(0, len(indices), TE_NVFP4_MAX_SPLITS):
                chunk = indices[start : start + TE_NVFP4_MAX_SPLITS]
                if len(chunk) == 1:
                    outputs[chunk[0]] = nvfp4_quantize_1d(weights[chunk[0]])
                    continue
                results = _nvfp4_quantize_split([weights[index] for index in chunk])
                for index, result in zip(chunk, results, strict=True):
                    outputs[index] = result
    return outputs


def _nvfp4_quantize_split(weights):
    e4m3_max = nvfp4_weight_e4m3_max()
    quantizers = [_make_nvfp4_quantizer(e4m3_max) for _ in weights]
    # Importing the quantizer initializes TE's extension loader first.
    import transformer_engine_torch as tex

    # Grouped AMAX requires each split to have a multiple of 64 rows; scalar
    # quantization retains its existing 16-row padding.
    padded = [_pad_rows_for_te_quantizer(weight.contiguous(), TE_NVFP4_SPLIT_ROW_ALIGNMENT) for weight in weights]
    split_rows = [weight.shape[0] for weight in padded]
    packed = padded[0]
    for weight in padded[1:]:
        packed = _contiguous_pair_view(packed, weight)
        if packed is None:
            packed = torch.cat(padded, dim=0)
            break

    quantized = tex.split_quantize(packed, split_rows, quantizers)
    del packed, padded
    amax = torch.stack([tensor._amax_rowwise.reshape(-1)[0] for tensor in quantized])
    scales = nvfp4_global_decode_scale_te(amax, e4m3_max).unbind()
    return [
        (*_unpack_nvfp4(tensor, *weight.shape), scale)
        for weight, tensor, scale in zip(weights, quantized, scales, strict=True)
    ]


def nvfp4_quantize_1d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    combined = _combine_nvfp4_pair(first, second)
    return _split_nvfp4_pair(nvfp4_quantize_1d(combined), first.shape[0], clone_scale=True)


def nvfp4_quantize_1d_pair_batch(pairs):
    """Keep each gate/up pair under its own AMAX while batching expert work."""
    combined = [_combine_nvfp4_pair(first, second) for first, second in pairs]
    outputs = nvfp4_quantize_1d_batch(combined)
    return [
        _split_nvfp4_pair(output, first.shape[0], clone_scale=False)
        for (first, _), output in zip(pairs, outputs, strict=True)
    ]


def _combine_nvfp4_pair(first, second):
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError("nvfp4_quantize_1d_pair expects two 2D tensors.")
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"NVFP4 paired quantization requires matching K dimensions, got {first.shape[1]} and {second.shape[1]}."
        )

    combined = _contiguous_pair_view(first, second)
    if combined is None:
        combined = torch.cat((first.contiguous(), second.contiguous()), dim=0)
    return combined


def _split_nvfp4_pair(output, first_rows, *, clone_scale):
    combined_qweight, combined_block_scale, global_scale = output
    # Batched export treats its scales as read-only; gate/up can share the scalar
    # view until ExpertGather copies each named tensor into the wire payload.
    first_result = (
        combined_qweight[:first_rows].contiguous(),
        combined_block_scale[:first_rows].contiguous(),
        global_scale.clone() if clone_scale else global_scale,
    )
    second_result = (
        combined_qweight[first_rows:].contiguous(),
        combined_block_scale[first_rows:].contiguous(),
        global_scale.clone() if clone_scale else global_scale,
    )
    return first_result, second_result


def _contiguous_pair_view(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor | None:
    if not first.is_contiguous() or not second.is_contiguous():
        return None
    if first.device != second.device or first.dtype != second.dtype or first.stride() != second.stride():
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if first.storage_offset() + first.numel() != second.storage_offset():
        return None

    try:
        return first.as_strided(
            (first.shape[0] + second.shape[0], first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )
    except RuntimeError:
        return None
