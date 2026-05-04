#!/usr/bin/env python3
"""
Compare MXFP8 quantization accuracy across Triton and FlashInfer backends.

Default tensor shape matches a DeepSeek MoE expert weight stack:
  [256, 2048, 7168]

The 1x32 scale block is applied along the last dimension (K=7168).
"""

import argparse
import gc
import math
from dataclasses import dataclass

import torch
from flashinfer import mxfp8_quantize
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize

GROUP_SIZE = 32


@dataclass
class Metrics:
    mae: float
    rmse: float
    max_abs: float
    mean_rel: float
    cosine: float


def parse_shape(shape_text: str) -> tuple[int, ...]:
    tokens = shape_text.replace("x", ",").split(",")
    shape = tuple(int(t.strip()) for t in tokens if t.strip())
    if len(shape) < 2:
        raise ValueError(f"Shape must have at least 2 dims, got: {shape_text}")
    return shape


def parse_dtype(dtype_text: str) -> torch.dtype:
    key = dtype_text.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_text}")


def ue8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    # UE8M0 byte -> float32 by placing the byte as FP32 exponent bits.
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def normalize_quantized_outputs(
    x_2d: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = x_2d.shape

    if q.numel() % rows != 0:
        raise ValueError(
            f"Quantized tensor cannot be reshaped to [rows, -1]: q.shape={tuple(q.shape)}, rows={rows}"
        )
    q_2d = q.contiguous().reshape(rows, -1)

    if scale.numel() % rows != 0:
        raise ValueError(
            f"Scale tensor cannot be reshaped to [rows, -1]: scale.shape={tuple(scale.shape)}, rows={rows}"
        )
    scale_2d = scale.contiguous().reshape(rows, -1)

    qk = q_2d.shape[1]
    if qk % group_size != 0:
        raise ValueError(f"Quantized K={qk} must be divisible by group_size={group_size}.")

    expected_scale_k = qk // group_size
    if scale_2d.shape[1] != expected_scale_k:
        raise ValueError(
            f"Scale/quantized shape mismatch after reshape: q.shape={tuple(q_2d.shape)}, "
            f"scale.shape={tuple(scale_2d.shape)}, expected scale second dim={expected_scale_k}."
        )
    return q_2d, scale_2d


def compute_dequant_metrics(
    x_2d: torch.Tensor,
    q_2d: torch.Tensor,
    scale_u8_2d: torch.Tensor,
    row_chunk: int,
    group_size: int = GROUP_SIZE,
) -> Metrics:
    if x_2d.ndim != 2 or q_2d.ndim != 2 or scale_u8_2d.ndim != 2:
        raise ValueError("Expected 2D tensors for x/q/scale.")

    rows, k = x_2d.shape
    if q_2d.shape[0] != rows or scale_u8_2d.shape[0] != rows:
        raise ValueError(
            f"Row mismatch: x={x_2d.shape}, q={q_2d.shape}, scale={scale_u8_2d.shape}"
        )
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}.")

    if q_2d.shape[1] < k:
        raise ValueError(f"Quantized K={q_2d.shape[1]} is smaller than input K={k}.")
    if q_2d.shape[1] > k:
        q_2d = q_2d[:, :k].contiguous()

    expected_scale_k = k // group_size
    if scale_u8_2d.shape[1] != expected_scale_k:
        raise ValueError(
            f"Scale shape mismatch: expected second dim {expected_scale_k}, got {scale_u8_2d.shape[1]}"
        )

    total = rows * k
    eps = 1e-12

    sum_abs = 0.0
    sum_sq = 0.0
    sum_rel = 0.0
    max_abs = 0.0
    dot = 0.0
    ref_norm_sq = 0.0
    dq_norm_sq = 0.0

    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)

        ref = x_2d[start:end].to(torch.float32)
        q = q_2d[start:end].to(torch.float32)
        scales = ue8m0_to_float(scale_u8_2d[start:end]).unsqueeze(-1)

        dq = (q.view(-1, expected_scale_k, group_size) * scales).view(-1, k)
        err = dq - ref
        abs_err = err.abs()

        sum_abs += abs_err.sum(dtype=torch.float64).item()
        sum_sq += err.square().sum(dtype=torch.float64).item()
        sum_rel += (abs_err / ref.abs().clamp_min(eps)).sum(dtype=torch.float64).item()
        max_abs = max(max_abs, abs_err.max().item())

        dot += (dq * ref).sum(dtype=torch.float64).item()
        ref_norm_sq += ref.square().sum(dtype=torch.float64).item()
        dq_norm_sq += dq.square().sum(dtype=torch.float64).item()

    cosine = dot / (math.sqrt(ref_norm_sq * dq_norm_sq) + 1e-12)
    return Metrics(
        mae=sum_abs / total,
        rmse=math.sqrt(sum_sq / total),
        max_abs=max_abs,
        mean_rel=sum_rel / total,
        cosine=cosine,
    )


def quantize_triton(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_group_quantize(x_2d.contiguous())


def quantize_flashinfer(
    x_2d: torch.Tensor,
    backend: str,
    quant_row_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = x_2d.shape
    if quant_row_chunk <= 0 or quant_row_chunk >= rows:
        return mxfp8_quantize(
            x_2d.contiguous(),
            is_sf_swizzled_layout=False,
            alignment=GROUP_SIZE,
            backend=backend,
        )

    q_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    for start in range(0, rows, quant_row_chunk):
        end = min(start + quant_row_chunk, rows)
        q_chunk, scale_chunk = mxfp8_quantize(
            x_2d[start:end].contiguous(),
            is_sf_swizzled_layout=False,
            alignment=GROUP_SIZE,
            backend=backend,
        )
        q_chunks.append(q_chunk)
        scale_chunks.append(scale_chunk.reshape(-1))

    return torch.cat(q_chunks, dim=0), torch.cat(scale_chunks, dim=0)


def quantize_flashinfer_cuda(x_2d: torch.Tensor, quant_row_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_flashinfer(x_2d=x_2d, backend="cuda", quant_row_chunk=quant_row_chunk)


def quantize_flashinfer_cute_dsl(x_2d: torch.Tensor, quant_row_chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_flashinfer(x_2d=x_2d, backend="cute-dsl", quant_row_chunk=quant_row_chunk)


def print_header(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, row_chunk: int) -> None:
    numel = math.prod(shape)
    print("MXFP8 quantizer accuracy comparison")
    print(f"shape={shape}  numel={numel:,}")
    print(f"dtype={dtype}  device={device}  group_size={GROUP_SIZE}  row_chunk={row_chunk}")
    print("scale layout: 1x32 along the last dimension (K)")
    print()


def print_results(rows: list[tuple[str, Metrics | None, str | None]], print_digits: int) -> None:
    col_width = max(14, print_digits + 8)
    header = (
        f"{'backend':<24}"
        f"{'mae':>{col_width}}"
        f"{'rmse':>{col_width}}"
        f"{'max_abs':>{col_width}}"
        f"{'mean_rel':>{col_width}}"
        f"{'cosine':>{col_width}}"
    )
    print(header)
    print("-" * len(header))
    for name, metrics, error in rows:
        if error is not None:
            print(f"{name:<24} ERROR: {error}")
            continue
        assert metrics is not None
        sci = f".{print_digits}e"
        fixed = f".{print_digits}f"
        print(
            f"{name:<24}"
            f"{metrics.mae:>{col_width}{sci}}"
            f"{metrics.rmse:>{col_width}{sci}}"
            f"{metrics.max_abs:>{col_width}{sci}}"
            f"{metrics.mean_rel:>{col_width}{sci}}"
            f"{metrics.cosine:>{col_width}{fixed}}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MXFP8 quantization accuracy for Triton/FlashInfer backends.")
    parser.add_argument(
        "--shape",
        type=str,
        default="256,2048,7168",
        help="Input tensor shape, e.g. 256,2048,7168",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "half"],
        help="Input dtype.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=2048,
        help="Rows per chunk for dequantization-error metric computation.",
    )
    parser.add_argument(
        "--flashinfer-cuda-quant-row-chunk",
        type=int,
        default=0,
        help="Rows per chunk for FlashInfer CUDA quantization (0 disables chunking).",
    )
    parser.add_argument(
        "--flashinfer-cute-dsl-quant-row-chunk",
        type=int,
        default=16384,
        help="Rows per chunk for FlashInfer CuTe-DSL quantization. Smaller values are safer for large tensors.",
    )
    parser.add_argument(
        "--print-digits",
        type=int,
        default=10,
        help="Number of digits to print for metrics.",
    )
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    dtype = parse_dtype(args.dtype)
    if shape[-1] % GROUP_SIZE != 0:
        raise ValueError(
            f"Last dimension K={shape[-1]} must be divisible by {GROUP_SIZE} for 1x{GROUP_SIZE} MXFP8 scaling."
        )
    if args.row_chunk <= 0:
        raise ValueError("--row-chunk must be > 0.")
    if args.flashinfer_cuda_quant_row_chunk < 0:
        raise ValueError("--flashinfer-cuda-quant-row-chunk must be >= 0.")
    if args.flashinfer_cute_dsl_quant_row_chunk <= 0:
        raise ValueError("--flashinfer-cute-dsl-quant-row-chunk must be > 0.")
    if args.print_digits <= 0:
        raise ValueError("--print-digits must be > 0.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This script expects CUDA for Triton and FlashInfer backends.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(device)

    print_header(shape=shape, dtype=dtype, device=device, row_chunk=args.row_chunk)
    x = torch.randn(shape, dtype=dtype, device=device).contiguous()
    x_2d = x.view(-1, shape[-1]).contiguous()

    backend_fns = [
        ("triton", quantize_triton),
        (
            "flashinfer-cuda",
            lambda t: quantize_flashinfer_cuda(
                t,
                quant_row_chunk=args.flashinfer_cuda_quant_row_chunk,
            ),
        ),
        (
            "flashinfer-cute-dsl",
            lambda t: quantize_flashinfer_cute_dsl(
                t,
                quant_row_chunk=args.flashinfer_cute_dsl_quant_row_chunk,
            ),
        ),
    ]

    results: list[tuple[str, Metrics | None, str | None]] = []
    for name, fn in backend_fns:
        q = None
        scale = None
        q_2d = None
        scale_2d = None
        try:
            q, scale = fn(x_2d)
            q_2d, scale_2d = normalize_quantized_outputs(x_2d=x_2d, q=q, scale=scale)
            metrics = compute_dequant_metrics(
                x_2d=x_2d,
                q_2d=q_2d,
                scale_u8_2d=scale_2d,
                row_chunk=args.row_chunk,
            )
            results.append((name, metrics, None))
        except Exception as exc:
            results.append((name, None, str(exc)))
        # finally:
        #     if q is not None:
        #         del q
        #     if scale is not None:
        #         del scale
        #     if q_2d is not None:
        #         del q_2d
        #     if scale_2d is not None:
        #         del scale_2d
        #     gc.collect()
        #     torch.cuda.empty_cache()

    print_results(results, print_digits=args.print_digits)


if __name__ == "__main__":
    main()
