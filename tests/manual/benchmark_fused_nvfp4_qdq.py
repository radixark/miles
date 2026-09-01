"""Benchmark fused NVFP4 QDQ against the former fake-QAT TE round trip.

The ``torch.utils.benchmark`` methodology follows Transformer Engine commit
83e230873f00676d4966ca151de22c8bfc68a77f. The benchmark shape is written as
``block-axis x rows`` and stored as a contiguous ``[rows, block-axis]`` tensor,
so every 1x16 NVFP4 block lies along the first reported dimension. It mirrors
``TEGroupedLinear._get_weight_tensors`` by timing the complete
``for w in weight_tensors`` loop over independently stored, equal-shaped
Parameters. The number of weights is configurable and defaults to eight.
The primary speedup compares complete user-visible loops:

* naive: pad + TE quantize + TE dequantize + crop + contiguous;
* fused: the paired Megatron env gate/local import and Miles per-weight STE adapter,
  including config resolution, PyTorch FP32 per-tensor amax, and register-resident
  CuTe DSL QDQ.

The separately labeled precomputed-amax API number still includes output
allocation, TVM-FFI argument marshalling, and host launch overhead. It is a
diagnostic and is not used for the primary speedup claim.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from statistics import geometric_mean

import cutlass
import torch
import torch.utils.benchmark as benchmark
import transformer_engine
import transformer_engine.pytorch as te

from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    fused_nvfp4_qdq,
)
from miles.utils.nvfp4_fake_qat import NVFP4_FAKE_QAT_FLAG

DEFAULT_LOGICAL_SHAPES = [(6144, 4096)]
DEFAULT_NUM_WEIGHTS = 8


def _configs() -> list[tuple[str, NVFP4QDQConfig]]:
    # Exact-error remains in the zero-tolerance correctness matrix, but the
    # performance target is standard NVFP4 plus TE's current FP16-error path.
    configs = [("nvfp4", NVFP4QDQConfig())]
    for error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
        for e4m3_max in (448, 256):
            configs.append(
                (
                    f"4over6-{error_mode.name.lower()}-e4m3-{e4m3_max}-fp16-error",
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=e4m3_max,
                        error_mode=error_mode,
                        error_use_fast_math=True,
                    ),
                )
            )
    return configs


def _make_te_quantizer(config: NVFP4QDQConfig):
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=config.use_4over6,
        nvfp4_e4m3_max=config.e4m3_max,
        nvfp4_4over6_err_mode=config.error_mode.name,
        with_random_sign_mask=False,
    )


def _naive_te_qdq(x: torch.Tensor, quantizer) -> torch.Tensor:
    m, n = x.shape
    padded_m = ((m + 15) // 16) * 16
    if padded_m == m:
        x_padded = x.contiguous()
    else:
        padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x.contiguous(), padding), dim=0)
    return quantizer.quantize(x_padded).dequantize(dtype=x.dtype)[:m, :n].contiguous()


def _median_us(function: Callable[[], object], min_run_time: float) -> float:
    timing = benchmark.Timer(stmt="function()", globals={"function": function}, num_threads=1).blocked_autorange(
        min_run_time=min_run_time
    )
    return timing.median * 1e6


def _benchmark_case(
    logical_shape: tuple[int, int],
    dtype: torch.dtype,
    config: NVFP4QDQConfig,
    num_weights: int,
    min_run_time: float,
    repeats: int,
) -> tuple[list[float], list[float], list[float]]:
    os.environ["NVTE_USE_FAST_MATH"] = "0"
    os.environ["NVTE_NVFP4_4OVER6"] = "weights" if config.use_4over6 else "none"
    os.environ["NVTE_NVFP4_4OVER6_E4M3_USE_256"] = "weights" if config.e4m3_max == 256 else "none"
    os.environ["NVTE_NVFP4_4OVER6_ERR_MODE"] = config.error_mode.name
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "1" if config.error_use_fast_math else "0"
    os.environ["OPEN_TRAINING_INT4_FAKE_QAT_FLAG"] = "0"
    os.environ[NVFP4_FAKE_QAT_FLAG] = "1"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    block_axis, rows = logical_shape
    weight_tensors = [
        torch.nn.Parameter(torch.randn((rows, block_axis), dtype=dtype, device="cuda")) for _ in range(num_weights)
    ]
    amax_tensors = [compute_nvfp4_amax(w) for w in weight_tensors]
    quantizer = _make_te_quantizer(config)

    def naive() -> list[torch.Tensor]:
        return [_naive_te_qdq(w, quantizer) for w in weight_tensors]

    def fused_end_to_end() -> list[torch.Tensor]:
        weights = weight_tensors
        if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") == "1":
            from miles.utils.nvfp4_fake_qat import maybe_fake_quantize_nvfp4_weight_tensors

            weights = maybe_fake_quantize_nvfp4_weight_tensors(weights)
        return weights

    def fused_precomputed_amax() -> list[torch.Tensor]:
        return [fused_nvfp4_qdq(w, amax, config) for w, amax in zip(weight_tensors, amax_tensors, strict=False)]

    # Warm native TE and every compiled CuTe specialization before timing.
    # Both primary closures retain their production QAT autograd wrappers.
    expected = naive()
    actual = fused_end_to_end()
    for weight_idx, (expected_weight, actual_weight) in enumerate(zip(expected, actual, strict=False)):
        if not torch.equal(expected_weight.detach().view(torch.uint16), actual_weight.detach().view(torch.uint16)):
            mismatch_count = torch.count_nonzero(
                expected_weight.detach().view(torch.uint16) != actual_weight.detach().view(torch.uint16)
            ).item()
            raise AssertionError(f"fused QDQ weight {weight_idx} differs from TE in " f"{mismatch_count} elements")
    del expected, actual
    fused_precomputed_amax()
    torch.cuda.synchronize()

    naive_us = []
    fused_e2e_us = []
    precomputed_amax_us = []
    for _ in range(repeats):
        # Interleaved A/B/A guards the primary comparison against clock drift
        # and neighboring-workload interference on a shared GPU node.
        naive_us.append(_median_us(naive, min_run_time))
        fused_e2e_us.append(_median_us(fused_end_to_end, min_run_time))
        naive_us.append(_median_us(naive, min_run_time))
        precomputed_amax_us.append(_median_us(fused_precomputed_amax, min_run_time))
    return naive_us, fused_e2e_us, precomputed_amax_us


def _format_samples(samples: list[float]) -> str:
    raw = ", ".join(f"{sample:.3f}" for sample in samples)
    return f"{_sample_median(samples):.3f} [{raw}]"


def _sample_median(samples: list[float]) -> float:
    ordered = sorted(samples)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        block_axis, rows = value.lower().split("x", maxsplit=1)
        shape = int(block_axis), int(rows)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shapes must use BLOCK_AXISxROWS, for example 6144x4096") from exc
    if shape[0] <= 0 or shape[1] <= 0 or shape[0] % 16 != 0:
        raise argparse.ArgumentTypeError("BLOCK_AXIS and ROWS must be positive and BLOCK_AXIS must be divisible by 16")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--num-weights", type=int, default=DEFAULT_NUM_WEIGHTS)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--image",
        default=os.getenv("MILES_IMAGE", "unknown"),
        help="Explicit Miles image tag or digest for the output metadata.",
    )
    parser.add_argument(
        "--miles-commit",
        default=os.getenv("MILES_COMMIT", "unknown"),
        help="Tested Miles commit for output metadata.",
    )
    parser.add_argument(
        "--megatron-commit",
        default=os.getenv("MEGATRON_COMMIT", "unknown"),
        help="Tested Megatron commit for output metadata.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.min_run_time <= 0.0:
        raise ValueError("--min-run-time must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.num_weights <= 0:
        raise ValueError("--num-weights must be positive")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    shapes = args.shapes or DEFAULT_LOGICAL_SHAPES
    dtypes = {
        "bf16": [torch.bfloat16],
        "fp16": [torch.float16],
        "both": [torch.bfloat16, torch.float16],
    }[args.dtype]

    print(f"image={args.image}")
    print(f"miles_commit={args.miles_commit}")
    print(f"megatron_commit={args.megatron_commit}")
    print(f"gpu={torch.cuda.get_device_name()}")
    print(f"compute_capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformer_engine={transformer_engine.__version__}")
    print(f"cutlass_dsl={getattr(cutlass, '__version__', 'unknown')}")
    print(f"min_run_time_s={args.min_run_time}")
    print(f"repeats={args.repeats}")
    print(f"num_weights={args.num_weights}")
    print(f"num_gemms={args.num_weights}")
    print("weight_storage=discrete_parameters")
    print("gradient_accumulation_fusion=false")
    print("moe_single_grouped_weight=false")
    print("fused_path=maybe_fake_quantize_nvfp4_weight_tensors")
    print("shape_contract=logical_block_axis_x_rows")
    print("tensor_layout=contiguous_[rows,block_axis]")
    for block_axis, rows in shapes:
        print(f"in_features={block_axis}")
        print(f"out_features={rows}")
        print(f"stored_weight_shape=[{rows},{block_axis}]")
    print("primary_order=naive/fused/naive per repeat")
    print("NVTE_USE_FAST_MATH=0")
    print()
    print(
        "| dtype | logical shape (block-axis x rows) | mode | "
        f"naive TE {args.num_weights}-weight loop median [A/B/A raw] (us) | "
        f"fused QAT {args.num_weights}-weight loop median [raw] (us) | "
        f"precomputed-amax {args.num_weights}-weight loop median [raw] (us) | "
        "end-to-end speedup |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    all_speedups = []
    speedups_by_dtype: dict[torch.dtype, list[float]] = {dtype: [] for dtype in dtypes}
    for dtype in dtypes:
        for shape in shapes:
            for label, config in _configs():
                naive_us, fused_e2e_us, precomputed_amax_us = _benchmark_case(
                    shape, dtype, config, args.num_weights, args.min_run_time, args.repeats
                )
                speedup = _sample_median(naive_us) / _sample_median(fused_e2e_us)
                all_speedups.append(speedup)
                speedups_by_dtype[dtype].append(speedup)
                print(
                    f"| {str(dtype).removeprefix('torch.')} | {shape[0]}x{shape[1]} | "
                    f"{label} | {_format_samples(naive_us)} | "
                    f"{_format_samples(fused_e2e_us)} | "
                    f"{_format_samples(precomputed_amax_us)} | "
                    f"{speedup:.3f}x |",
                    flush=True,
                )
    print()
    print(f"geomean_speedup={geometric_mean(all_speedups):.3f}x")
    for dtype, speedups in speedups_by_dtype.items():
        print(f"geomean_speedup_{str(dtype).removeprefix('torch.')}=" f"{geometric_mean(speedups):.3f}x")


if __name__ == "__main__":
    main()
