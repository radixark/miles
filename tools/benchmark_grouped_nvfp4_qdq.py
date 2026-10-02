"""Compare packed QDQ with the existing per-expert fused QDQ loop on SM10x.

Run from the repository root with PYTHONPATH=. This measures QDQ forward, not
training throughput. Both paths read the same already-packed weights; no
stacking, packing, or TE quantize/dequantize baseline is timed. The adapter
scope includes fresh per-expert amax, STE, and native TE output wrapping.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
import transformer_engine
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from miles.utils.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_grouped_nvfp4_amax,
    compute_nvfp4_amax,
    fused_grouped_nvfp4_qdq,
    fused_nvfp4_qdq,
)
from miles.utils.nvfp4_fake_qat import maybe_fake_quantize_nvfp4_weight_tensors


def _timer(fn, mode, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    retained = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            retained = fn()
        run = graph.replay
    else:
        run = fn

    def measure():
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            output = run()
        end.record()
        end.synchronize()
        # Keep graph outputs alive until measurement is complete.
        del output
        _ = retained
        return start.elapsed_time(end) * 1000 / iterations

    return measure


def _set_config(config):
    os.environ["OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"] = "1"
    os.environ["NVTE_USE_FAST_MATH"] = "0"
    os.environ["NVTE_NVFP4_4OVER6"] = "all" if config.use_4over6 else "none"
    os.environ["NVTE_NVFP4_4OVER6_E4M3_USE_256"] = "all" if config.e4m3_max == 256 else "none"
    os.environ["NVTE_NVFP4_4OVER6_ERR_MODE"] = config.error_mode.name
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = str(int(config.error_use_fast_math))


def _benchmark_case(shape, config, iterations, repeats):
    _set_config(config)
    storage = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    scales = 2.0 ** ((torch.arange(shape[0], device="cuda") % 5) - 2)
    storage.mul_(scales.view(-1, 1, 1))
    packed = torch.nn.Parameter(
        GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=shape[0],
            tensor_shape=shape[1:],
            rowwise_data=storage,
        )
    )
    weights = [torch.nn.Parameter(w) for w in storage.unbind(0)]
    amaxes = compute_grouped_nvfp4_amax(storage)
    scalar_amaxes = [compute_nvfp4_amax(w) for w in weights]

    def loop_kernel():
        return [fused_nvfp4_qdq(w, a, config) for w, a in zip(weights, scalar_amaxes, strict=True)]

    def grouped_kernel():
        return fused_grouped_nvfp4_qdq(storage, amaxes, config)

    def loop_adapter():
        return maybe_fake_quantize_nvfp4_weight_tensors(weights)

    def grouped_adapter():
        return maybe_fake_quantize_nvfp4_weight_tensors([packed])[0]

    for scope, loop, grouped in (("kernel", loop_kernel, grouped_kernel), ("adapter", loop_adapter, grouped_adapter)):
        expected = torch.stack(loop())
        actual = grouped()
        if scope == "adapter":
            actual = actual.rowwise_data.view(shape)
        if not torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)):
            raise AssertionError(f"Mismatch for {shape}, {config}, {scope}")
        del actual, expected
        for mode in ("eager", "graph"):
            loop_timer, grouped_timer = (_timer(fn, mode, iterations) for fn in (loop, grouped))
            loop_us, grouped_us = [], []
            for repeat in range(repeats):
                if repeat % 2:
                    grouped_us.append(grouped_timer())
                    loop_us.append(loop_timer())
                else:
                    loop_us.append(loop_timer())
                    grouped_us.append(grouped_timer())
            yield {
                "shape": shape,
                "config": asdict(config),
                "scope": scope,
                "mode": mode,
                "loop_us": statistics.median(loop_us),
                "grouped_us": statistics.median(grouped_us),
                "speedup": statistics.median(loop_us) / statistics.median(grouped_us),
                "loop_samples_us": loop_us,
                "grouped_samples_us": grouped_us,
            }
            del loop_timer, grouped_timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) < 1:
        parser.error("iterations and repeats must be positive")
    configs = [NVFP4QDQConfig()]
    configs.extend(
        NVFP4QDQConfig(use_4over6=True, e4m3_max=maximum, error_mode=mode, error_use_fast_math=fast)
        for mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE)
        for maximum in (448, 256)
        for fast in (False, True)
    )
    torch.manual_seed(42)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "te": transformer_engine.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "dtype": "bf16",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        for rows, columns in ((128, 1024), (4096, 6144)):
            for groups in (1, 3, 8):
                for config in configs:
                    for result in _benchmark_case((groups, rows, columns), config, args.iterations, args.repeats):
                        line = json.dumps(result)
                        print(line, flush=True)
                        output.write(line + "\n")
                        output.flush()


if __name__ == "__main__":
    main()
