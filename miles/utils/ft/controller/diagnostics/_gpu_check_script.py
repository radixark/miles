"""Standalone GPU health-check script, executed as a subprocess.

Runs pynvml extended checks and matmul correctness verification on all
visible GPUs, then prints a JSON array of per-GPU results to stdout.

Usage::

    python -m miles.utils.ft.controller.diagnostics._gpu_check_script

The caller (GpuDiagnostic) launches this via asyncio.create_subprocess_exec
so that pynvml init/shutdown and torch computation happen in an isolated
process and never block the NodeAgent event loop (see 3-discussions.md #48).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NvmlCheckResult:
    ecc_errors_uncorrectable: int
    retired_pages_count: int
    power_state_abnormal: bool
    row_remap_failure: bool


@dataclass
class GpuCheckResult:
    gpu_index: int
    passed: bool
    ecc_errors_uncorrectable: int
    retired_pages_count: int
    power_state_abnormal: bool
    row_remap_failure: bool
    matmul_passed: bool
    details: str


_ABNORMAL_POWER_STATES = frozenset({8, 15})


def _check_nvml(handle: object) -> NvmlCheckResult:
    """Run pynvml extended checks on a single GPU handle."""
    import pynvml

    ecc_uncorrectable = pynvml.nvmlDeviceGetTotalEccErrors(
        handle,
        pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
        pynvml.NVML_VOLATILE_ECC_COUNTER_TYPE,
    )

    retired_double_bit = pynvml.nvmlDeviceGetRetiredPages(
        handle, pynvml.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR,
    )

    power_state: int = pynvml.nvmlDeviceGetPowerState(handle)

    remap_info = pynvml.nvmlDeviceGetRemappedRows(handle)

    return NvmlCheckResult(
        ecc_errors_uncorrectable=ecc_uncorrectable,
        retired_pages_count=len(retired_double_bit),
        power_state_abnormal=power_state in _ABNORMAL_POWER_STATES,
        row_remap_failure=bool(remap_info[3]),
    )


def _generate_matmul_reference() -> tuple[Any, Any, Any]:
    """Generate deterministic input matrices and CPU reference result.

    Returns (a_fp16_cpu, b_fp16_cpu, expected_fp32_cpu).
    Called once; reused for all GPUs.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(42)

    size = 1024
    a_fp32 = torch.randn(size, size, generator=generator)
    b_fp32 = torch.randn(size, size, generator=generator)

    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()
    expected = (a_fp16 @ b_fp16).float()

    return a_fp16, b_fp16, expected


def _check_matmul(
    gpu_index: int,
    a_fp16: Any,
    b_fp16: Any,
    expected: Any,
) -> bool:
    """Run matmul on a single GPU and compare against pre-computed reference.

    Returns True if the result matches within tolerance.
    """
    import torch

    device = torch.device(f"cuda:{gpu_index}")
    actual = (a_fp16.to(device) @ b_fp16.to(device)).float().cpu()

    return bool(torch.allclose(actual, expected, atol=1e-2, rtol=1e-3))


def _check_single_gpu(
    gpu_index: int,
    handle: object,
    matmul_ref: tuple[Any, Any, Any],
) -> GpuCheckResult:
    """Run all checks on one GPU and produce a GpuCheckResult."""
    failures: list[str] = []

    nvml = _check_nvml(handle)

    if nvml.ecc_errors_uncorrectable > 0:
        failures.append(
            f"uncorrectable ECC errors: {nvml.ecc_errors_uncorrectable}"
        )
    if nvml.retired_pages_count > 0:
        failures.append(f"retired pages: {nvml.retired_pages_count}")
    if nvml.power_state_abnormal:
        failures.append("abnormal power state")
    if nvml.row_remap_failure:
        failures.append("row remap failure")

    matmul_passed = _check_matmul(gpu_index, *matmul_ref)
    if not matmul_passed:
        failures.append("matmul mismatch")

    passed = len(failures) == 0
    details = "; ".join(failures) if failures else "all checks passed"

    return GpuCheckResult(
        gpu_index=gpu_index,
        passed=passed,
        ecc_errors_uncorrectable=nvml.ecc_errors_uncorrectable,
        retired_pages_count=nvml.retired_pages_count,
        power_state_abnormal=nvml.power_state_abnormal,
        row_remap_failure=nvml.row_remap_failure,
        matmul_passed=matmul_passed,
        details=details,
    )


def main() -> None:
    import pynvml

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        matmul_ref = _generate_matmul_reference()

        results: list[GpuCheckResult] = []
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                result = _check_single_gpu(i, handle, matmul_ref=matmul_ref)
            except Exception as exc:
                result = GpuCheckResult(
                    gpu_index=i,
                    passed=False,
                    ecc_errors_uncorrectable=0,
                    retired_pages_count=0,
                    power_state_abnormal=False,
                    row_remap_failure=False,
                    matmul_passed=False,
                    details=f"check failed: {exc}",
                )
            results.append(result)
    finally:
        pynvml.nvmlShutdown()

    json.dump([asdict(r) for r in results], sys.stdout)


if __name__ == "__main__":
    main()
